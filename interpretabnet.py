import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, seed=1e-5):
        self.seed = seed

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        return np.log(X + self.seed)


class InterpreTabNet:
    def __init__(self, path: str = 'data/log_data.csv') -> None:

        df = pd.read_excel("default of credit card clients.xls")
        df.drop_duplicates(inplace=True)
        df.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)

        dataset = df.copy()

        X = dataset.drop(['ID', 'default payment next month'], axis=1)
        y = dataset['default payment next month']

        # trainval : test = 8 : 2
        X_train_val, self.X_test, y_train_val, self.y_test = \
            train_test_split(X, y, test_size=0.2,
                             stratify=y, random_state=42)  # 0.2 1

        # train : val = 7 : 1
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(X_train_val, y_train_val,
                             test_size=1/8, stratify=y_train_val,
                             random_state=42)  # 1/8  2

        num_features = \
            self.X_train.select_dtypes(include=['int64']).columns.tolist()
        features = [feature for feature in num_features
                    if feature not in ['PAY_2', 'PAY_3', 'PAY_4', 'PAY_5',
                                       'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                                       'BILL_AMT4', 'BILL_AMT5']]
        self.to_log = ['LIMIT_BAL']
        self.no_log = [feature for feature in features
                       if feature not in self.to_log]

        log_pipeline = make_pipeline(SimpleImputer(),
                                     LogTransformer(),
                                     StandardScaler())
        num_pipeline = make_pipeline(SimpleImputer(), StandardScaler())
        self.final_pipeline = ColumnTransformer([('log_num',
                                                log_pipeline,
                                                self.to_log),
                                                ('num',
                                                num_pipeline,
                                                self.no_log)])

        self.X_train_prepared = self.final_pipeline.fit_transform(self.X_train)
        self.X_val_prepared = self.final_pipeline.transform(self.X_val)

        self.model = make_pipeline(self.final_pipeline,
                                   TabNetClassifierWrapper(
                                       self.X_val_prepared,
                                       self.y_val,
                                       optimizer_fn=torch.optim.SGD,
                                       optimizer_params=dict(lr=1e-1, momentum=0.938, weight_decay=1e-4),
                                       scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                                       scheduler_params={"mode": "min", "factor": 0.1, "patience": 5},
                                       mask_type='entmax',
                                       verbose=10,
                                   ))

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    # def x_test_score(self) -> np.float64:

    #     return roc_auc_score(self.y_test, self.model.predict(self.X_test))

    def get_Xs(self) -> (pd.DataFrame, pd.DataFrame):

        return self.X_train, self.X_val, self.X_test

    def get_ys(self) -> (pd.DataFrame, pd.DataFrame):

        return self.y_train, self.y_val, self.y_test

    # def predict(self, path_to_new_file: str = 'data/new_data.csv') -> np.array:
    #     '''Uses the trained algorithm to predict and return the predicted
    #     labels on an unseen file.
    #     The default file is the unknown_data.csv file in your data folder.

    #     Return a numpy array (the default for the "predict()" function of
    #     sklearn estimator)'''

    #     if path_to_new_file is not None:
    #         new_data = pd.read_csv(path_to_new_file)
    #         new_data.drop_duplicates(inplace=True)
    #     else:
    #         new_data = self.X_test
    #     return self.model.predict(new_data)

    def get_model(self) -> Pipeline:
        '''returns the entire trained pipeline, i.e. your model.
        This will include the data preprocessor and the final estimator.'''

        return self.model


class TabNetClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, X_val_prepared, y_val, optimizer_fn=torch.optim.SGD, optimizer_params=None, scheduler_fn=None, scheduler_params=None, mask_type='entmax', verbose=10):
        if optimizer_params is None:
            optimizer_params = dict(lr=1e-1, momentum=0.938, weight_decay=1e-4)
        if scheduler_params is None:
            scheduler_params = {"mode": "min", "factor": 0.1, "patience": 5}

        self.X_val_prepared = X_val_prepared
        self.y_val = y_val

        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params
        self.scheduler_fn = scheduler_fn
        self.scheduler_params = scheduler_params
        self.mask_type = mask_type
        self.verbose = verbose
        self.tabnet_pretrainer = None
        self.tabnet_model = None

    def fit(self, X, y):
        torch.manual_seed(42)

        # TabNet pretrain
        self.tabnet_pretrainer = TabNetPretrainer(
            optimizer_fn=self.optimizer_fn,
            optimizer_params=self.optimizer_params,
            mask_type=self.mask_type,
            verbose=self.verbose
        )
        self.tabnet_pretrainer.fit(X_train=X,
                                   eval_set=[self.X_val_prepared],
                                   max_epochs=100,
                                   batch_size=1024,
                                   virtual_batch_size=128,
                                   patience=20)

        # TabNet fine tune
        self.tabnet_model = TabNetClassifier(
            optimizer_fn=self.optimizer_fn,
            optimizer_params=self.optimizer_params,
            scheduler_fn=self.scheduler_fn,
            scheduler_params=self.scheduler_params,
            mask_type=self.mask_type,
            verbose=self.verbose
        )
        self.tabnet_model.fit(
            X_train=X, y_train=y,
            eval_set=[(self.X_val_prepared, self.y_val)],
            max_epochs=100,
            batch_size=64,
            virtual_batch_size=32,
            from_unsupervised=self.tabnet_pretrainer,
            patience=20
        )
        return self

    def predict_proba(self, X):
        if self.tabnet_model is not None:
            return self.tabnet_model.predict_proba(X)
        else:
            raise AttributeError("TabNet model is not trained yet. \
                                 Please call fit before predict_proba.")

    def predict(self, X):
        return self.tabnet_model.predict(X)

    # def save_model(self, path):
    #     """Save TabNet model."""
    #     if self.tabnet_model is not None:
    #         self.tabnet_model.save_model(path)
    #     else:
    #         raise AttributeError("TabNet model is not trained yet.\
    #                              Please call fit before saving.")

    # @staticmethod
    # def load_model(path):
    #     """Load TabNet model."""
    #     loaded_model = TabNetClassifier()
    #     loaded_model.load_model(path)
    #     wrapper = TabNetClassifierWrapper()
    #     wrapper.tabnet_model = loaded_model
    #     return wrapper
