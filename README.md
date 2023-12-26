# InterpreTabNet

This repo is part of the code for the paper: [Stable and Interpretable Deep Learning for Tabular Data: Introducing InterpreTabNet with the Novel InterpreStability Metric](https://arxiv.org/pdf/2310.02870.pdf), including:

- Exploratory Data Analysis (EDA)
- Pipeline construction
- Modeling
- Package scripts
- The new evaluation metric.
  
It also contains the scripts of InterpreTabNet which has been modified based on the `pytorch-tabnet` package. Below is a figure shows the workflow of the paper:

![image](https://drive.google.com/uc?export=view&id=17iKeZ-5jOis9hCMIu9_nGtzSfZMjmdOM)
<small><i>**Fig.4** The architecture of InterpreTabNet and primary innovations in this work. (1) The center part is the main framework of InterpreTabNet, consisting of three decision steps and a decoder with three decoding steps. Each output of one decision step is aggregated with the following one and ultimately passed to the decoder. The decoder gives the final output, selected features, and feature importances. (2) At the top left corner is each decision step in the encoder, where we propose the Multilayer Perceptron-Attentive Transformer (MLP-Attentive Transformer) and Multi-branch WLU. (3) The bottom left corner is the detailed architecture of the MLP-Attentive Transformer. (4) The entire right part visualizes the scheme of our novel evaluation metric, InterpreStability. Heatmaps represent feature correlations, whose elements are groups of feature importance. The little squares denote specific InterpreStability values after the calculation. (5) The middle bottom and top are the input and output of this paper’s classification task, respectively. The input is the tabular data, while the output is the classification result.</i></small>

Additionally, since the paper is a preprinted one, we have been improving the designs and experiments, the repo is more likely to be a draft to show our ideas of what is going on. Meanwhile, we added some data analysis to make the process more rigorous, so the experimental results in the repo may be a little different from what has been written in the paper.

More details can be found in the paper, and we hope the repo can be completed soon!
