# Transfer learning

This repository contains code for transfer learning using pretrained Inception network on CIFAR-10 dataset.

## Installation

### Required packages
1. Tensorflow: https://www.tensorflow.org/install/install_linux#InstallingAnaconda
2. Tensorpack - used for data loading and preprocessing:  
`pip install -U git+https://github.com/ppwwyyxx/tensorpack.git`
3. Skiimage - used for extracting HOG features from images:  
`pip install scikit-image`
4. GPyOpt - used for Bayesian Optimisation of hyper-parameters:  
`pip install gpyopt`



### Pretrained CNN codes
Pretrained CNN codes for CIFAR-100 are available here: 
https://drive.google.com/open?id=0B6fInPVjwoO1d0kzUTY1OVBtbDg

### Testing
To test the setup run:  
`python -m unittest tests`

## Scripts

### CNN codes extraction

Extract CNN codes for the test examples:  
`python extract.py --mode test`  

Extract CNN codes for the training examples:  
`python extract.py --mode train`  

### Visualisations

Examples from test dataset, output stored in examples.png:  
```python plotting.py --mode plot_examples```

Examples from test dataset and their HOG features, outputs stored in examples.png and examples_hog.png:  
```python plotting.py --mode plot_examples```

T-SNE embedding of test examples with class segmentation and original images:  
```python plotting.py --mode plot_cnn```

### Classification

Available classification methods include: softmax_raw, svm_raw, svm_hog, svm_hog_kern, svm_cnn, svm_cnn_kern.  

Example:  
`python classify.py --mode svm_cnn`

### Bayesian Optimisation

For tunning the hyperparamters. Example:  
`python bayes_opt.py --mode linear`
