# Importing matplotlib should happen before importing tensorpack
import argparse
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from skimage.feature import hog
from skimage import color, exposure
from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale

import scipy

from utils import *

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def plot_examples(examples, save_path, plot_hog=False):
    n_img = len(examples[0])
    f, ax = plt.subplots(n_img,n_img)
    for i in range(n_img):
        for j in range(n_img):
            ax[i][j].set_axis_off()
            if plot_hog:
                img = examples[i][j]
                img = color.rgb2gray(img)
                fd, hog_img = hog(img, orientations=8, pixels_per_cell=(8, 8), visualise=True)
                # Rescale histogram for better display
                hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 0.02))
                ax[i][j].imshow(hog_img_rescaled, cmap=plt.cm.gray)
            else:
                ax[i][j].imshow(examples[i][j])
    f.savefig(save_path)


def plot_tsne_cnncodes(save_path, max_examples = 1000):
    logging.info('Loading CNN codes...')
    data_test = np.load('cnncodes_test.npz')
    X_test, y_test, orig_test = data_test['cnn_codes'], data_test['y'], data_test['orig_imgs']
    
    n_classes = len(np.unique(y_test))
    n_examples = max_examples
    X_test = X_test[:n_examples,:]
    
    X_test_tsne = TSNE(n_components=2, random_state=0, n_iter=1000, verbose=1).fit_transform(X_test)
    
    # Group elements by the label
    groups = [[] for x in xrange(n_classes)]
    for i in range(n_examples):
        groups[y_test[i]].append(X_test_tsne[i, :])

    scatters = []
    cm = plt.cm.tab10.colors
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h']
    for i in range(n_classes):
        groups[i] = np.vstack(groups[i])
        x1 = groups[i][:, 1]
        x2 = -groups[i][:, 0]
        scatters.append(plt.scatter(x1, x2, s=10, marker=markers[i], edgecolors='none'))

    plt.legend(scatters,
           CIFAR10_CLASSES,
           scatterpoints=1,
           loc='upper center',
           ncol=5,
           fontsize=8)
    plt.axis('off')
    plt.savefig('{}_tsne_scatter.png'.format(save_path), bbox_inches='tight')

    # Plot the original images of in the t-SNE dimensions
    RES = 2000
    img = np.zeros((RES,RES,3),dtype='uint8')
    X_test_tsne  = minmax_scale(X_test_tsne)

    for i in range(n_examples):
        x1_scaled = int(X_test_tsne[i,0] * (RES - 32))
        x2_scaled = int(X_test_tsne[i,1] * (RES - 32))
        img[x1_scaled:x1_scaled+32,x2_scaled:x2_scaled+32,:] = orig_test[i]

    plt.imshow(img)

    scipy.misc.imsave('{}_tsne_orig_imgs.jpg'.format(save_path), img)


# def __main TODO: add main with command options
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    modes = ['plot_examples', 'plot_hog', 'plot_cnn']
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=modes[0], choices=modes,
                      help='Operation to perform. Possible values: {}'.format(', '.join(modes)))
    args = parser.parse_args()

    if args.mode == 'plot_examples': # Plots examples from the CIFAR-100 dataset
        examples = get_examples(10)
        plot_examples(examples, 'examples.png')

    elif args.mode == 'plot_hog': # Plots examples from CIFAR-100 and their HOG features
        examples = get_examples(10)
        plot_examples(examples, 'examples.png')
        plot_examples(examples, 'examples_hog.png', plot_hog=True)

    elif args.mode == 'plot_cnn': # Plots t-SNE embeddings of the examples
        plot_tsne_cnncodes('cnn')
    else:
        logging.warning('Uknown mode. Possible values: {}'.format(', '.join(modes)))
