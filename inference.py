import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import LocalOutlierFactor
import argparse
import json
from density_map import load_dataset, reduce_dimensionality

parser = argparse.ArgumentParser(description='Density Map for simple dataset')
parser.add_argument('--dataset', type=str, default='iris', help='Dataset to use', choices=['iris', 'diabetes', 'wine', 'breast_cancer'])

args = parser.parse_args()

dataset_name = args.dataset
data = load_dataset(dataset_name)
X_data = StandardScaler().fit_transform(data.data)
X_pca = reduce_dimensionality(data)

# load metadata from json
with open('{}.json'.format(dataset_name + '_metadata'), 'r') as f:
    metadata = json.load(f)
pixel_coords = metadata['pixel_coords']
fig_width = metadata['fig_width']
fig_height = metadata['fig_height']

# load the saliency map
saliency_map = Image.open('{}.png'.format(dataset_name + '_WL')).convert('L')
saliency_map_np = np.array(saliency_map)
saliency_values = []
for i, data_point in enumerate(X_pca):
    x_px, y_px = pixel_coords[i]
    saliency_values.append(saliency_map_np[y_px, x_px])
saliency_values = np.array(saliency_values)
saliency_values = (saliency_values - saliency_values.min()) / (saliency_values.max() - saliency_values.min())
new_saliency_values = 1 - saliency_values

# generate outliers using LOF
def generate_outliers(mixed_data):
    lof = LocalOutlierFactor(n_neighbors=20)
    y_lof_pred = lof.fit_predict(mixed_data)
    y_lof_pred = np.where(y_lof_pred == -1, 1, 0)
    return y_lof_pred

# inferencing and plotting results
def inference(y_lof_full, y_lof_pca, markov_saliency_values, pca_data):
    fpr_full, tpr_full, thresholds_full = roc_curve(y_lof_full, markov_saliency_values)
    f1_full = 2 * (tpr_full * (1 - fpr_full)) / (tpr_full + (1 - fpr_full))
    optimized_threshold_full = thresholds_full[np.argmax(f1_full)]
    y_pred_full = markov_saliency_values > optimized_threshold_full
    acc_full = np.mean(y_pred_full == y_lof_full)

    fpr_pca, tpr_pca, thresholds_pca = roc_curve(y_lof_pca, markov_saliency_values)
    f1_pca = 2 * (tpr_pca * (1 - fpr_pca)) / (tpr_pca + (1 - fpr_pca))
    optimized_threshold_pca = thresholds_pca[np.argmax(f1_pca)]
    y_pred_pca = markov_saliency_values > optimized_threshold_pca
    acc_pca = np.mean(y_pred_pca == y_lof_pca)

    print(dataset_name)
    print('F1 Score (full data): {:.2f}'.format(np.max(f1_full)))
    print('Accuracy (Full data): {:.2f}%'.format(acc_full * 100))
    print('*'*50)
    print('F1 Score (PCA data): {:.2f}'.format(np.max(f1_pca)))
    print('Accuracy (PCA data): {:.2f}%'.format(acc_pca * 100))

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0, 0].scatter(pca_data[y_lof_full == 0, 0], pca_data[y_lof_full == 0, 1], c='g', s=50, label='Inliers')
    ax[0, 0].scatter(pca_data[y_lof_full == 1, 0], pca_data[y_lof_full == 1, 1], c='r', s=50, label='Outliers')
    ax[0, 0].set_title('Full Data LOF')
    ax[0, 0].legend()

    ax[0, 1].scatter(pca_data[y_pred_full == 0, 0], pca_data[y_pred_full == 0, 1], c='g', s=50, label='Inliers')
    ax[0, 1].scatter(pca_data[y_pred_full == 1, 0], pca_data[y_pred_full == 1, 1], c='r', s=50, label='Outliers')
    ax[0, 1].set_title('Full Data Predicted')
    ax[0, 1].legend()

    ax[1, 0].scatter(pca_data[y_lof_pca == 0, 0], pca_data[y_lof_pca == 0, 1], c='g', s=50, label='Inliers')
    ax[1, 0].scatter(pca_data[y_lof_pca == 1, 0], pca_data[y_lof_pca == 1, 1], c='r', s=50, label='Outliers')
    ax[1, 0].set_title('PCA Data LOF')
    ax[1, 0].legend()

    ax[1, 1].scatter(pca_data[y_pred_pca == 0, 0], pca_data[y_pred_pca == 0, 1], c='g', s=50, label='Inliers')
    ax[1, 1].scatter(pca_data[y_pred_pca == 1, 0], pca_data[y_pred_pca == 1, 1], c='r', s=50, label='Outliers')
    ax[1, 1].set_title('PCA Data Predicted')
    ax[1, 1].legend()

    plt.tight_layout()
    plt.savefig('{}_inference.png'.format(dataset_name))

y_lof_full = generate_outliers(X_data)
y_lof_pca = generate_outliers(X_pca)
inference(y_lof_full, y_lof_pca, new_saliency_values, X_pca)