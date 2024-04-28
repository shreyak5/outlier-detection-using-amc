import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
import argparse
import json

# Load dataset
def load_dataset(dataset_name):
    if dataset_name == 'iris':
        dataset = datasets.load_iris()
    elif dataset_name == 'diabetes':
        dataset = datasets.load_diabetes()
    elif dataset_name == 'wine':
        dataset = datasets.load_wine()
    elif dataset_name == 'breast_cancer':
        dataset = datasets.load_breast_cancer()
    return dataset

# reduce the dimensionality of the dataset to 2
def reduce_dimensionality(data):
    X_dataset = data.data
    X_dataset = StandardScaler().fit_transform(X_dataset)
    pca = PCA(n_components=2)
    X_pca_dataset = pca.fit_transform(X_dataset)
    return X_pca_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Density Map for simple dataset')
    parser.add_argument('--dataset', type=str, default='iris', help='Dataset to use', choices=['iris', 'diabetes', 'wine', 'breast_cancer'])

    args = parser.parse_args()

    dataset_name = args.dataset
    data = load_dataset(dataset_name)
    X_pca = reduce_dimensionality(data)

    # generate the density map
    sns.set_style('dark')
    fig, ax = plt.subplots()
    ax = sns.kdeplot(x=X_pca[:, 0], y=X_pca[:, 1], cmap='coolwarm', fill=True, levels=10, cbar=False, thresh=0.02)
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], s=10) # comment this line to remove the scatter plot while generating input for saliency map

    # plot dimensions
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    fig_width, fig_height = fig.get_size_inches()*fig.dpi

    # pixel coordinates for each data point
    pixel_coords = []
    for data_point in X_pca:
        x = (data_point[0] - xmin) / (xmax - xmin)
        y = (ymax - data_point[1]) / (ymax - ymin)
        x = int(x * fig_width)
        y = int(y * fig_height)
        pixel_coords.append((x, y))

    color = ax.collections[1].get_facecolor()
    ax.set_facecolor(color)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.gcf().set_facecolor(color)

    ax.spines['bottom'].set_color(color)
    ax.spines['top'].set_color(color)
    ax.spines['right'].set_color(color)
    ax.spines['left'].set_color(color)
    plt.savefig('{}.png'.format(dataset_name + '_density'), dpi=100)

    # save the pixel coordinates and plot dimensions in json file
    json_data = {'pixel_coords': pixel_coords, 'fig_width': fig_width, 'fig_height': fig_height}
    with open('{}.json'.format(dataset_name + '_metadata'), 'w') as f:
        json.dump(json_data, f)