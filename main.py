from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def scale(dataFrame):
    result = dataFrame.copy()
    for feature_name in dataFrame.columns:
        if feature_name != 'Class':
            max_value = dataFrame[feature_name].max()
            min_value = dataFrame[feature_name].min()
            result[feature_name] = (
                dataFrame[feature_name] - min_value) / (max_value - min_value)
        else:
            result[feature_name] = dataFrame[feature_name]
    return result


def show_pca(dataFrame):
    dataFrame = dataFrame.copy()
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(dataFrame[dataFrame.columns[:-1]].values)
    print(pca_result)

    dataFrame['pca-one'] = pca_result[:, 0]
    dataFrame['pca-two'] = pca_result[:, 1]

    plt.figure(figsize=(3, 2))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="Class",
        palette=sns.color_palette("hls", 7),
        data=dataFrame,
        legend="full",
        alpha=1
    )

    plt.show()


def show_tsne(dataFrame):
    dataFrame = dataFrame.copy()
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=5000)
    tsne_results = tsne.fit_transform(dataFrame[dataFrame.columns[:-1]].values)

    dataFrame['tsne-one'] = tsne_results[:, 0]
    dataFrame['tsne-two'] = tsne_results[:, 1]
    plt.figure(figsize=(3, 2))
    sns.scatterplot(
        x="tsne-one", y="tsne-two",
        hue="Class",
        palette=sns.color_palette("hls", 7),
        data=dataFrame,
        legend="full",
        alpha=1
    )

    plt.show()


while(True):
    df = pd.read_csv("Dry_Bean.txt", sep='\t')

    print("Do you want scale data frame? ([y]/n)", end='')
    answer = input()
    if not answer or answer.lower() == 'y':
        df = scale(df)
        print("\t scaled...")

    print("Do you want show pca? (y/[n])", end='')
    answer = input()
    if answer.lower() == 'y':
        show_pca(df)

    print("Do you want show t-SNE? (y/[n])", end='')
    answer = input()
    if answer.lower() == 'y':
        show_tsne(df)

    print("Do you want quit? (y/[n])", end='')
    answer = input()
    if answer.lower() == 'y':
        break
