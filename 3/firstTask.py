import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_scores(X_principal, title):
    silhouette_score = {}
    davies_bouldin_score = {}
    calinski_harabasz_score = {}
    for i in range(1, 10):
        silhouette_score[i] = \
            metrics.silhouette_score(X_principal, KMeans(n_clusters=3, max_iter=i).fit_predict(X_principal))
        davies_bouldin_score[i] = \
            metrics.davies_bouldin_score(X_principal, KMeans(n_clusters=3, max_iter=i).fit_predict(X_principal))
        calinski_harabasz_score[i] = \
            metrics.calinski_harabasz_score(X_principal, KMeans(n_clusters=3, max_iter=i).fit_predict(X_principal))
    plt.plot(list(silhouette_score.keys()), list(silhouette_score.values()))
    plt.title(title)
    plt.xlabel("Max iterations")
    plt.ylabel("Silhouette score")
    plt.show()
    plt.plot(list(davies_bouldin_score.keys()), list(davies_bouldin_score.values()))
    plt.title(title)
    plt.xlabel("Max iterations")
    plt.ylabel("Davies-Bouldin score")
    plt.show()
    plt.plot(list(calinski_harabasz_score.keys()), list(calinski_harabasz_score.values()))
    plt.title(title)
    plt.xlabel("Max iterations")
    plt.ylabel("Calinski_Harabasz score")
    plt.show()


if __name__ == '__main__':
    for j in range(0, 2):
        X = read_csv('pluton.csv', sep=',').to_numpy()
        pca = PCA(n_components=2)
        title = 'Non-standardize'
        if j == 1:
            X = StandardScaler().fit_transform(X)
            title = 'Standardize'
        X_principal = pca.fit_transform(X)
        X_principal = pd.DataFrame(X_principal, columns=['P1', 'P2'])
        plt.scatter(X_principal['P1'], X_principal['P2'], c=KMeans(n_clusters=3).fit_predict(X_principal))
        plt.title(title)
        plt.show()
        plot_scores(X_principal, title)
