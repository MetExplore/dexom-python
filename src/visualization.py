
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans, MeanShift


def pca_analysis(X):
    X2 = pd.read_csv("par_1_obj001_an/all_sol.csv", index_col=0)
    X3 = pd.read_csv("par_1_newobjtol_an/full_rxn_sol.csv", index_col=0)

    pca = PCA(n_components=2)
    X_t = pd.concat([df, X2, X3])

    pca.fit(X_t)

    comp1 = pca.transform(X)
    x = [c[0] for c in comp1]
    y = [c[1] for c in comp1]

    comp2 = pca.transform(X2)
    x2 = [c[0] for c in comp2]
    y2 = [c[1] for c in comp2]

    comp3 = pca.transform(X3)
    x3 = [c[0] for c in comp3]
    y3 = [c[1] for c in comp3]

    # clus = KMeans(n_clusters=4) #, init=np.array([(-4, 0), (-1, 0), (2, 2), (6, 3)]))
    # clus = AgglomerativeClustering(n_clusters=4, linkage='single')
    # clus = SpectralClustering(n_clusters=4)
    clus = MeanShift()
    clus.fit(np.concatenate((comp1, comp2, comp3)))

    clus1 = clus.predict(comp1)
    clus2 = clus.predict(comp2)
    clus3 = clus.predict(comp3)

    clusters = np.load("par_1_newobjtol_an/AggloClusters_singlelink.npy")

    colors = np.array(['g', 'b', 'c', 'y'])
    plt.clf()
    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=20)
    plt.ylabel('Principal Component 2', fontsize=20)
    plt.title("PCA of enumeration solutions with different methods", fontsize=20)
    plt.scatter(x, y, color=colors[clus1], label="DEXOM 10% tolerance")
    plt.scatter(x2, y2, color=colors[clus2], label="DEXOM 1% tolerance")
    plt.scatter(x3, y3, color=colors[clus3], label="rxn-enum")
    # plt.legend()
    plt.show()

    br = pd.DataFrame(pca.components_, index=["pc1", "pc2"], columns=list(range(7785)))
    br.T.to_csv("importance_of_features_total.csv")

    c = [idx for idx, e in enumerate(clus3) if e == 2]
    temp = X3.T[c].cumsum().loc["7784"]
    print("Cluster %i range: %i - %i, mean: %i, number of solutions: %i" %
          (3, min(temp), max(temp), sum(temp)/len(temp), len(temp)))
    (X3.T[c].cumsum(axis=1)[c[-1]]/len(c)).to_csv("cluster%i_rxnfrequency.csv" % 3)

    #
    # cluster_index = []
    # for i in range(max(clusters)+1):
    #     cluster_index.append([])
    #     cluster_index[i] = [idx for idx, e in enumerate(clusters) if e == i]
    #
    # for i, c in enumerate(cluster_index):
    #     temp = df.T[c].cumsum().loc["7784"]
    #     print("Cluster %i range: %i - %i, mean: %i, number of solutions: %i" %
    #           (i+1, min(temp), max(temp), sum(temp)/len(temp), len(temp)))
    #     (df.T[c].cumsum(axis=1)[c[-1]]/len(c)).to_csv("cluster%i_rxnfrequency.csv" % (i+1))

    return pca, clusters


def plot_Fischer_pathways(filename, sublist):
    df = pd.read_csv(filename, index_col=0)
    df.columns = sublist
    plt.clf()
    plt.figure(figsize=(20, 30))
    fig, ax = plt.subplots(figsize=(10, 20))
    # fig.subplots_adjust()
    df.boxplot(vert=False)
    plt.plot(list(df.values)[0], ax.get_yticks(), 'r.')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    plt.title("Overrepresentation analysis of active reactions per pathway", fontsize=23, loc='right', pad='20')
    plt.xlabel('-log10 p-value', fontsize=15)
    plt.show()
    return df


if __name__=="__main__":

    df = pd.read_csv("par_1_newobjtol_an/all_sol.csv", index_col=0)
    df = df.drop_duplicates(ignore_index=True)
    pca, clus = pca_analysis(df)

    # with open("recon2_2/recon2v2_subsystems_list.txt", "r") as file:
    #     sublist = file.read().split(";")
    # sublist = [s.replace("metabolism", "m.").replace(" Metabolism", "m.") for s in sublist]
    # pvals = plot_Fischer_pathways("par_1_obj001_an/newobj_Fischer.csv", sublist=sublist)


