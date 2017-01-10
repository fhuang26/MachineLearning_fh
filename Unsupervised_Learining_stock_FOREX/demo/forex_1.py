import random
import pandas.io.data as web
import datetime
import wfvg
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import pickle

download = False

pairs = ['DEXJPUS', 'DEXCHUS', 'DEXUSEU', 'DEXMXUS', 'DEXUSUK', 'DEXCAUS',
         'DEXUSAL', 'DEXBZUS', 'DEXKOUS', 'DEXSZUS', 'DEXINUS', 'DTWEXO',
         'DEXTHUS', 'DEXHKUS', 'DEXNOUS', 'DEXTAUS', 'DEXSFUS', 'DEXSDUS',
         'DEXSIUS', 'DEXMAUS', 'DEXSLUS', 'DEXDNUS', 'DEXUSNZ', 'DEXVZUS']

start = datetime.datetime(2011, 4, 20)
end = datetime.datetime(2015, 3, 23)

if download:
    TS = {}
    for ticker in pairs:
        key = "TS."+ticker
        L = web.DataReader(ticker, 'fred', start, end).interpolate()
        L2 = list(L[ticker].values)
        TS[key] = L2

    pickle.dump(TS,open("savedTSforex.dat",'wb'))
else:
    # Retrieve time series from file
    TS = pickle.load(open("savedTSforex.dat","rb"))

# Generate normalized time series
TSpct = {}
for TSname in TS:
    TSpct[TSname] = [0]
    for i in range(1,len(TS[TSname])):
        TSpct[TSname].append(TS[TSname][i]/TS[TSname][0])

# L = web.DataReader('DEXJPUS', 'fred', start, end).interpolate()
# L2 = list(L['DEXJPUS'].values)


# Each entry TSwfv[TSname] is an array of vector coefficients
# Each item TSwfv[TSname][i] in the array is a feature vector (FV)
# Input to the clustering algorithm is all FVs with the same value of i
# Example: TSwfv['TS.1'][1], TSwfv['TS.2'][1], TSwfv['TS.3'][1], ...

TSwfv,TSsse,TSrec = wfvg.generate_feature_vectors(TSpct)

# Retrieve the L4 coefficients for all time series
dataset = [ TSwfv[k][1] for k,v in TSwfv.iteritems() ]
datasetnames = [k for k,v in TSwfv.iteritems()]

np.random.seed(0)


clustering_names = [
    'MiniBatchKMeans', 'AffinityPropagation',
    'SpectralClustering', 'Ward', 'AgglomerativeClustering',
    'Birch']

X = np.array(dataset)
# estimate bandwidth for mean shift
bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
# connectivity matrix for structured Ward
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)
# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)


# create clustering estimators
two_means = cluster.MiniBatchKMeans(n_clusters=8)
ward = cluster.AgglomerativeClustering(n_clusters=8, linkage='ward',
                                        connectivity=connectivity)
spectral = cluster.SpectralClustering(n_clusters=8,
                                        eigen_solver='arpack',
                                        affinity="nearest_neighbors")
# dbscan = cluster.DBSCAN(eps=.2)
affinity_propagation = cluster.AffinityPropagation(damping=.9,
                                                    preference=-200)

average_linkage = cluster.AgglomerativeClustering(
    linkage="average", affinity="cityblock", n_clusters=8,
    connectivity=connectivity)

birch = cluster.Birch(n_clusters=8)
clustering_algorithms = [
    two_means, affinity_propagation, spectral, ward, average_linkage,
    birch]


plt.figure(figsize=(len(clustering_names) * 2 + 3, 9.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)
plot_num = 1
col = 0
for name, algorithm in zip(clustering_names, clustering_algorithms):
    col += 1
    algorithm.fit(X)
    ds = dict(zip(datasetnames, algorithm.labels_))
    # plot
    row = 0
    for i in range(0,8):
        row += 1
        plt.subplot(8,len(clustering_algorithms),(row-1)*len(clustering_algorithms)+col)
        if i == 0:
            plt.title(name,size=8)
        for k in [key for key,val in ds.iteritems() if ds[key]==i]:
            plt.plot(TSrec[k][1],label=k+' L4 ',hold=True)
        plt.legend(prop={'size':4},loc=0)
        plt.xticks(())
        plt.yticks(())
        plot_num += 1

plt.savefig('ForexCompleteGrid.pdf',edgecolor='b', format='pdf')
