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

download = True

pairs = ['DEXJPUS', 'DEXCHUS', 'DEXUSEU', 'DEXMXUS', 'DEXUSUK', 'DEXCAUS',
         'DEXUSAL', 'DEXBZUS', 'DEXKOUS', 'DEXSZUS', 'DEXINUS', 'DTWEXO',
         'DEXTHUS', 'DEXHKUS', 'DEXNOUS', 'DEXTAUS', 'DEXSFUS', 'DEXSDUS',
         'DEXSIUS', 'DEXMAUS', 'DEXSLUS', 'DEXDNUS', 'DEXUSNZ', 'DEXVZUS']

start = datetime.datetime(2011, 4, 20)
end = datetime.datetime(2015, 3, 23)

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


# Clustering with Kmeans
model = KMeans(n_clusters=8).fit(dataset)
TScL4 = dict(zip(datasetnames, model.labels_))
for i in range(min(model.labels_),max(model.labels_)+1):
    print "[Kmeans] Cluster", i
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k in [key for key,val in TScL4.iteritems() if TScL4[key]==i]:
        ax.plot(TSrec[k][1],label=k+' L4 ')
    plt.legend(prop={'size':6},loc=0)
    plt.savefig('kmeans-forexCluster'+str(i)+'.pdf',edgecolor='b', format='pdf')
