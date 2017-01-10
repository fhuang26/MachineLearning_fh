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

if __name__ == '__main__':
    TS = {}
    TSpct = {}
    symbols = ['YHOO','GOOGL','AAPL','MSFT','BIDU','IBM','EBAY','ORCL','CSCO',
    'SAP','VZ','T','CMCSA','AMX','QCOM','NOK','AMZN','WMT','COST','TGT','CVX',
    'TOT','BP','XOM','E','COP','APA','GS','MS','BK','CS','SMFG','DB','RY','CS',
    'BCS','SAN','BNPQY','NKE','DECK','PCLN','EMC','INTC','AMD','NVDA','TXN',
    'BRCM','ADI','WFM','TFM','INFN','CIEN','CSC','TMO','BSX','TIVO','DISH',
    'SATS','LORL','ORAN','IMASF','IRDM','HRS','GD','BA','LMT','NOC','RTN',
    'TXT','ERJ','UTX','SPR','BDRBF','AAL','DAL','HA','UAL','LUV','JBLU','ALGT',
    'RJET','RCL','CCL','DIS','CBS','FOXA','QVCA','DWA','VIAB','TM',
    'TWX','DISCA','SNI','MSG','PG','ENR','HRG','SPB','KMB','TSLA']
    stock_split = {}
    stock_split['GOOGL'] = ('2014-04-02',2)
    stock_split['AAPL'] = ('2014-06-06',7)
    stock_split['AMX'] = ('2011-06-30',2)
    stock_split['NKE'] = ('2012-12-25',2)
    stock_split['WFM'] = ('2013-05-29',2)

    start = datetime.datetime(2011, 4, 20)
    end = datetime.datetime(2015, 5, 16)

    if download:
        for ticker in symbols:
            key = "TS."+ticker
            print "Retrieving {0} ...".format(ticker)
            L = web.DataReader(ticker, 'yahoo', start, end)
            # Correct stock splits
            if ticker in stock_split:
                L.loc[:stock_split[ticker][0],'Close'] = L[:stock_split[ticker][0]]['Close']/stock_split[ticker][1]
            TS[key] = list(L['Close'].values)
        # Save time series to file
        pickle.dump(TS,open("savedTS.dat",'wb'))
    else:
        # Retrieve time series from file
        TS = pickle.load(open("savedTS.dat","rb"))

    # Generate normalized time series
    for TSname in TS:
        TSpct[TSname] = [0]
        for i in range(1,len(TS[TSname])):
            TSpct[TSname].append(TS[TSname][i]/TS[TSname][0])


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
    plt.savefig('CompleteGrid.pdf',edgecolor='b', format='pdf')





    # # Clustering with Kmeans
    # model = KMeans(n_clusters=8).fit(dataset)
    # TScL4 = dict(zip(datasetnames, model.labels_))
    # for i in range(min(model.labels_),max(model.labels_)+1):
    #     print "[Kmeans] Cluster", i
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     for k in [key for key,val in TScL4.iteritems() if TScL4[key]==i]:
    #         ax.plot(TSrec[k][1],label=k+' L4 ')
    #     plt.legend(prop={'size':6},loc=0)
    #     plt.savefig('Cluster'+str(i)+'-kmeans.pdf',edgecolor='b', format='pdf')

    # # Clustering with Spectral Clustering
    # spectral = cluster.SpectralClustering(n_clusters=8,
    #                                       eigen_solver='arpack',
    #                                       affinity="nearest_neighbors")
    # model = spectral.fit(dataset)
    # TSspecdenL4 = dict(zip(datasetnames, model.labels_))
    # print TSspecdenL4
    # for i in range(min(model.labels_),max(model.labels_)+1):
    #     print "[Spectral density] Cluster", i
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     for k in [key for key,val in TSspecdenL4.iteritems() if TSspecdenL4[key]==i]:
    #         ax.plot(TSrec[k][1],label=k+' L4 ')
    #     plt.legend(prop={'size':6},loc=0)
    #     plt.savefig('Cluster'+str(i)+'-sd.pdf',edgecolor='b', format='pdf')
