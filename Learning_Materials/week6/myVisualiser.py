import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def cluster_and_visualise(datafilename, K, featureNames):
    ##your code goes here


# load the data
    data = np.genfromtxt(datafilename,delimiter=',')

# create the cluster labels
    clusterModel = KMeans(n_clusters=K)
    clusterModel.fit(data)
    cluster_ids = clusterModel.predict(data)

    f = data.shape[1]
    fig, ax = plt.subplots(f,f,figsize=(12,12))

    plt.set_cmap('jet')
    for feature1 in range(f):
        ax[feature1,0].set_ylabel( featureNames[feature1])
        ax[0,feature1].set_xlabel( featureNames[feature1])
        ax[0,feature1].xaxis.set_label_position('top') 
        for feature2 in range(f):
            xdata = data[:,feature1]
            ydata = data[:,feature2]
            if (feature1!=feature2):
                ax[feature1, feature2].scatter(xdata,ydata,c=cluster_ids)
            else:
                # if the features are the same, then create a histogram with different colours for different clusters
                for cluster_id in range(K):
                    ax[feature1,feature2].hist(xdata[cluster_ids==cluster_id],alpha=0.5)

    fig.suptitle("Fruit Data",fontsize=16,y=0.925)

    
    

    return fig,ax
    
