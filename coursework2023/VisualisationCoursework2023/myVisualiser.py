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

# create the visualisation
    f = data.shape[1]
    fig, ax = plt.subplots(f,f,figsize=(12,12))

    plt.set_cmap('jet')

# add labels to the axes
    for feature1 in range(f):
        ax[feature1,0].set_ylabel( featureNames[feature1])
        ax[0,feature1].set_xlabel( featureNames[feature1])
        ax[0,feature1].xaxis.set_label_position('top') 
        # add data to the axes
        for feature2 in range(f):
            xdata = data[:,feature1]
            ydata = data[:,feature2]

            # if the features are different, scatter plot
            if (feature1!=feature2):
                ax[feature1, feature2].scatter(xdata,ydata,c=cluster_ids)
            # if the features are the same, histogram
            else:
                # plot the histogram for each cluster
                for k in range(K):
                    ax[feature1,feature2].hist(xdata[cluster_ids==k])

    fig.suptitle(f"Representation of {K} clusters, modeled by c2-hornblower",fontsize=16,y=0.925)

# save the figure
    fig.savefig("myVisualisation.png")

    return fig,ax
    
