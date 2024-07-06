import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class KMeans():
    def __init__(self, k, max_iteration=10):
        self.k = k
        self.max_iteration = max_iteration
        self.all_centroids = []
        self.all_labels = []

    # Hàm lấy đầu vào là một bộ dữ liệu, số lượng clusters K, trả về tâm của K cụm
    def fit(self, dataSet):
        #init k centroids
        numFeatures = dataSet.shape[1]
        centroids = self.get_random_centroids(numFeatures, self.k)
        self.all_centroids.append(centroids)
        self.all_labels.append(None)

        #init variable iterations, oldCentroids
        iterations = 0
        oldCentroids = np.zeros_like(centroids)

        #Loop update centroid in K-Means
        while not self.should_stop(oldCentroids, centroids, iterations):
            #Save old centroids to check clustering
            oldCentroids = centroids
            iterations += 1

            #label to points depend on centroids
            labels = self.get_labels(dataSet, centroids)
            self.all_labels.append(labels)

            #Update centroids depend on label data
            centroids = self.get_centroids(dataSet, labels, self.k)
            self.all_centroids.append(centroids)

        return centroids

    #Init random centroids
    def get_random_centroids(self, numFeatures, k):
        return np.random.rand(k, numFeatures)
    
    #return label of each point in datasets
    def get_labels(self, dataset, centroids):
        labels = []
        for x in dataset:
            label = np.argmin(np.sum((x - centroids)**2, axis=1))
            labels.append(label)
        return labels

    #check true false if finnish K_means algo
    #if out of idx loop(10) or centroids stop change -> stop
    def should_stop(self, oldCentroids, centroids, iterations):
        if iterations > self.max_iteration:
            return True
        return np.all(oldCentroids == centroids)
    
    #get new pos to k centroid each chiều
    def get_centroids(self, dataSet, labels, k):
        centroids = []
        for j in np.arange(k):
            idx_j = np.where(np.array(labels) == j)[0]
            centroid_j = dataSet[idx_j, :].mean(axis=0)
            centroids.append(centroid_j)
        return np.array(centroids)

dataset, _ = make_blobs(n_samples=250, cluster_std=3.0, random_state=123)

kmean = KMeans(k=2, max_iteration=8)
centroids = kmean.fit(dataset)

gs = GridSpec(nrows=3, ncols=3)
plt.figure(figsize=(20, 20))
plt.subplots_adjust(wspace=0.2, hspace=0.4)
colors = ['green', 'blue']
labels = ['cluster 1', 'cluster 2']

for i in np.arange(len(kmean.all_centroids)):
    ax = plt.subplot(gs[i])
    if i == 0:
        centroids_i = kmean.all_centroids[i]
        plt.scatter(dataset[:, 0], dataset[:, 1], s=50, alpha=0.5, color='red')
        for j in np.arange(kmean.k):
            plt.scatter(centroids_i[j, 0], centroids_i[j, 1], marker='^', s=100, color='blue')
        plt.title('All points in original dataset')
    else:
        # Lấy centroids và labels tại bước thứ i
        centroids_i = kmean.all_centroids[i]
        labels_i = kmean.all_labels[i]
        # Visualize các điểm cho từng cụm
        for j in np.arange(kmean.k):
            idx_j = np.where(np.array(labels_i) == j)[0]
            plt.scatter(dataset[idx_j, 0], dataset[idx_j, 1], color=colors[j], label=labels[j], s=50, alpha=0.3, lw=0)
            plt.scatter(centroids_i[j, 0], centroids_i[j, 1], marker='^', color=colors[j], s=100, label=labels[j])
        plt.title(f'Iteration {i}')

plt.show()
