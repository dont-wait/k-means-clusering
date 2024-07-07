from tkinter import messagebox
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
import tkinter as tk
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
        oldCentroids = None

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
            if len(idx_j) == 0:
            # Nếu không có điểm nào thuộc cụm j, chọn một điểm ngẫu nhiên làm centroid
                centroid_j = dataSet[np.random.choice(dataSet.shape[0])]
            else:
                centroid_j = dataSet[idx_j, :].mean(axis=0)
            centroids.append(centroid_j)
        return np.array(centroids)

# Vòng lặp yêu cầu người dùng nhập k 
def run_kmeans(k):
    dataset, _ = make_blobs(n_samples=500, cluster_std=3.0, random_state=123)
    kmean = KMeans(k, max_iteration=8)
    centroids = kmean.fit(dataset)

    # Tính toán số hàng và cột cần thiết cho GridSpec
    num_iterations = len(kmean.all_centroids)
    rows = int(np.ceil(np.sqrt(num_iterations)))
    cols = int(np.ceil(num_iterations / rows))

    gs = GridSpec(nrows=rows, ncols=cols)
    plt.figure(figsize=(20, 20))
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    #random color base on k
    colors = []
    hex_chars = '0123456789ABCDEF'
    for _ in range(k):
        colors.append("#" + ''.join([random.choice(hex_chars) for _ in range(6)]))
    #nit label base on k
    labels = [f'cluster {i + 1}' for i in range(k)]

    #draw
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
                plt.scatter(dataset[idx_j, 0], dataset[idx_j, 1], color=colors[j % len(colors)], label=labels[j % len(labels)], s=50, alpha=0.3, lw=0)
                plt.scatter(centroids_i[j, 0], centroids_i[j, 1], marker='^', color='black', s=100, label=labels[j % len(labels)])
            plt.title(f'Iteration {i}')
    plt.show()
def on_submit():
    try:
        k = int(entry.get())
        if k > 2:
            root.destroy()
            run_kmeans(k)
        else:
            messagebox.showerror("Invalid Input", "Please enter a value greater than 2.")
    except ValueError:
            messagebox.showerror("Invalid Input", "Please enter an integer.")

root = tk.Tk()
root.title("K-Means Clustering")

tk.Label(root, text="Enter the value of k (must be an integer greater than 2):").pack(pady=10)
entry = tk.Entry(root)
entry.pack(pady=5)

submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.pack(pady=10)

root.mainloop()   
