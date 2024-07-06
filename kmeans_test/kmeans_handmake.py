import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Công thức tính khoảng cách giữa mỗi point và cluster gần nhất
def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

# Tìm khoảng cách và gán tọa độ cho cluster gần nhất
def assign_cluster(x, clusters, k):
    for cluster in clusters.values():
        cluster['points'] = []  # Clear previous points
    total_distance = 0
    for curr_x in x:
        dist = [distance(curr_x, clusters[i]['center']) for i in range(k)]
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
        total_distance += dist[curr_cluster]
    return clusters, total_distance

# Cập nhật các point gần cluster
def update_cluster(clusters, k):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            clusters[i]['center'] = points.mean(axis=0)
    return clusters

# Dự đoán chính xác các point thuộc clusters mà nó thuộc về
def pred_cluster(x, clusters, k):
    pred = []
    for i in range(x.shape[0]):
        #dist = [distance(x[i], clusters[j]['center']) for j in range(k)]
        pred.append(np.argmin([distance(x[i], clusters[j]['center']) for j in range(k)]))
    return pred

# Khởi tạo các tham số và tạo biểu đồ ban đầu
k = int(input("What's k? "))
x, y = make_blobs(n_samples=500, n_features=2, centers=k, random_state=23)
fig = plt.figure(0)
plt.scatter(x[:, 0], x[:, 1])
np.random.seed(5)

clusters = {}
# Khởi tạo tọa độ k cluster ngẫu nhiên
for idx in range(k):
    center = 2 * (2 * np.random.random((x.shape[1],)) - 1)
    clusters[idx] = {'center': center, 'points': []}

# Vẽ các điểm dữ liệu và các trung tâm cluster ban đầu
plt.scatter(x[:, 0], x[:, 1])
plt.grid(True)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0], center[1], marker='*', c='red')
plt.show()

# Thuật toán K-means với kiểm tra hội tụ
prev_total_distance = None
for _ in range(100):  # Giới hạn số lần lặp để tránh vòng lặp vô hạn
    clusters, total_distance = assign_cluster(x, clusters, k)
    clusters = update_cluster(clusters, k)
    if prev_total_distance is not None and abs(prev_total_distance - total_distance) < 1e-4:
        break
    prev_total_distance = total_distance

# Vẽ kết quả cuối cùng
pred = pred_cluster(x, clusters, k)
plt.scatter(x[:, 0], x[:, 1], c=pred)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0], center[1], marker='>', c='red')
plt.show()
