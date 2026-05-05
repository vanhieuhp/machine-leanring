import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def euclidean_distance(x, y):
    # point = [x1, x2, ...] — một điểm dữ liệu
    # centroid = [c1, c2, ...] — một centroid
    # Trả về: khoảng cách (một số thực)
    return np.linalg.norm(x - y)

def assign_clusters(points, centroids):
    # points: list các điểm, mỗi điểm là array [x1, x2, ...]
    # centroids: list các centroid
    # Trả về: list các index — ví dụ [0, 2, 1, 0, ...]
    #         nghĩa là điểm 0 thuộc nhóm 0, điểm 1 thuộc nhóm 2, ...
    assignments = []

    for point in points:
        # Với mỗi điểm, tính khoảng cách tới từng centroid
        distances = []
        for centroid in centroids:
            distance = euclidean_distance(point, centroid)
            distances.append(distance)

        # rồi lấy index của centroid gần nhất
        assignments.append(np.argmin(distances))

    return assignments

def update_centroids(points, assignments, K):
    # points: list các điểm
    # assignments: list index nhóm, ví dụ [0, 2, 1, 0, ...]
    # K: số nhóm
    # Trả về: list K centroid mới

    points = np.array(points)
    assignments = np.array(assignments)
    new_centroids = []

    for k in range(K):
        # Bước 1: Lấy tất cả các điểm có nhãn (assignment) bằng k
        # Sử dụng Boolean Indexing giúp code gọn và nhanh hơn
        cluster_points = points[assignments == k]

        # Bước 2: Kiểm tra nếu cụm có điểm thì mới tính trung bình
        if len(cluster_points) > 0:
            # Tính trung bình cộng theo cột (axis=0) để ra tọa độ mới
            new_centroid = np.mean(cluster_points, axis=0)
            new_centroids.append(new_centroid)
        else:
            # Nếu cụm rỗng, có thể giữ nguyên centroid cũ hoặc xử lý tùy chọn
            pass

    return new_centroids

def kmeans(points, K, max_iters=100, n_init=100):
    points = np.array(points)
    best_assignments = None
    best_centroids = None
    best_inertia = float('inf')

    for _ in range(n_init):
        # 1. Khởi tạo centroid ngẫu nhiên — chọn K điểm từ data
        random_indices = np.random.choice(len(points), K, replace=False)
        centroids = points[random_indices]

        for i in range(max_iters):
            assignments = np.array(assign_clusters(points, centroids))
            new_centroids = np.array(update_centroids(points, assignments, K))
            if np.allclose(new_centroids, centroids):
                print(f"Hội tụ sau {i+1} vòng lặp")
                break
            centroids = new_centroids

        # 1. get collections of points for each cluster
        # 2. calculate the centroid of each cluster
        # 3. sum all distances between points and centroids

        # Tính Inertia (Tổng bình phương khoảng cách) để đánh giá độ tốt của cụm
        inertia = 0
        for k in range(K):
            cluster_points = points[assignments == k]
            if len(cluster_points) > 0:
                distances = np.linalg.norm(cluster_points - centroids[k], axis=1)
                inertia += np.sum(distances ** 2)

        if inertia < best_inertia:
            best_inertia = inertia
            best_assignments = assignments
            best_centroids = centroids

    return best_assignments, best_centroids, best_inertia


points, true_labels = make_blobs(n_samples=150, centers=3, random_state=42)
assignments, centroids, best_inertia = kmeans(points, K=3, n_init=10)
print(f"Inertia của mình: {best_inertia:.2f}")
# Vẽ kết quả
colors = ['#378ADD', '#D4537E', '#639922']

for k in range(3):
    cluster_points = points[assignments == k]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                color=colors[k], label=f'Nhóm {k+1}', alpha=0.7)

plt.scatter(centroids[:, 0], centroids[:, 1],
            color='black', marker='X', s=200, label='Centroid')

plt.legend()
plt.title('K-Means từ scratch (n_init=10)')
plt.show()

from sklearn.cluster import KMeans

sklearn_model = KMeans(n_clusters=3, n_init=10, random_state=42)
sklearn_model.fit(points)

print(f"Inertia sklearn: {sklearn_model.inertia_:.2f}")