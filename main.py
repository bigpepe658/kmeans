import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def fit(self, X):
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx].copy()

        for _ in range(self.max_iters):
            distances = np.sqrt(((X[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
            self.labels = np.argmin(distances, axis=1)

            old_centroids = self.centroids.copy()
            self.centroids = np.array([X[self.labels == k].mean(axis=0)
                                       for k in range(self.n_clusters)])

            if np.allclose(old_centroids, self.centroids):
                break

        self.inertia_ = np.sum((X - self.centroids[self.labels]) ** 2)
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels


def find_optimal_k(X, max_k=10):
    """Поиск оптимального k методом локтя и силуэта"""
    inertias = []
    silhouettes = []

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k)
        km.fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(2, max_k + 1), inertias, 'bo-')
    ax1.set_title('Метод локтя')
    ax1.set_xlabel('Количество кластеров (k)')
    ax1.set_ylabel('Инерция')
    ax1.grid(True)

    ax2.plot(range(2, max_k + 1), silhouettes, 'ro-')
    ax2.set_title('Метод силуэта')
    ax2.set_xlabel('Количество кластеров (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    best_k = np.argmax(silhouettes) + 2
    print(f"Оптимальное количество кластеров: {best_k}")
    return best_k


def generate_random_data():
    """Генерация случайных данных"""
    np.random.seed(np.random.randint(0, 10000))

    # Случайное количество кластеров от 2 до 5
    n_clusters = np.random.randint(2, 6)
    X = []

    for _ in range(n_clusters):
        # Случайный центр кластера
        center = np.random.uniform(-5, 5, 2)
        # Случайный размер кластера
        n_points = np.random.randint(30, 100)
        # Случайный разброс
        spread = np.random.uniform(0.3, 1.5)
        # Генерируем точки вокруг центра
        cluster = np.random.randn(n_points, 2) * spread + center
        X.append(cluster)

    X = np.vstack(X)

    # Добавляем случайный шум
    if np.random.random() > 0.5:
        n_noise = np.random.randint(5, 20)
        noise = np.random.uniform(X.min(), X.max(), (n_noise, 2))
        X = np.vstack([X, noise])

    return X


# Демонстрация
if __name__ == "__main__":
    # Генерируем случайные данные
    X = generate_random_data()

    print(f"Сгенерировано точек: {len(X)}")

    # Показываем исходные данные
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6, s=50, c='gray')
    plt.title('Случайные исходные данные')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Ищем оптимальное k
    best_k = find_optimal_k(X, max_k=8)

    # Кластеризация с оптимальным k
    km = KMeans(n_clusters=best_k)
    labels = km.fit_predict(X)

    # Визуализация результата
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, best_k))

    for k in range(best_k):
        mask = labels == k
        plt.scatter(X[mask, 0], X[mask, 1], c=[colors[k]], alpha=0.6,
                    s=50, label=f'Кластер {k + 1} ({np.sum(mask)} точек)')

    plt.scatter(km.centroids[:, 0], km.centroids[:, 1],
                c='black', marker='X', s=300, linewidths=3, label='Центроиды')

    plt.title(f'KMeans: {best_k} кластеров')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"Инерция: {km.inertia_:.2f}")

