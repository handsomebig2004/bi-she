import numpy as np
from sklearn.cluster import KMeans

def kmeans_3bin_labels(y_train, y_all=None, random_state=0, n_init=50):
    """
    y_train: (N_train,) 连续标签（例如 NASA overall / weighted score）
    y_all:   (N_all,)   需要被转换的所有标签（训练/验证/测试一起转）
             如果 None，就只转 y_train
    return:
      y_train_cls, y_all_cls, kmeans, cluster2class
    """
    y_train = np.asarray(y_train).reshape(-1, 1).astype(np.float32)

    kmeans = KMeans(n_clusters=3, n_init=n_init, random_state=random_state)
    kmeans.fit(y_train)

    # 关键：按簇中心从小到大排序，得到 low/mid/high
    centers = kmeans.cluster_centers_.reshape(-1)          # (3,)
    order = np.argsort(centers)                            # 小 -> 大
    cluster2class = {int(cluster_id): int(rank) for rank, cluster_id in enumerate(order)}
    # rank: 0 low, 1 mid, 2 high

    def convert(y):
        y = np.asarray(y).reshape(-1, 1).astype(np.float32)
        cluster = kmeans.predict(y)
        cls = np.vectorize(cluster2class.get)(cluster).astype(np.int64)
        return cls

    y_train_cls = convert(y_train.reshape(-1))

    if y_all is None:
        return y_train_cls, None, kmeans, cluster2class
    else:
        y_all_cls = convert(y_all)
        return y_train_cls, y_all_cls, kmeans, cluster2class

