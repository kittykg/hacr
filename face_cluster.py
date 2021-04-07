from joblib import dump
import time

from sklearn.cluster import MiniBatchKMeans, KMeans
import numpy as np
from tqdm import tqdm

from common import BBT_PEOPLE

if __name__ == '__main__':
    start_time = time.time()

    cluster_buffer = 2
    mode = 'mini'
    iter_num = 100

    n_cluster = len(BBT_PEOPLE) + cluster_buffer

    arr = np.load('./face_collection.npz')
    faces = arr['faces']

    if mode == 'mini':
        kmeans = MiniBatchKMeans(n_cluster,
                                 random_state=0,
                                 batch_size=32)
        for i in range(iter_num):
            kmeans.fit(faces)
    else:
        kmeans = KMeans(n_cluster, random_state=0)
        kmeans.fit(faces)

    dump(kmeans, 'face_cluster.joblib')
    print(F'Running time: {time.time() - start_time}')
