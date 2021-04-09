from joblib import dump
import time

from sklearn.cluster import KMeans
import numpy as np

from common import BBT_PEOPLE

start_time = time.time()

cluster_buffer = 2
npz_file_path = './face_collection.npz'
n_cluster = len(BBT_PEOPLE) + cluster_buffer

faces = np.load(npz_file_path)['faces']

# Ensure the NPZ file has all faces are included
assert len(faces) == 67035

kmeans = KMeans(n_cluster, random_state=0)
kmeans.fit(faces)

dump(kmeans, 'face_cluster.joblib')

# Print parameters
print(F'NPZ file:     {npz_file_path}')
print(F'n_cluster:    {n_cluster}')
print(F'Running time: {time.time() - start_time}')
