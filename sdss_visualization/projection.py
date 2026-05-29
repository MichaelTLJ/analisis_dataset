import numpy as np


def standardize_matrix(frame, feature_columns):
    values = frame[feature_columns].to_numpy(dtype=float)
    mean = values.mean(axis=0)
    std = np.where(values.std(axis=0) == 0, 1, values.std(axis=0))
    return (values - mean) / std


def pca_2d(embedding):
    _, singular_values, vt = np.linalg.svd(embedding, full_matrices=False)
    coordinates = embedding @ vt[:2].T
    explained = (singular_values ** 2) / np.sum(singular_values ** 2)
    return coordinates, explained

