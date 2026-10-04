import numpy as np


def cosine_distance(a, b):
    distance = 1 - np.dot(a, b)
    return distance


def is_match(stored, live, threshold=0.4):
    dist = cosine_distance(stored, live)
    return dist < threshold, dist
