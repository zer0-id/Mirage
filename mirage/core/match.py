import numpy as np
from numpy.typing import NDArray


def cosine_distance(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    distance = 1 - np.dot(a, b)
    return float(distance)


def is_match(stored: NDArray[np.float32], live: NDArray[np.float32], threshold: float) -> tuple[bool, float]:
    dist = cosine_distance(stored, live)
    return dist < threshold, dist
