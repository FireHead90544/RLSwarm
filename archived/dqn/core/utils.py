import numpy as np

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def distance(a, b):
    return np.linalg.norm(a - b)

def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else np.zeros_like(v)

def reflect_angle(angle, normal_axis):
    """Reflects angle when hitting a wall: normal_axis='x' or 'y'"""
    if normal_axis == 'x':
        return 180 - angle
    elif normal_axis == 'y':
        return -angle
    return angle
