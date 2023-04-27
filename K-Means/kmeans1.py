import numpy as np
from collections import defaultdict



data = np.loadtxt('ieeedata.csv', skiprows=1, delimiter=',')
K = 3


class Point:
    def __init__(self, data):
        self.data = data
        self.k = np.random.randint(0, K)


points = [Point(i) for i in data]


def make_k_mapping(points):
    point_dict = defaultdict(list)
    for p in points:
        point_dict[p.k] = point_dict[p.k] + [p.data]
    return point_dict


def calc_k_means(point_dict):
    means = [np.mean(point_dict[k], axis=0) for k in range(K)]
    return means


def update_k(points, means):
    for p in points:
        dists = [np.linalg.norm(means[k] - p.data) for k in range(K)]
        p.k = np.argmin(dists)


def fit(points, epochs=500):
    for e in range(epochs):
        point_dict = make_k_mapping(points)
        means = calc_k_means(point_dict)
        update_k(points, means)
    return means


new_means = fit(points)


def AIC(points):
    sum = 0
    for p in points:
        dist = (np.linalg.norm(new_means[p.k] - p.data))**2
        sum = sum + dist

    value = sum/(5*K)
    print(sum)

print(new_means)

