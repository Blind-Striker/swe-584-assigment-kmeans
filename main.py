import math
import random
import statistics
import copy

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

number_of_clusters_const = 4
points_per_cluster_const = 150


def generate_landscape(number_of_clusters: int, points_per_cluster: int):
    cluster_mean_x = 100
    cluster_mean_y = 100
    cluster_deviation_x = 50
    cluster_deviation_y = 50
    point_deviation_x = 5
    point_deviation_y = 5

    axis_x = []
    axis_y = []

    for i in range(number_of_clusters):
        center_x = random.gauss(cluster_mean_x, cluster_deviation_x)
        center_y = random.gauss(cluster_mean_y, cluster_deviation_y)

        for j in range(points_per_cluster):
            axis_x.append(random.gauss(center_x, point_deviation_x))
            axis_y.append(random.gauss(center_y, point_deviation_y))

    return axis_x, axis_y


def generate_random_points(number_of_clusters: int, x_axis: list[float], y_axis: list[float]):
    random_x_axis: list[float] = []
    random_y_axis: list[float] = []

    for i in range(number_of_clusters):
        random_x = random.uniform(min(x_axis), max(x_axis))
        random_y = random.uniform(min(y_axis), max(y_axis))
        random_x_axis.append(random_x)
        random_y_axis.append(random_y)

    return random_x_axis, random_y_axis


def pick_random_points(number_of_clusters: int, x_axis: list[float], y_axis: list[float]):
    random_x_axis: list[float] = []
    random_y_axis: list[float] = []

    for i in range(number_of_clusters):
        temp_index = random.randint(0, len(x_axis) - 1)
        random_x = x_axis[temp_index]
        random_y = y_axis[temp_index]
        random_x_axis.append(random_x)
        random_y_axis.append(random_y)

    return random_x_axis, random_y_axis


def kmeans(number_of_clusters: int, x_axis: list[float], y_axis: list[float]):
    diff = 1
    centroid_point_x_mapping: dict[int, list[float]] = {}
    centroid_point_y_mapping: dict[int, list[float]] = {}
    point_centroid_mapping: list[int] = [0] * len(x_axis)
    centroids_x_axis: list[float]
    centroids_y_axis: list[float]
    centroids_x_axis, centroids_y_axis = pick_random_points(number_of_clusters, x_axis, y_axis)

    centroids_x_axis_by_iterations: list[list[float]] = []
    centroids_y_axis_by_iterations: list[list[float]] = []
    point_centroid_mapping_by_iterations: list[list[int]] = []

    random_fix_attempt: int = 0
    attempt: int = 0
    while diff:
        # for each observation
        for point_index in range(len(x_axis)):
            max_distance = float('inf')
            point_x = x_axis[point_index]
            point_y = y_axis[point_index]

            # dist of the point from all centroids
            for centroid_index in range(len(centroids_x_axis)):
                centroid_x = centroids_x_axis[centroid_index]
                centroid_y = centroids_y_axis[centroid_index]
                distance = math.sqrt((centroid_x - point_x) ** 2 + (centroid_y - point_y) ** 2)

                # store closest centroid
                if max_distance > distance:
                    max_distance = distance
                    point_centroid_mapping[point_index] = centroid_index

        centroids_x_axis_by_iterations.append(copy.deepcopy(centroids_x_axis))
        centroids_y_axis_by_iterations.append(copy.deepcopy(centroids_y_axis))
        point_centroid_mapping_by_iterations.append(copy.deepcopy(point_centroid_mapping))

        for point_index, centroid_index in enumerate(point_centroid_mapping):
            if centroid_index not in centroid_point_x_mapping:
                centroid_point_x_mapping[centroid_index] = []
            if centroid_index not in centroid_point_y_mapping:
                centroid_point_y_mapping[centroid_index] = []

            centroid_point_x_mapping[centroid_index].append(x_axis[point_index])
            centroid_point_y_mapping[centroid_index].append(y_axis[point_index])

        if len(centroid_point_x_mapping) != len(centroids_x_axis) and random_fix_attempt < 10:
            centroids_x_axis, centroids_y_axis = pick_random_points(number_of_clusters, x_axis, y_axis)
            random_fix_attempt += 1
            continue

        fixed_centroid = 0
        for centroid_index in range(len(centroids_x_axis)):
            if centroid_index not in centroid_point_x_mapping:
                centroid_point_x_mapping[centroid_index] = [random.uniform(min(x_axis), max(x_axis))]
            if centroid_index not in centroid_point_y_mapping:
                centroid_point_y_mapping[centroid_index] = [random.uniform(min(y_axis), max(y_axis))]

            centroid_x = centroids_x_axis[centroid_index]
            centroid_y = centroids_y_axis[centroid_index]
            new_centroid_x = statistics.mean(centroid_point_x_mapping[centroid_index])
            new_centroid_y = statistics.mean(centroid_point_y_mapping[centroid_index])

            x_diff = centroid_x - new_centroid_x
            if x_diff < 0:
                x_diff = x_diff * -1

            y_diff = centroid_y - new_centroid_y
            if y_diff < 0:
                y_diff = y_diff * -1

            if x_diff >= 0.1 or y_diff >= 0.1:
                centroids_x_axis[centroid_index] = new_centroid_x
                centroids_y_axis[centroid_index] = new_centroid_y
            else:
                fixed_centroid += 1

        if number_of_clusters == fixed_centroid or attempt == 300:
            diff = 0
        else:
            attempt += 1

    return centroids_x_axis_by_iterations, centroids_y_axis_by_iterations, point_centroid_mapping_by_iterations


axis_x_result, axis_y_result = generate_landscape(number_of_clusters_const, points_per_cluster_const)

# Get results of my k-means
centroids_x_axis_results, centroids_y_axis_results, point_centroid_mapping_results = kmeans(number_of_clusters_const,
                                                                                            axis_x_result,
                                                                                            axis_y_result)

centroids_first_iter_x = centroids_x_axis_results[0]
centroids_first_iter_y = centroids_y_axis_results[0]
point_centroid_mapping_first_iter = point_centroid_mapping_results[0]

centroids_second_iter_x = centroids_x_axis_results[1]
centroids_second_iter_y = centroids_y_axis_results[1]
point_centroid_mapping_second_iter = point_centroid_mapping_results[1]

centroids_third_iter_x = centroids_x_axis_results[2]
centroids_third_iter_y = centroids_y_axis_results[2]
point_centroid_mapping_third_iter = point_centroid_mapping_results[2]

centroids_x_axis_result = centroids_x_axis_results[len(centroids_x_axis_results) - 1]
centroids_y_axis_result = centroids_y_axis_results[len(centroids_y_axis_results) - 1]
point_centroid_mapping_result = point_centroid_mapping_results[len(point_centroid_mapping_results) - 1]
