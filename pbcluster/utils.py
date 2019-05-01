# -*- coding:utf-8 -*-
#
# utils.py

import numpy as np
import pandas as pd
import networkx as nx


def distinct_pairwise_distances(X, Y, L):
    X = np.array(X)
    Y = np.array(Y)
    try:
        if len(L) == X.shape[1]:
            L = np.array(L)
    except:
        L = L * np.ones(X.shape[1])
    distances = (X[:, None, :] - Y[None, :, :]) / L[None, None, :]
    distances = np.where(distances > 0.5, distances - 1.0, distances)
    distances = np.where(distances <= -0.5, distances + 1.0, distances)
    distances *= L[None, None, :]
    distances = np.linalg.norm(distances, axis=2)
    return distances


def pairwise_distances(X, L):
    return distinct_pairwise_distances(X, X, L)


def get_within_cutoff_matrix(distances, cutoff):
    return np.where(distances < cutoff, 1, 0) - np.eye(distances.shape[0])


def get_within_cutoff_graph(distances, cutoff):
    within_cutoff_matrix = get_within_cutoff_matrix(distances, cutoff)
    return nx.from_numpy_matrix(within_cutoff_matrix)


def get_graph_from_particle_positions(
    particle_positions, box_lengths, cutoff_distance, store_positions=False
):
    distances = pairwise_distances(particle_positions, box_lengths)
    graph = get_within_cutoff_graph(distances, cutoff_distance)
    if store_positions is True:
        for particle_id, particle_position in zip(
            graph.nodes, particle_positions
        ):
            for i, x_i in enumerate(particle_position):
                graph.nodes[particle_id][f"x{i}"] = x_i
    return graph
