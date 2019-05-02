# -*- coding:utf-8 -*-
#
# utils.py

"Utils Module."

import numpy as np
import pandas as pd
import networkx as nx


def pairwise_distances_distinct(
    particle_positions_1, particle_positions_2, box_lengths
):
    """Returns pairwise distance matrix between 2 distinct sets of particle
    positions accounting for periodic boundary conditions
    
    Args:
        particle_positions_1 (ndarray or dataframe): Shape
            (`n_particles_1`, `n_dimensions`). Each of the `n_particles_1`
            rows is a length `n_dimensions` particle position vector.
            Positions must be in range [0, `box_lengths[d]`) for each
            dimension `d`.
        particle_positions_2 (ndarray or dataframe): Shape
            (`n_particles_2`, `n_dimensions`). Each of the `n_particles_2`
            rows is a length `n_dimensions` particle position vector.
            Positions must be in range [0, `box_lengths[d]`) for each
            dimension `d`.
        box_lengths (ndarray): Shape (`n_dimensions`,) array of box lengths for
            each box dimension.
    
    Raises:
        ValueError: Length of last dimension of each argument doesn't match
    
    Returns:
        ndarray: Shape (`n_particles_1`, `n_particles_2`) matrix of pairwise
        euclidean distances.
    """
    particle_positions_1 = np.array(particle_positions_1)
    particle_positions_2 = np.array(particle_positions_2)
    _, n_dimensions_1 = particle_positions_1.shape
    _, n_dimensions_2 = particle_positions_2.shape
    if n_dimensions_1 != n_dimensions_2:
        raise ValueError(
            "Both particle positions arrays must have the same number of "
            "columns!"
        )
    n_dimensions = n_dimensions_1
    if len(box_lengths) != n_dimensions:
        raise ValueError(
            "Length of 'box_lengths' must match number of columns in particle "
            "position arrays!"
        )
    # Create 3-dimensional array of normalized diffs. Shape is
    # (`n_particles_1`, `n_particles_2`, `n_dimensions`), and values are scaled
    # by box_lengths so that they fall in the range of (-1, 1)
    diffs = (
        particle_positions_1[:, None, :] - particle_positions_2[None, :, :]
    ) / box_lengths[None, None, :]
    # Apply periodic boundary conditions, which make it so that maximum absolute
    # difference between 2 particles in any dimension is half the box length in
    # that dimension
    diffs = np.where(diffs < -0.5, diffs + 1.0, diffs)
    diffs = np.where(diffs >= 0.5, diffs - 1.0, diffs)
    # Reapply box lengths
    diffs *= box_lengths[None, None, :]
    # calculate Euclidean distances by taking the L2 norm along the 3rd axis
    distances = np.linalg.norm(diffs, axis=2)
    return distances


def pairwise_distances(particle_positions, box_lengths):
    """Returns pairwise distance matrix between row vectors in a single
    positions matrix
    
    Args:
        particle_positions (ndarray or dataframe): Shape
            (`n_particles`, `n_dimensions`). Each of the `n_particles`
            rows is a length `n_dimensions` particle position vector.
            Positions must be in range [0, `box_lengths[d]`) for each
            dimension `d`.
        box_lengths (ndarray): Shape (`n_dimensions`,) array of box lengths for
            each box dimension.
    
    Returns:
        ndarray: Shape (`n_particles`, `n_particles`) symmetric matrix of
        pairwise euclidean distances.
    """
    return pairwise_distances_distinct(
        particle_positions, particle_positions, box_lengths
    )


def flatten_dict(input_dict):
    """Returns flattened dictionary given an input dictionary with maximum depth
    of 2
    
    Args:
        input_dict (dict): `str → number` key-value pairs, where value can be a
            number or a dictionary with `str → number` key-value paris.
    
    Returns:
        dict: Flattened dictionary with underscore-separated keys if
        `input_dict` contained nesting
    """
    output_dict = dict()
    for key, value in input_dict.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                output_dict[f"{key}_{subkey}"] = subvalue
        else:
            output_dict[key] = value
    return output_dict


def get_within_cutoff_matrix(distances, cutoff_distance):
    """Returns matrix of 0s and 1s that can be fed into networkx to initialize a
    graph
    
    Args:
        distances (ndarray or dataframe): Shape (`n_particles`, `n_particles`)
            symmetric matrix of pairwise euclidean distances.
        cutoff_distance (float): Maximum length between particle pairs to
            consider them connected
    
    Returns:
        ndarray: Shape (`n_particles`, `n_particles`) symmetric binary array
    """
    return np.where(distances < cutoff_distance, 1, 0) - np.eye(
        distances.shape[0]
    )


def get_within_cutoff_graph(distances, cutoff_distance):
    """Converts pairwise distances matrix into networkx graph of connections
    between `i` and `j` (`i` :math:`\\ne` `j`) where
    `distances[i, j]` :math:`\\le` `cutoff_distance`.
    
    Args:
        distances (ndarray or dataframe): Shape (`n_particles`, `n_particles`)
            symmetric matrix of pairwise euclidean distances.
        cutoff_distance (float): Maximum length between particle pairs to
            consider them connected
    
    Returns:
        networkx Graph: Graph of connections between all particle pairs with
        distance below cutoff_distance
    """
    within_cutoff_matrix = get_within_cutoff_matrix(distances, cutoff_distance)
    return nx.from_numpy_matrix(within_cutoff_matrix)


def get_graph_from_particle_positions(
    particle_positions, box_lengths, cutoff_distance, store_positions=False
):
    """Returns a networkx graph of connections between neighboring particles
    
    Args:
        particle_positions (ndarray or dataframe): Shape
            (`n_particles`, `n_dimensions`). Each of the `n_particles`
            rows is a length `n_dimensions` particle position vector.
            Positions must be in range [0, `box_lengths[d]`) for each
            dimension `d`.
        box_lengths (ndarray): Shape (`n_dimensions`,) array of box lengths for
            each box dimension.
        cutoff_distance (float): Maximum length between particle pairs to
            consider them connected
        store_positions (bool, optional): If True, store position vector data
            within each node in the graph. Defaults to False.
    
    Returns:
        networkx Graph: Graph of connections between all particle pairs with
        distance below cutoff_distance
    """
    distances = pairwise_distances(particle_positions, box_lengths)
    graph = get_within_cutoff_graph(distances, cutoff_distance)
    if store_positions is True:
        for particle_id, particle_position in zip(
            graph.nodes, particle_positions
        ):
            for i, x_i in enumerate(particle_position):
                graph.nodes[particle_id][f"x{i}"] = x_i
    return graph
