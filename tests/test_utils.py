# -*- coding:utf-8 -*-
#
# test_utils.py

import numpy as np

from pbcluster.utils import get_graph_from_particle_positions

"""Tests for utils module."""


def test_get_graph_from_particle_positions_linear_chain():
    particle_positions = np.array([[0.25, 2], [3.25, 2], [3.75, 2]])
    box_lengths = np.array([4, 4])
    cutoff_distance = 1.0
    graph = get_graph_from_particle_positions(
        particle_positions, box_lengths, cutoff_distance
    )
    assert len(graph) == 3
    for i in range(3):
        assert i in graph.nodes
    for edge in [(0, 2), (1, 2)]:
        assert edge in graph.edges
    assert (0, 1) not in graph.edges
    # No data stored in nodes
    for i in range(3):
        assert len(graph.nodes[i]) == 0


def test_get_graph_from_particle_positions_linear_chain_with_data():
    particle_positions = np.array([[0.25, 2], [3.25, 2], [3.75, 2]])
    box_lengths = np.array([4, 4])
    cutoff_distance = 1.0
    graph = get_graph_from_particle_positions(
        particle_positions, box_lengths, cutoff_distance, store_positions=True
    )
    assert len(graph) == 3
    for i in range(3):
        assert i in graph.nodes
    for edge in [(0, 2), (1, 2)]:
        assert edge in graph.edges
    assert (0, 1) not in graph.edges
    # Position data stored in nodes
    for i in range(3):
        for j, key in enumerate(["x0", "x1"]):
            assert graph.nodes[i][key] == particle_positions[i, j]
