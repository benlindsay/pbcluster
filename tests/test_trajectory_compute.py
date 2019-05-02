# -*- coding:utf-8 -*-
#
# test_trajectory_compute.py

import numpy as np

from pbcluster.trajectory import Trajectory

"""Tests for Trajectory compute."""


def test_compute_n_particles():
    particle_positions = np.array([[0.25, 2], [3.25, 2], [3.75, 2]])
    box_lengths = np.array([4, 4])
    cutoff_distance = 1.0
    t = Trajectory(particle_positions, box_lengths, cutoff_distance)
    df = t.compute_cluster_properties(["n_particles"])
    assert len(df) == 1
    assert df.iloc[0]["n_particles"] == 3
    for column in ["cluster_id", "n_particles", "timestep"]:
        assert column in df.columns


def test_compute_coordination_number():
    particle_positions = np.array([[0.25, 2], [3.25, 2], [3.75, 2]])
    box_lengths = np.array([4, 4])
    cutoff_distance = 1.0
    t = Trajectory(particle_positions, box_lengths, cutoff_distance)
    df = t.compute_particle_properties(["coordination_number"])
    assert len(df) == 3
    assert np.all(df["coordination_number"] == np.array([1, 1, 2]))
    for column in [
        "particle_id",
        "x0",
        "x1",
        "coordination_number",
        "timestep",
        "cluster_id",
    ]:
        assert column in df.columns
