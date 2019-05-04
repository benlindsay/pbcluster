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


def test_compute_minimum_node_cuts():
    particle_positions = np.array([[0.25, 2], [3.25, 2], [3.75, 2]])
    box_lengths = np.array([4, 4])
    cutoff_distance = 1.0
    t = Trajectory(particle_positions, box_lengths, cutoff_distance)
    df = t.compute_cluster_properties(["minimum_node_cuts"])
    assert len(df) == 1
    assert df.iloc[0]["minimum_node_cuts_x0"] == 0
    assert df.iloc[0]["minimum_node_cuts_x1"] == 0
    for column in [
        "cluster_id",
        "timestep",
        "minimum_node_cuts_x0",
        "minimum_node_cuts_x1",
    ]:
        assert column in df.columns


def test_compute_unwrapped_center_of_mass():
    particle_positions = np.array([[0.25, 2], [3.25, 2], [3.75, 2]])
    box_lengths = np.array([4, 4])
    cutoff_distance = 1.0
    t = Trajectory(particle_positions, box_lengths, cutoff_distance)
    df = t.compute_cluster_properties(["unwrapped_center_of_mass"])
    assert len(df) == 1
    assert df.iloc[0]["unwrapped_center_of_mass_x0"] == -0.25
    assert df.iloc[0]["unwrapped_center_of_mass_x1"] == 2.0
    for column in [
        "cluster_id",
        "timestep",
        "unwrapped_center_of_mass_x0",
        "unwrapped_center_of_mass_x1",
    ]:
        assert column in df.columns


def test_compute_all_cluster_properties():
    particle_positions = np.array([[0.25, 2], [3.25, 2], [3.75, 2]])
    box_lengths = np.array([4, 4])
    cutoff_distance = 1.0
    t = Trajectory(particle_positions, box_lengths, cutoff_distance)
    df = t.compute_cluster_properties("all")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["n_particles"] == 3
    assert row["minimum_node_cuts_x0"] == 0
    assert row["minimum_node_cuts_x1"] == 0
    assert row["center_of_mass_x0"] == 3.75
    assert row["center_of_mass_x1"] == 2.0
    assert row["unwrapped_center_of_mass_x0"] == -0.25
    assert row["unwrapped_center_of_mass_x1"] == 2.0
    for column in [
        "cluster_id",
        "n_particles",
        "timestep",
        "minimum_node_cuts_x0",
        "minimum_node_cuts_x1",
        "center_of_mass_x0",
        "center_of_mass_x1",
        "unwrapped_center_of_mass_x0",
        "unwrapped_center_of_mass_x1",
    ]:
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


def test_compute_distance_from_com():
    particle_positions = np.array([[0.25, 2], [3.25, 2], [3.75, 2]])
    box_lengths = np.array([4, 4])
    cutoff_distance = 1.0
    t = Trajectory(particle_positions, box_lengths, cutoff_distance)
    df = t.compute_particle_properties(["distance_from_com"])
    assert np.all(df["dx_from_com_x0"] == np.array([0.5, -0.5, 0]))
    assert np.all(df["dx_from_com_x1"] == np.array([0, 0, 0]))
    assert np.all(df["distance_from_com"] == np.array([0.5, 0.5, 0]))
    for column in [
        "particle_id",
        "x0",
        "x1",
        "timestep",
        "cluster_id",
        "dx_from_com_x0",
        "dx_from_com_x1",
        "distance_from_com",
    ]:
        assert column in df.columns
