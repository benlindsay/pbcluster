# -*- coding:utf-8 -*-
#
# test_trajectory_compute.py

import numpy as np

from pbcluster import Trajectory

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
    assert row["rg"] == 0.5 / 3.0
    assert row["asphericity"] == 1.0
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
        "rg",
        "asphericity",
    ]:
        assert column in df.columns


def test_compute_all_percolated_cluster_properties():
    x = np.linspace(0.25, 3.75, 8)
    y = np.ones(len(x))
    particle_positions = np.stack([x, y], axis=1)
    box_lengths = np.array([4, 4])
    cutoff_distance = 0.75
    t = Trajectory(particle_positions, box_lengths, cutoff_distance)
    df = t.compute_cluster_properties("all")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["n_particles"] == 8
    assert row["minimum_node_cuts_x0"] == 1
    assert row["minimum_node_cuts_x1"] == 0
    assert np.isnan(row["center_of_mass_x0"])
    assert np.isnan(row["center_of_mass_x1"])
    assert np.isnan(row["unwrapped_center_of_mass_x0"])
    assert np.isnan(row["unwrapped_center_of_mass_x1"])
    assert np.isnan(row["rg"])
    assert np.isnan(row["asphericity"])
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
        "rg",
        "asphericity",
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


def test_compute_all_particle_properties():
    particle_positions = np.array([[0.25, 2], [3.25, 2], [3.75, 2]])
    box_lengths = np.array([4, 4])
    cutoff_distance = 1.0
    t = Trajectory(particle_positions, box_lengths, cutoff_distance)
    df = t.compute_particle_properties(properties="all")
    assert len(df) == 3
    assert np.all(df["coordination_number"] == np.array([1, 1, 2]))
    assert np.all(df["dx_from_com_x0"] == np.array([0.5, -0.5, 0]))
    assert np.all(df["dx_from_com_x1"] == np.array([0, 0, 0]))
    assert np.all(df["distance_from_com"] == np.array([0.5, 0.5, 0]))
    for column in [
        "particle_id",
        "x0",
        "x1",
        "coordination_number",
        "timestep",
        "cluster_id",
        "dx_from_com_x0",
        "dx_from_com_x1",
        "distance_from_com",
    ]:
        assert column in df.columns


def test_compute_bonds():
    particle_positions = np.array([[0.25, 2], [3.25, 2], [3.75, 2]])
    box_lengths = np.array([4, 4])
    cutoff_distance = 1.0
    t = Trajectory(particle_positions, box_lengths, cutoff_distance)
    bonds_df = t.compute_bonds()
    assert len(bonds_df) == 2
    assert np.all(bonds_df["particle_id_1"] == np.array([0, 1]))
    assert np.all(bonds_df["particle_id_2"] == np.array([2, 2]))
    assert np.all(bonds_df["timestep"] == np.array([0, 0]))
    assert np.all(bonds_df["cluster_id"] == np.array([0, 0]))
    for column in ["particle_id_1", "particle_id_2", "timestep", "cluster_id"]:
        assert column in bonds_df.columns


def test_compute_bond_durations():
    particle_positions = np.array([[0.25, 2], [3.25, 2], [3.75, 2]])
    box_lengths = np.array([4, 4])
    cutoff_distance = 1.0
    t = Trajectory(particle_positions, box_lengths, cutoff_distance)
    bond_durations_df = t.compute_bond_durations()
    assert len(bond_durations_df) == 2
    assert np.all(bond_durations_df["particle_id_1"] == np.array([0, 1]))
    assert np.all(bond_durations_df["particle_id_2"] == np.array([2, 2]))
    assert np.all(bond_durations_df["start"] == np.array([0, 0]))
    assert np.all(bond_durations_df["duration"] == np.array([1, 1]))
    assert np.all(bond_durations_df["bonded_at_end"] == np.array([True, True]))
    for column in [
        "particle_id_1",
        "particle_id_2",
        "start",
        "duration",
        "bonded_at_end",
    ]:
        assert column in bond_durations_df.columns
