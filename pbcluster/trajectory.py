# -*- coding: utf-8 -*-
#
# trajectory.py

"""Trajectory module."""

import networkx as nx
import numpy as np
import pandas as pd

from .cluster import Cluster
from .utils import get_graph_from_particle_positions


class Trajectory:
    """Object to store and compute data about particle clusters in sequential
    timesteps

    Args:
        trajectory_data (dataframe or ndarray): Dataframe or ndarray containing
            trajectory data. If dataframe, it must contain columns
            `particle_id` and `x0`, `x1`, ... `xN`. If there are multiple
            timesteps, a `timestep` column must be included. If there are
            multiple  particle types, a `particle_type` column must be
            included. If ndarray, its dimensions must be
            (n_timesteps, n_particles, n_dimensions), and it will be assumed
            that all particles are of the same type.
        box_lengths (float or ndarray): If float, the length of each side of
            a cubic box. If ndarray, it must contain `n_dimensions` values
            representing the lengths of each dimension of a rectangular box.
        cutoff_distance (float): Maximum distance two particles can be from
            each other to be considered part of the same cluster

    Attributes:
        trajectory_df (dataframe): Dataframe containing trajectory data with
            columns `timestep`, `particle_id`, `particle_type`,
            `x0`, x1`, ... `xN`
        n_dimensions (int): Number of dimensions in the system
        box_lengths (ndarray): Length `n_dimensions` array representing the
            lengths of each dimension of a rectangular box.
        cutoff_distance (float): Maximum distance two particles can be from
            each other to be considered part of the same cluster
        cluster_dict (dict): `timestep → cluster_list` key-value pairs
    """

    def __init__(self, trajectory_data, box_lengths, cutoff_distance):
        if isinstance(trajectory_data, pd.DataFrame):
            self.trajectory_df = trajectory_data.copy()
        elif isinstance(trajectory_data, np.ndarray):
            self.trajectory_df = self._convert_ndarray_to_df(trajectory_data)
        self.trajectory_df = self._verify_dataframe(self.trajectory_df)
        self.n_dimensions = self._get_n_dimensions(self.trajectory_df)
        self.box_lengths = self._verify_box_lengths(box_lengths)
        self.cutoff_distance = float(cutoff_distance)
        self.cluster_dict = None
        self.cluster_dict = self._get_cluster_dict()

    def _convert_ndarray_to_df(self, trajectory_array):
        """Convert numpy array with trajectory information into a dataframe
        
        Args:
            trajectory_array (ndarray): 2D or 3D array of shape
                (n_particles, n_dimensions) if 2D, or
                (n_timesteps, n_particles, n_dimensions) if 3D. If 2D, it will
                be assumed that it represents a single timestep.
        
        Raises:
            ValueError: If trajectory_array isn't a 2D or 3D ndarray
        
        Returns:
            dataframe: Each row represents a single particle at a particular
                timestep. Columns are `timestep`, `particle_id`, `x0`, `x1`, ...
                `xN`.
        """
        if not isinstance(trajectory_array, np.ndarray):
            raise ValueError("trajectory_array must be a numpy ndarray!")
        array_dim = trajectory_array.ndim
        if array_dim < 2 or 3 < array_dim:
            raise ValueError("trajectory_array must have 2 or 3 dimensions!")
        elif array_dim == 2:
            # Reshape array with dimensions (n_particles, n_dimensions) into
            # array with dimensions (n_timesteps, n_particles, n_dimensions),
            # where n_timesteps = 1
            trajectory_array = trajectory_array[None, :, :]
        n_timesteps, n_particles, n_dimensions = trajectory_array.shape
        trajectory_array = trajectory_array.reshape(
            (n_timesteps * n_particles, n_dimensions)
        )
        x_columns = ["x{}".format(i) for i in range(n_dimensions)]
        trajectory_df = (
            pd.DataFrame(trajectory_array, columns=x_columns)
            .assign(
                timestep=np.arange(n_timesteps * n_particles) // n_particles
            )
            .assign(
                particle_id=np.arange(n_timesteps * n_particles) % n_particles
            )
        )
        # `particle_type` column is not created because all particles are
        # assumed to have the same type if an ndarray is passed, and that column
        # is created in _verify_dataframe anyway
        return trajectory_df

    def _verify_dataframe(self, trajectory_df):
        """Perform quality check on input array and return array with desired
        specifications
        
        Args:
            trajectory_data (dataframe): Dataframe containing trajectory data.
                It must contain at least column `particle_id` and columns `x0`,
                `x1`, ... `xN`. If there are multiple timesteps, a `timestep`
                column must be included. If there are multiple particle
                types, a `particle_type` column must be included.

        Raises:
            ValueError: trajectory_df does not meet specifications
        
        Returns:
            dataframe: Contains columns in order `timestep`, `particle_id`,
                `particle_type`, `x0`, `x1`, ... `xN`
        """
        x_column_names = self._get_x_column_names(trajectory_df)
        if "particle_id" not in trajectory_df.columns:
            raise ValueError("particle_id column does not exist!")
        if "timestep" not in trajectory_df.columns:
            # Assume we only have 1 timestep, meaning there should be no
            # duplicate `particle_id`s
            pids = trajectory_df["particle_id"]
            # import pdb; pdb.set_trace();
            if len(pids) != len(pids.unique()):
                raise ValueError(
                    "Duplicate particle_ids exist with no timestep column!"
                )
            trajectory_df["timestep"] = 0
        if "particle_type" not in trajectory_df.columns:
            # Assume all particles have the same type if type is not provided
            trajectory_df["particle_type"] = 0
        for _, group in trajectory_df.groupby("timestep"):
            assert np.all(group["particle_id"] == np.arange(len(group)))
        # Arrange columns in desired order
        trajectory_df = trajectory_df[
            ["timestep", "particle_id", "particle_type"] + x_column_names
        ]
        return trajectory_df

    def _get_x_column_names(self, trajectory_df):
        """Return a list of column names containing particle position info
        
        Args:
            trajectory_df (dataframe): Dataframe of trajectory data
        
        Raises:
            ValueError: If columns starting with `x` aren't just `x0`, `x1`, ...
                `xN`, or `x0` doesn't exist.
        
        Returns:
            list: list of all column names with particle position data
        """
        x_column_names = [
            c for c in trajectory_df.columns if c.startswith("x")
        ]
        x_column_names = sorted(x_column_names, key=lambda x: int(x[1:]))
        ints_from_column_names = [int(x[1:]) for x in x_column_names]
        expected_ints = list(range(len(x_column_names)))
        if ints_from_column_names != expected_ints:
            raise ValueError(
                "Expected columns x0, x1, ... xN and no other "
                "columns beginning with 'x'"
            )
        return x_column_names

    def _get_n_dimensions(self, trajectory_df):
        """Return the number of dimensions in this trajectory
        
        Args:
            trajectory_df (dataframe): Dataframe containing trajectory
                information 
        
        Raises:
            ValueError: If no `x0` column exists
        
        Returns:
            int: Number of dimensions
        """
        x_columns = self._get_x_column_names(trajectory_df)
        n_dimensions = len(x_columns)
        if n_dimensions == 0:
            raise ValueError("At least one dimension (x0 column) is required!")
        return n_dimensions

    def _verify_box_lengths(self, box_lengths):
        """Verifies a proper box_lengths input
        
        Args:
            box_lengths (float or ndarray): If float, the length of each side of
                a cubic box. If ndarray, it must contain `n_dimensions` values
                representing the lengths of each dimension of a rectangular box.
        
        Raises:
            TypeError: If the input is not a float or ndarray
            ValueError: If the input has the wrong dimensions or non-positive
                numbers
        
        Returns:
            ndarray: `n_dimensions` length array of simulation box lengths
        """
        if not isinstance(box_lengths, np.ndarray):
            try:
                # Assume it's the length of each side of a cube if it isn't a
                # numpy array
                box_lengths = float(box_lengths)
                box_lengths = box_lengths * np.ones(self.n_dimensions)
            except (TypeError, ValueError):
                raise TypeError("box_lengths must be a float or numpy array!")
        if box_lengths.ndim > 1 or len(box_lengths) != self.n_dimensions:
            raise ValueError(
                f"box_lengths must have length {self.n_dimensions}!"
            )
        if np.any(box_lengths <= 0):
            raise ValueError(f"box_lengths must be greater than 0!")
        return box_lengths

    def _get_one_timestep_cluster_list(self, timestep_df):
        """Returns list of Cluster objects for one timestep
        
        Args:
            timestep_df (dataframe): This should be the rows of `trajectory_df`
                corresponding to one timestep
        
        Returns:
            list: List of Cluster objects from the current timestep
        """
        particle_positions_df = timestep_df.set_index(
            "particle_id", drop=True
        ).filter(regex="x.*")
        full_graph = get_graph_from_particle_positions(
            particle_positions_df, self.box_lengths, self.cutoff_distance
        )
        cluster_graphs = [
            full_graph.subgraph(c).copy()
            for c in nx.connected_components(full_graph)
        ]
        cluster_list = []
        for cluster_graph in cluster_graphs:
            particle_id_list = sorted(list(cluster_graph.nodes))
            cluster_particle_df = particle_positions_df.query(
                "particle_id in @particle_id_list"
            )
            cluster = Cluster(
                cluster_graph,
                cluster_particle_df,
                self.box_lengths,
                self.cutoff_distance,
            )
            cluster_list.append(cluster)
        return cluster_list

    def _get_cluster_dict(self, verbosity=0):
        """Returns dictionary of one cluster list for each timestep.
        
        Args:
            verbosity (int, optional): A value greater than 0 will print some
                output to show which timesteps have been computed. Defaults to
                0.
        
        Returns:
            dict: `timestep` → `cluster_list` as key-value pairs
        """
        if self.cluster_dict is not None:
            return self.cluster_dict
        cluster_dict = dict()
        for timestep, ts_group in self.trajectory_df.groupby("timestep"):
            if verbosity > 0:
                print("Timestep:", timestep)
            cluster_list = self._get_one_timestep_cluster_list(ts_group)
            cluster_dict[timestep] = cluster_list
        return cluster_dict

    def compute_cluster_properties(
        self, properties=["n_particles"], verbosity=0
    ):
        """Returns dataframe of properties for each cluster in each timestep.
        
        Args:
            properties (list, optional): List of properties to computer for each
                cluster. Defaults to ["n_particles"].
            verbosity (int, optional): A value greater than 0 will print some
                output to show which timesteps have been computed. Defaults to
                0.
        
        Returns:
            dataframe: columns of `timestep`, `cluster_id`, and
            columns corresponding to properties in `properties` list.
        """
        properties_dict_list = list()
        for timestep, cluster_list in self.cluster_dict.items():
            if verbosity > 0:
                print("Timestep:", timestep)
            for i, cluster in enumerate(cluster_list):
                properties_dict = cluster.compute_cluster_properties(
                    properties
                )
                properties_dict["timestep"] = timestep
                properties_dict["cluster_id"] = i
                properties_dict_list.append(properties_dict)
        properties_df = pd.DataFrame(properties_dict_list)
        return properties_df

    def compute_particle_properties(
        self, properties=["coordination_number"], verbosity=0
    ):
        """Returns dataframe of properties for each particle in each timestep.
        
        Args:
            properties (list, optional): List of properties to computer for each
                particle. Defaults to ["coordination_number"].
            verbosity (int, optional): A value greater than 0 will print some
                output to show which timesteps have been computed. Defaults to
                0.

        Returns:
            dataframe: columns of `timestep`, `cluster_id`, `particle_id`, and
            columns corresponding to properties in `properties` list.
        """
        properties_df_list = list()
        for timestep, cluster_list in self.cluster_dict.items():
            for i, cluster in enumerate(cluster_list):
                properties_df = (
                    cluster.compute_particle_properties(properties)
                    .reset_index()  # Move particle_id column out of index
                    .assign(timestep=timestep)
                    .assign(cluster_id=i)
                )
                properties_df_list.append(properties_df)
        properties_df = pd.concat(properties_df_list, ignore_index=True)
        return properties_df

    def compute_bonds(self):
        """Returns dataframe of bonds for the whole trajectory.
        
        Returns:
            dataframe: Shape `(n_bonds, 4)`. Column names `particle_id_1`,
            `particle_id_2`, `timestep`, and `cluster_id`.
        """
        bonds_df_list = list()
        for timestep, cluster_list in self.cluster_dict.items():
            for i, cluster in enumerate(cluster_list):
                bonds_df = (
                    cluster.compute_bonds()
                    .assign(timestep=timestep)
                    .assign(cluster_id=i)
                )
                bonds_df_list.append(bonds_df)
        bonds_df = pd.concat(bonds_df_list, axis=0, ignore_index=True)
        return bonds_df

    def compute_bond_durations(self):
        """Returns dataframe with duration of bond for every distinct bonding
        event. If particles 3 and 5 bond, then unbond, then bond again, that is
        2 distinct bonding events.
        
        Returns:
            dataframe: Shape `(n_bond_events, 5)`. Columns `particle_id_1`,
            `particle_id_2`, `start`, `duration`, `bonded_at_end`
        """
        bonds_df = self.compute_bonds()

        def get_bond_start_and_duration(df, final_timestep, delta_timestep=1):
            df = df.copy()
            timesteps = df["timestep"].values
            dt = timesteps[1:] - timesteps[:-1]
            is_start = np.zeros(len(df))
            is_start[0] = 1
            is_start[1:] = dt > delta_timestep
            start_inds, = np.where(is_start == 1)
            start_inds = np.concatenate((start_inds, [len(is_start)]))
            durations = (start_inds[1:] - start_inds[:-1]) * delta_timestep
            start_and_duration_df = pd.DataFrame(
                dict(
                    start=timesteps[start_inds[:-1]],
                    duration=durations,
                    bonded_at_end=False,
                )
            )
            if timesteps[-1] == final_timestep:
                start_and_duration_df.loc[
                    len(start_and_duration_df) - 1, "bonded_at_end"
                ] = True
            return start_and_duration_df

        bond_durations_df = (
            bonds_df.sort_values(["particle_id_1", "particle_id_2"])
            .groupby(["particle_id_1", "particle_id_2"])
            .apply(
                get_bond_start_and_duration,
                final_timestep=bonds_df["timestep"].max(),
            )
            .reset_index()
            .drop("level_2", axis=1)
        )
        return bond_durations_df
