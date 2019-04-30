# -*- coding: utf-8 -*-
#
# trajectory.py

import numpy as np
import pandas as pd

"""Trajectory module."""


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

    Attributes:
        trajectory_df (dataframe): Dataframe containing trajectory data with
            columns `timestep`, `particle_id`, `particle_type`,
            `x0`, x1`, ... `xN`
    """

    def __init__(self, trajectory_data):
        if isinstance(trajectory_data, pd.DataFrame):
            self.trajectory_df = trajectory_data.copy()
        elif isinstance(trajectory_data, np.ndarray):
            self.trajectory_df = self._convert_ndarray_to_df(trajectory_data)
        self.trajectory_df = self._verify_dataframe(self.trajectory_df)

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
        trajectory_df = pd.DataFrame(trajectory_array, columns=x_columns)
        timestep_array = (
            np.arange(n_timesteps)[:, None] * np.ones(n_particles)[None, :]
        ).flatten()
        trajectory_df["timestep"] = timestep_array
        trajectory_df["particle_id"] = np.arange(n_timesteps * n_particles)
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
        x_columns = [c for c in trajectory_df.columns if c.startswith("x")]
        x_columns = sorted(x_columns, key=lambda x: int(x[1:]))
        if [int(x[1:]) for x in x_columns] != list(range(len(x_columns))):
            raise ValueError(
                "Expected columns x0, x1, ... xN and no other "
                "columns beginning with 'x'"
            )
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
        # Arrange columns in desired order
        trajectory_df = trajectory_df[
            ["timestep", "particle_id", "particle_type"] + x_columns
        ]
        return trajectory_df
