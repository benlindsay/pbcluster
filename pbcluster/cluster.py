# -*- coding:utf-8 -*-
#
# cluster.py

"""Cluster module."""

import networkx as nx
import numpy as np
import pandas as pd

from .utils import flatten_dict
from .utils import get_within_cutoff_matrix
from .utils import pairwise_distances


class Cluster:
    """Object to store and compute data about an individual particle cluster

    Args:
        graph (networkx Graph): Contains nodes and edges corresponding to
            particles and bonds, respectively, where a bond implies the particles
            are within a distance of `cutoff_distance` from each other.
        particle_df (dataframe): Dataframe where index is `particle_id`, and
            there are `n_dimensions` columns labelled `x0`, x1`, ... `xN`
        box_lengths (ndarray): Must contain `n_dimensions` values representing
            the lengths of each dimension of a rectangular box.
        cutoff_distance (float): Maximum distance two particles can be from
            each other to be considered part of the same cluster
    
    Attributes:
        graph (networkx Graph): Contains nodes and edges corresponding to
            particles and bonds, respectively, where a bond implies the particles
            are within a distance of `cutoff_distance` from each other.
        particle_df (dataframe): Dataframe where index is `particle_id`, and
            there are `n_dimensions` columns labelled `x0`, x1`, ... `xN`
        box_lengths (ndarray): Must contain `n_dimensions` values representing
            the lengths of each dimension of a rectangular box.
        cutoff_distance (float): Maximum distance two particles can be from
            each other to be considered part of the same cluster
        n_dimensions (int): Number of dimensions in the system
        n_particles (int): Number of particles in the cluster
    """

    def __init__(self, graph, particle_df, box_lengths, cutoff_distance):
        self._cluster_property_map = dict(
            n_particles=self.compute_n_particles,
            minimum_node_cuts=self.compute_minimum_node_cuts,
            center_of_mass=self.compute_center_of_mass,
            unwrapped_center_of_mass=self.compute_unwrapped_center_of_mass,
            rg=self.compute_rg,
            asphericity=self.compute_asphericity,
        )
        self._particle_property_map = dict(
            coordination_number=self.compute_coordination_number,
            distance_from_com=self.compute_distance_from_com,
        )
        self.graph = graph
        self.particle_df = particle_df.copy()
        self.box_lengths = box_lengths
        self.cutoff_distance = cutoff_distance
        self.n_dimensions = len(box_lengths)
        self.n_particles = len(particle_df)
        self._minimum_node_cuts_dict = None
        self._unwrapped_x_df = None
        self._unwrapped_center_of_mass_dict = None
        self._center_of_mass_dict = None
        self._gyration_tensor = None
        self._gyration_eigenvals = None
        self._rg = None

    def _split_edges_with_faces_1_dim(self, graph, dim):
        """Breaks all edges that cross the `dim`-dimension's periodic boundary
        condition (PBC). Replaces those edges with edges connecting the lower
        particle to the lower wall and the higher particle to the higher wall
        
        Args:
            graph (networkx Graph): [description]
            dim (int): Dimension in which to break the PBC.
                (0, 1, 2, ... etc.)
        
        Returns:
            networkx Graph: Copy of the input graph, modified to replace
            PBC-crossing edges with conections to "face" nodes
        """
        graph = graph.copy()
        dim_str = f"x{dim}"
        low_face_node_position = {
            f"x{d}": None for d in range(self.n_dimensions)
        }
        low_face_node_position[dim_str] = 0
        high_face_node_position = {
            f"x{d}": None for d in range(self.n_dimensions)
        }
        high_face_node_position[dim_str] = self.box_lengths[dim]
        low_face_node_str = f"{dim_str}_low"
        high_face_node_str = f"{dim_str}_high"
        graph.add_node(low_face_node_str, **low_face_node_position)
        graph.add_node(high_face_node_str, **high_face_node_position)
        edges_to_add = []
        edges_to_remove = []
        for u, v in graph.edges:
            if isinstance(u, str) or isinstance(v, str):
                # Only the face nodes are strings
                continue
            u_x = self.particle_df.loc[u, dim_str]
            v_x = self.particle_df.loc[v, dim_str]
            if np.abs(u_x - v_x) > self.cutoff_distance:
                edges_to_remove.append((u, v))
                if u_x < v_x:
                    low_node = u
                    high_node = v
                else:
                    low_node = v
                    high_node = u
                edges_to_add.append((low_face_node_str, low_node))
                edges_to_add.append((high_face_node_str, high_node))
        for u, v in edges_to_add:
            graph.add_edge(u, v)
        for u, v in edges_to_remove:
            graph.remove_edge(u, v)
        return graph

    def compute_cluster_properties(self, properties=["n_particles"]):
        """Compute cluster properties passed in `properties` variable
        
        Args:
            properties (list or str, optional): List of cluster properties to
                compute, or "all" to compute all available properties.
                Defaults to ["n_particles"].
        
        Returns:
            dict: `property_name → property_value` key-value pairs
        """
        if properties == "all":
            properties = self._cluster_property_map.keys()
        cluster_properties_dict = dict()
        for prop in properties:
            if prop not in self._cluster_property_map:
                raise ValueError(f"Property '{prop}' is not valid!")
            prop_function = self._cluster_property_map[prop]
            cluster_properties_dict[prop] = prop_function()
        cluster_properties_dict = flatten_dict(cluster_properties_dict)
        return cluster_properties_dict

    def compute_particle_properties(self, properties=["coordination_number"]):
        """Compute particle properties passed in `properties` variable
        
        Args:
            properties (list or str, optional): List of particle properties to
                compute, or "all" to compute all available properties.
                Defaults to ["coordination_number"].
        
        Returns:
            dataframe: Shape (`n_particles`, `n_dimensions` + `n_properties`)
            `particle_id` as index, `x*` and particle property columns. 
        """
        if properties == "all":
            properties = self._particle_property_map.keys()
        particle_df = self.particle_df.copy()
        for prop in properties:
            if prop not in self._particle_property_map:
                raise ValueError(f"Property '{prop}' is not valid!")
            prop_function = self._particle_property_map[prop]
            property_df = prop_function()
            assert np.all(property_df.index == self.particle_df.index)
            particle_df = particle_df.join(property_df, how="left")
        return particle_df

    def compute_bonds(self):
        """Returns a dataframe with 2 columns, where each row has a pair of `particle_id`s associated with bonded particles
        
        Returns:
            dataframe: Shape `(n_bonds, 2)`. Column names `particle_id_1` and
            `particle_id_2`. 
        """
        bonds_df = pd.DataFrame(
            self.graph.edges(), columns=["particle_id_1", "particle_id_2"]
        ).sort_values(["particle_id_1", "particle_id_2"])
        return bonds_df

    ######################
    # Cluster Properties #
    ######################

    def compute_n_particles(self):
        """Returns the number of particles in the cluster
        
        Returns:
            int: number of particles in the cluster
        """
        return self.n_particles

    def compute_minimum_node_cuts(self):
        """Returns dictionary of minimum node cuts required to break the
        connection between faces normal to a given direction.
        
        Returns:
            dict: `dimension_str → minimum_node_cuts` key-value pairs
        """
        # If this was already computed, return the stored dictionary
        if self._minimum_node_cuts_dict is not None:
            return self._minimum_node_cuts_dict
        minimum_node_cuts_dict = dict()
        for dim in range(self.n_dimensions):
            split_graph = self._split_edges_with_faces_1_dim(self.graph, dim)
            node_cut = nx.minimum_node_cut(
                split_graph, f"x{dim}_low", f"x{dim}_high"
            )
            minimum_node_cuts_dict[f"x{dim}"] = len(node_cut)
        # Store this because other computations like center of mass rely on it
        self._minimum_node_cuts_dict = minimum_node_cuts_dict
        return minimum_node_cuts_dict

    def compute_center_of_mass(self, wrapped=True):
        """Returns cluster center of mass dictionary

        Args:
            wrapped (boolean, optional): If True, a center of mass that
                falls outside the box bounds is forced to be in range
                [0, `box_lengths[d]`) for each dimension `d`. If using this to
                compare to unwrapped particle coordinates, leave as False.
                Defaults to False.
        
        Returns:
            dict: `{"x0": x0, "x1": x1, ...}`
        """
        if wrapped is True and self._center_of_mass_dict is not None:
            return self._center_of_mass_dict
        if (
            wrapped is False
            and self._unwrapped_center_of_mass_dict is not None
        ):
            return self._unwrapped_center_of_mass_dict
        unwrapped_x_df = self._compute_unwrapped_x()
        if unwrapped_x_df is None:
            # _compute_unwrapped_x returns None if the particles bridge the
            # faces of at least 1 dimension. If that's the case, center of mass
            # can't necessarily be computed either
            center_of_mass = {
                f"x{d}": np.nan for d in range(self.n_dimensions)
            }
            self._unwrapped_center_of_mass_dict = center_of_mass
            self._center_of_mass_dict = center_of_mass
            return center_of_mass
        unwrapped_x_df.columns = [f"x{d}" for d in range(self.n_dimensions)]
        center_of_mass = unwrapped_x_df.mean(axis=0)
        self._unwrapped_center_of_mass_dict = center_of_mass.to_dict()
        if wrapped is True:
            while np.any(center_of_mass < 0) or np.any(
                center_of_mass > self.box_lengths
            ):
                center_of_mass = np.where(
                    center_of_mass < 0,
                    center_of_mass + self.box_lengths,
                    center_of_mass,
                )
                center_of_mass = np.where(
                    center_of_mass >= self.box_lengths,
                    center_of_mass - self.box_lengths,
                    center_of_mass,
                )
            self._center_of_mass_dict = {
                f"x{i}": v for i, v in enumerate(center_of_mass)
            }
            return self._center_of_mass_dict
        else:
            return self._unwrapped_center_of_mass_dict

    def compute_unwrapped_center_of_mass(self):
        """Returns unwrapped center of mass, meaning it's the center of mass of
        the unwrapped particle coordinates, and isn't necessarily inside the box
        coordinates.
        
        Returns:
            dict: Unwrapped center of mass coordinates, `"x*" → number`
            key-value pairs. Technically, no max or min restriction, but
            probably within 1 period of the box bounds.
        """
        return self.compute_center_of_mass(wrapped=False)

    def _compute_gyration_tensor(self):
        """Returns cluster gyration tensor
        
        Returns:
            ndarray: Shape (`n_dimensions`, `n_dimensions`) gyration tensor
        """
        if self._gyration_tensor is not None:
            return self._gyration_tensor
        dx_from_com = self.compute_distance_from_com(
            include_distance=False
        ).values
        if np.isnan(dx_from_com).sum() > 0:
            # If there are NaN values, that means it's a percolated cluster
            gyration_tensor = np.nan * np.ones(
                (self.n_dimensions, self.n_dimensions)
            )
        else:
            # This implements the first equation in
            # https://en.wikipedia.org/wiki/Gyration_tensor
            gyration_tensor = (
                np.sum(
                    dx_from_com[:, :, None] * dx_from_com[:, None, :], axis=0
                )
                / self.n_particles
            )
            # Make sure gyration_tensor is symmetric
            assert np.allclose(gyration_tensor, gyration_tensor.T)
        self._gyration_tensor = gyration_tensor
        return gyration_tensor

    def _compute_gyration_eigenvals(self):
        """Returns numpy array of eigenvalues of the gyration tensor. Values are
        not sorted.
        
        Returns:
            ndarry: Shape (`n_dimensions`,) eigenvalues array
        """
        if self._gyration_eigenvals is not None:
            return self._gyration_eigenvals
        gyration_tensor = self._compute_gyration_tensor()
        if np.isnan(gyration_tensor).sum() > 0:
            # NaNs exist if the cluster is percolated
            eigenvals = np.nan * np.ones(self.n_dimensions)
        else:
            eigenvals, eigenvecs = np.linalg.eig(gyration_tensor)
            # Make sure all the numbers are real
            assert np.isclose(np.sum(np.abs(np.imag(eigenvals))), 0.0)
            # Drop the 0 imaginary part if it's there
            eigenvals = eigenvals.real
        self._gyration_eigenvals = eigenvals
        return eigenvals

    def compute_rg(self):
        """Returns cluster radius of gyration.
        
        Returns:
            float: Cluster radius of gyration
        """
        if self._rg is not None:
            return self._rg
        eigenvals = self._compute_gyration_eigenvals()
        if np.any(np.isnan(eigenvals)):
            rg = np.nan
        else:
            rg = np.sqrt(np.sum(eigenvals ** 2))
        self._rg = rg
        return rg

    def compute_asphericity(self):
        """Returns cluster asphericity
        (see https://en.wikipedia.org/wiki/Gyration_tensor#Shape_descriptors)
        
        Returns:
            float: Asphericity, normalized by radius of gyration squared
        """
        rg = self.compute_rg()
        if rg == 0.0:
            asphericity = 0
        elif np.isnan(rg):
            asphericity = np.nan
        else:
            scaled_eigenvals = self._compute_gyration_eigenvals() / rg
            evals_squared = np.sort(scaled_eigenvals ** 2)
            asphericity = evals_squared[-1] - np.mean(evals_squared[:-1])
        return asphericity

    #######################
    # Particle Properties #
    #######################

    def compute_coordination_number(self):
        """Returns a dataframe of coordination numbers corresponding to each
        particle in the cluster
        
        Returns:
            dataframe: Coordination numbers for particles in the cluster. Index
            is `particle_id`s and matches `particle_df.index`
        """
        distances = pairwise_distances(self.particle_df, self.box_lengths)
        within_cutoff_matrix = get_within_cutoff_matrix(
            distances, self.cutoff_distance
        )
        coordination_numbers = within_cutoff_matrix.sum(axis=1).astype(int)
        coordination_numbers_df = pd.DataFrame(
            dict(coordination_number=coordination_numbers),
            index=self.particle_df.index,
        )
        return coordination_numbers_df

    def _compute_unwrapped_x(self):
        """Returns unwrapped particle coordinates dataframe
        
        Returns:
            dataframe: Index is `particle_id`, matching index of `particle_df`.
            Columns are `unwrapped_x*` where `*` represents 0, 1, ...
            `n_particles`
        """
        if self._unwrapped_x_df is not None:
            return self._unwrapped_x_df
        minimum_node_cuts_dict = self.compute_minimum_node_cuts()
        n_node_cuts = sum(value for value in minimum_node_cuts_dict.values())
        # If n_node_cuts is greater than 0, that means the cluster spans the
        # length of at least 1 dimension, and a center of mass can't
        # necessarily be computed.
        if n_node_cuts > 0:
            return None
        column_names_1 = [f"x{d}" for d in range(self.n_dimensions)]
        column_names_2 = [f"unwrapped_x{d}" for d in range(self.n_dimensions)]
        if len(self.graph) == 1:
            unwrapped_x_df = self.particle_df.filter(column_names_1).copy()
            unwrapped_x_df.columns = column_names_2
            return unwrapped_x_df
        x_array_dict = dict()
        first = True
        x_df = self.particle_df.filter(column_names_1)
        for node_1, node_2 in nx.dfs_edges(self.graph):
            if first:
                x_array_1 = x_df.loc[node_1, :].values
                x_array_dict[node_1] = x_array_1
                first = False
            else:
                x_array_1 = x_array_dict[node_1]
            x_array_2 = x_df.loc[node_2, :].values
            dx_array = x_array_2 - x_array_1
            dx_array = np.where(
                dx_array < -self.box_lengths / 2,
                dx_array + self.box_lengths,
                dx_array,
            )
            dx_array = np.where(
                dx_array >= self.box_lengths / 2,
                dx_array - self.box_lengths,
                dx_array,
            )
            x_array_2 = x_array_1 + dx_array
            x_array_dict[node_2] = x_array_2
        unwrapped_x_df = pd.DataFrame(x_array_dict).transpose().sort_index()
        assert np.all(unwrapped_x_df.index == self.particle_df.index)
        assert unwrapped_x_df.shape == (self.n_particles, self.n_dimensions)
        unwrapped_x_df.columns = column_names_2
        self._unwrapped_x_df = unwrapped_x_df
        return unwrapped_x_df

    def compute_distance_from_com(
        self, include_dx=True, include_distance=True
    ):
        """Returns dataframe of distances from the center of mass for each
        particle
        
        Args:
            include_dx (bool, optional): If True, includes `dx_from_com_x*`
                columns. Defaults to True.
            include_distance (bool, optional): If True, includes
                `distance_from_com` column. Defaults to True
        
        Raises:
            ValueError: both include_dx and include_distance are False
        
        Returns:
            dataframe: Index is `particle_id` (matching index of `particle_df`),
            columns are `distance_from_com` (Euclidean distance from center of
            mass), and `dx_from_com_x*` (Vector difference) where `*` represents
            0, 1, ... `n_particles`.
        """
        if include_dx is False and include_distance is False:
            raise ValueError(
                "one of include_dx or include_distance must be True"
            )
        x_columns = [f"x{d}" for d in range(self.n_dimensions)]
        unwrapped_x_df = self._compute_unwrapped_x()
        if unwrapped_x_df is None:
            center_of_mass_dict = {xc: None for xc in x_columns}
            distance_from_com_df = pd.DataFrame(index=self.particle_df.index)
            if include_dx is True:
                for d in range(self.n_dimensions):
                    distance_from_com_df[f"dx_from_com_x{d}"] = np.nan
            if include_distance is True:
                distance_from_com_df["distance_from_com"] = np.nan
        else:
            unwrapped_x = unwrapped_x_df.values
            center_of_mass_dict = self.compute_center_of_mass(wrapped=False)
            center_of_mass = (
                pd.DataFrame([center_of_mass_dict]).filter(x_columns).values
            )
            dx = unwrapped_x - center_of_mass
            if include_dx is True:
                arrays_dict = {
                    f"dx_from_com_x{d}": dx[:, d]
                    for d in range(self.n_dimensions)
                }
            else:
                arrays_dict = {}
            if include_distance is True:
                distances = np.linalg.norm(dx, axis=1)
                arrays_dict["distance_from_com"] = distances
            distance_from_com_df = pd.DataFrame(
                dict(arrays_dict), index=self.particle_df.index
            )
        return distance_from_com_df
