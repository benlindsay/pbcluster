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
        self._center_of_mass_dict = None

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
        for prop in properties:
            if prop not in self._particle_property_map:
                raise ValueError(f"Property '{prop}' is not valid!")
            prop_function = self._particle_property_map[prop]
            property_df = prop_function()
            assert np.all(property_df.index == self.particle_df.index)
            particle_df = self.particle_df.join(property_df, how="left")
        return particle_df

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

    def compute_center_of_mass(self):
        """Returns cluster center of mass dictionary
        
        Returns:
            dict: `{"x0": x0, "x1": x1, ...}`
        """
        if self._center_of_mass_dict is not None:
            return self._center_of_mass_dict
        minimum_node_cuts_dict = self.compute_minimum_node_cuts()
        n_node_cuts = sum(value for value in minimum_node_cuts_dict.values())
        # If n_node_cuts is greater than 0, that means the cluster spans the
        # length of at least 1 dimension, and a center of mass can't
        # necessarily be computed.
        if n_node_cuts > 0:
            return None
        if len(self.graph) == 1:
            node = list(self.graph.nodes)[0]
            center_of_mass_dict = (
                self.particle_df.filter(regex="x.*").loc[node, :].to_dict()
            )
            self._center_of_mass_dict = center_of_mass_dict
            return center_of_mass_dict
        x_array_dict = dict()
        first = True
        pdf = self.particle_df.filter(regex="x.*")
        for node_1, node_2 in nx.dfs_edges(self.graph):
            if first:
                x_array_1 = pdf.loc[node_1, :].values
                x_array_dict[node_1] = x_array_1
                first = False
            else:
                x_array_1 = x_array_dict[node_1]
            x_array_2 = pdf.loc[node_2, :].values
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
        assert len(x_array_dict) == self.n_particles
        center_of_mass = (
            sum([xyz for xyz in x_array_dict.values()]) / self.n_particles
        )
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
        center_of_mass_dict = {
            f"x{i}": v for i, v in enumerate(center_of_mass)
        }
        self._center_of_mass_dict = center_of_mass_dict
        return center_of_mass_dict

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
        center_of_mass_dict = self.compute_center_of_mass()
        x_columns = [f"x{d}" for d in range(self.n_dimensions)]
        center_of_mass = (
            pd.DataFrame([center_of_mass_dict]).filter(x_columns).values
        )
        dx = self.particle_df[x_columns].values - center_of_mass
        dx = np.where(dx < -self.box_lengths / 2, dx + self.box_lengths, dx)
        dx = np.where(dx >= self.box_lengths / 2, dx - self.box_lengths, dx)
        if include_dx is True:
            arrays_dict = {
                f"dx_from_com_x{d}": dx[:, d] for d in range(self.n_dimensions)
            }
        else:
            arrays_dict = {}
        if include_distance is True:
            distances = np.linalg.norm(dx, axis=1)
            arrays_dict["distance_from_com"] = distances
        distance_from_com_df = pd.DataFrame(dict(arrays_dict))
        return distance_from_com_df
