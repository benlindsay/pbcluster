# -*- coding:utf-8 -*-
#
# cluster.py

"""Cluster module."""

import pandas as pd

from .utils import get_within_cutoff_matrix
from .utils import pairwise_distances


class Cluster:
    """Object to store and compute data about an individual particle cluster
    """

    def __init__(self, graph, particle_df, box_lengths, cutoff_distance):
        self._cluster_property_map = dict(n_particles=self.compute_n_particles)
        self._particle_property_map = dict(
            coordination_number=self.compute_coordination_number
        )
        self.graph = graph
        self.particle_df = particle_df.copy()
        self.box_lengths = box_lengths
        self.cutoff_distance = cutoff_distance
        self.cluster_properties_dict = dict()

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

    ######################
    # Cluster Properties #
    ######################

    def compute_n_particles(self):
        """Returns the number of particles in the cluster
        
        Returns:
            int: number of particles in the cluster
        """
        n_particles = self.graph.number_of_nodes()
        return n_particles

    #######################
    # Particle Properties #
    #######################

    def compute_coordination_number(self):
        """Returns a numpy array of coordination numbers corresponding to each
        particle in the cluster
        
        Returns:
            ndarray: Coordination numbers for particles in the cluster
        """
        distances = pairwise_distances(self.particle_df, self.box_lengths)
        within_cutoff_matrix = get_within_cutoff_matrix(
            distances, self.cutoff_distance
        )
        coordination_numbers = within_cutoff_matrix.sum(axis=1).astype(int)
        return coordination_numbers
