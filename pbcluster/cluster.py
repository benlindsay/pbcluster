# -*- coding:utf-8 -*-
#
# cluster.py

"""Cluster module."""


class Cluster:
    """Object to store and compute data about an individual particle cluster
    """

    def __init__(self, graph):
        self._property_map = dict(n_particles=self.compute_n_particles)
        self.graph = graph
        self.properties = dict()

    def compute_properties(self, properties=["n_particles"]):
        """Compute cluster properties passed in `properties` variable
        
        Args:
            properties (list or str, optional): List of cluster properties to
                compute, or "all" to compute all available properties.
                Defaults to ["n_particles"].
        """
        if properties == "all":
            properties = self._property_map.keys()
        for prop in properties:
            if prop not in self._property_map:
                raise ValueError(f"Property '{prop}' is not valid!")
            prop_function = self._property_map[prop]
            self.properties[prop] = prop_function()

    def compute_n_particles(self):
        n_particles = self.graph.number_of_nodes()
        return n_particles
