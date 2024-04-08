#
# File:    BipartiteGraph.py
# Author:  Alex Stivala
# Created: August 2022
#
# Defines the bipartite undirected graph structure BipartiteGraph
# with edge list graph, as a subclass of Graph
#

import sys
import networkx as nx
from Graph import Graph
from SparseMatrix import SparseMatrix



# Mode (node type) of a node for bipartite (two-mode) networks
MODE_A = 0
MODE_B = 1


class BipartiteGraph(Graph):
    """
    BipartiteGraph is a bipartite (two-mode) graph, implemented as a
    subclass of the undirected graph class Graph.

    
    This reads from the Pajek format for bipartite (two-mode) netwowkrs, 
    in which the first lines should be e.g.
    *vertices 36 10
    first number is total number of nodes
    second number is number of mode A nodes
    the rest are mode B - conventionally in the affiliation
    matrix the rows are mode A and the columns mode B, e.g. mode A is
    actors and mode B is their affiliations.
    They must be numbered 1 ... N where N = num_A + num_B
    so nodes 1 .. num_A are type A and num_A+1 .. N are type B
    see e.g. http://www.pfeffer.at/txt2pajek/txt2pajek.pdf
    
    """

    def __init__(self, num_v, num_vA, edgelist_filename, binattr_filename=None,
                 contattr_filename=None, catattr_filename=None,
                 zone_filename=None):
        """
        Construct graph from Pajek format network and binary attributes.

        Parameters:
            edgelist_filename - edge list 
            binattr_filename  - binary attributes
                                Default None: no binary attributes loaded
            contattr_filename - continuous attributes
                                Default None: no continuous attributes loaded
            catattr_filename - categorical attributes
                                Default None: no categorical attributes loaded
            zone_filename    - snowball sample zone for each node
                                Deafult None: no zone information loaded
        """

        self.num_A_nodes = num_vA
        self.num_B_nodes = num_v - num_vA
        # sparse matrix of two-path counts
        self.twoPathsMatrix = SparseMatrix(num_vA+num_vB)

        super().__init__(num_v, edgelist_filename, binattr_filename,
                         contattr_filename, catattr_filename, zone_filename)

        
    def density(self):
        """
        Return the graph density 
        """
        edges = self.numEdges()
        return float(edges) / float(self.num_A_nodes * self.num_B_nodes)

    
    def printSummary(self):
        """
        Print summary of Graph object
        """
        print('Bipartite graph')
        print('number of mode A nodes = ', self.num_A_nodes)
        print('number of mode B nodes = ', self.num_B_nodes)
        super().printSummary()


    def bipartite_node_mode(self, i):
        """
        Return node type (mode) of node i
        """
        return MODE_A if i < self.num_A_nodes else MODE_B
                            

    def insertEdge(self, i, j):
        """
        Insert edge i -- j in place
        """
        if self.bipartite_node_mode(i) == self.bipartite_node_mode(j):
            raise ValueError("edge in bipartite graph inserted between nodes in same mode")
        super().insertEdge(i, j)
        self.updateTwoPathsMatrix(i, j)

    def nodeModeIterator(self, mode):
        """
        Return iterator over nodes of graph with supplied mode
        (MODE_A or MODE_B)
        """
        return filter(
            lambda v: self.bipartite_node_mode(v) == mode, self.G.keys())

    def updateTwoPathsMatrix(self, i, j):
        """
        Update the two-paths sparse matrix used for fast computation
        of some change statistics (specifically 4-cycles), for addition
        of the edge i -- j.
        """
        for u in self.neighbourIterator(i):
            if u == i or u == j:
                continue
            self.twoPathsMatrix.incrementValue(u, j)
            self.twoPathsMatrix.incrementValue(j, u)
        for u in self.neighbourIterator(j):
            if u == i or u == j:
                continue
            self.twoPathsMatrix.incrementValue(u, i)
            self.twoPathsMatrix.incrementValue(i, u)
