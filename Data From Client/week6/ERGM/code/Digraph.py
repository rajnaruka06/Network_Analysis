#
# File:    Digraph.py
# Author:  Alex Stivala
# Created: October 2017
#
#
#
# Defines the directed graph structure Digraph with arc list graph
# representations including forward and reverse arc lists for
# fast traversal in computing change statistics.
#
#

import math
import functools
import networkx as nx
from utils import int_or_na,float_or_na,NA_VALUE



class Digraph:
    """
    The network is represented as a dictionary of dictionaries.
    Nodes are indexed by integers 0..n-1. The outermost dictionary has the node
    v as a key, and dictionary as value. Then this dictionary has the neighbours
    of v as the keys, with the values as arc weights (or labels).
    This is the structure suggested by David Eppstein (UC Irvine) in
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/117228
    and as noted there it supprts easy modification by edge addition and
    removal, which is required by Algorithm EE.
    So G[i] is a dictionary with k entries, where k is the degree of node i.
    and G[i][j] exists exactly when j is a neighbour of i.
    And simple operations are:
      outdegree of node i:                   len(G[i])
      does arc i->j exist?:                  j in G[i]                 
      dict where keys are out-neighbours of i:  G[i]
      iterator over out-neighbourhood of i   G[i].iterkeys()
      in-place insert edge i->j:             G[i][j] = w [for some value w]
      in-place remove edge i->j:             G[i].pop(j) 
      copy of G (not modified when G is):    deepcopy(G)
      number of arcs:              sum([len(v.keys()) for v in G.itervalues()])
    To make these operations simple (no special cases for nodes with no
    neighbours),  the graph is initialized so that G[i] = dict() for
    all 0 <= i < n: dict(zip(range(n), [dict() for i in range(n)]))

    The format of the attributes files is header line with
    whitespace-delimited attribute names, followed by (whitespace
    delimited) attributes one line per node (corresponding to node
    number order, i.e the first non-header line is for node is 0, the
    second for 1, etc.). E.g. (categorical):

    gender   class
    1        1
    0        2
    0        3

    Binary attributes must be 0 or 1; categorical positive integers, and
    continuous floating point. For NA values -1 is used for binary and
    categorical, and NaN for continuous.

    Node attributes (binary, continuous, categorical; separately)
    are each stored in a dictionary of lists. The key of the dictionary
    is the attribute name, and the value is a list which is simply
    indexed by node id i.e. a simple list of the attribute values in
    node id order. So e.g. the categorical attribute 'class' for node id 2
    (the third node, so row 4 in data which has header)
    would be catattr['class'][2]

    Also there can be optionally be a 'zone' for each node, which is
    the snowball sampling zone: 0 for the seed nodes, 1 for nodes
    reached directly from those in zone 0, and so on. These are read
    from a text file with zone number one per line for each node in
    order, with a header line that must just be one column name "zone", and
    stored simply as a list (in node id order just as for attributes).

    """
    def __init__(self, num_v, edgelist_filename, binattr_filename=None,
                 contattr_filename=None, catattr_filename=None,
                 zone_filename=None):
        """
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
        self.G = None  # dict of dicts as described above
        self.Grev = None # version with all arcs reversed to get in-neighbours
        
        self.binattr = None # binary attributes: dict name, list by node (int not boolean)
        self.contattr = None # continuous attributes: dict name, list by node
        self.catattr = None  # categorical attributes: dict name, list by node

        # for conditional estimation on snowball sampling structure
        self.zone    = None  # node snowball zone, list by node
        self.max_zone= None  # maximum snowball zone number
        self.inner_nodes = None # list of nodes with zone < max_zone

        n = num_v


        # empty graph n nodes        
        self.G = dict(list(zip(list(range(n)), [dict() for i in range(n)])))
        self.Grev = dict(list(zip(list(range(n)), [dict() for i in range(n)])))
        
        with open(edgelist_filename) as f:
            lsplit = f.readline().split()
            while len(lsplit) >= 2:
                lsplit = lsplit[:2]  # only used first two (i,j) ignore weight
                (i, j) = list(map(int, lsplit))
                assert(i >= 1 and i <= n and j >= 1 and j <= n)
                self.insertArc(i-1, j-1)    # input is 1-based but we are 0-based
                lsplit = f.readline().split()

        # Note in the following,
        #  map(list, zip(*[row.split() for row in open(filename).readlines()]))
        # reads the data and transposes it so we have a list of columns
        # not a list of rows, which then makes it easy to convert to
        # the dict where key is column header and value is list of values
        # (converted to the appropraite data type from sting with int_or_na()
        # etc.)
        # https://stackoverflow.com/questions/6473679/transpose-list-of-lists#
        
        if binattr_filename is not None:
            self.binattr = dict([(col[0], list(map(int_or_na, col[1:]))) for col in map(list, list(zip(*[row.split() for row in open(binattr_filename).readlines()])))])
            assert(all([len(v) == n for v in self.binattr.values()]))

        if contattr_filename is not None:
            self.contattr = dict([(col[0], list(map(float_or_na, col[1:]))) for col in map(list, list(zip(*[row.split() for row in open(contattr_filename).readlines()])))])
            assert(all([len(v) == n for v in self.contattr.values()]))

        if catattr_filename is not None:
            self.catattr = dict([(col[0], list(map(int_or_na, col[1:]))) for col in map(list, list(zip(*[row.split() for row in open(catattr_filename).readlines()])))])
            assert(all([len(v) == n for v in self.catattr.values()]))

        if zone_filename is not None:
            self.zone = [int(s) for s in open(zone_filename).readlines()[1:]]
            assert(len(self.zone) == n)
            self.max_zone = max(self.zone)
            assert(min(self.zone) == 0)
            assert(len(set(self.zone)) == self.max_zone + 1) # zones must be 0,1,..,max
            # get list of nodes in inner waves, i.e. with zone < max_zone
            self.inner_nodes = [i for (i, z) in enumerate(self.zone) if z < self.max_zone]


    def numNodes(self):
        """
        Return number of nodes in digraph
        """
        return len(self.G)
    
    def numArcs(self):
        """
        Return number of arcs in digraph
        """
        return sum([len(list(v.keys())) for v in self.G.values()])
    
    def density(self):
        """
        Return the digraph density 
        """
        edges = self.numArcs()
        nodes = self.numNodes()
        return float(edges) / float(nodes*(nodes-1))

    def outdegree(self, i):
        """
        Return Out-degree of node i
        """
        return len(self.G[i])

    def indegree(self, i):
        """
        Return In-degree of node i
        """
        return len(self.Grev[i])

    def isArc(self, i, j):
        """
        Return True iff arc i -> j in digraph
        """
        return j in self.G[i]

    def outIterator(self, i):
        """
        Return iterator over out-neighbours of i
        """
        return iter(self.G[i].keys())

    def inIterator(self, i):
        """
        Return iterator over in-neighbours of i
        """
        return iter(self.Grev[i].keys())

    def insertArc(self, i, j, w = 1):
        """
        Insert arc i -> j with arc weight (or label) w, in place
        """
        self.G[i][j] = w
        self.Grev[j][i] = w

    def removeArc(self, i, j):
        """
        Delete arc i -> j in place
        """
        self.G[i].pop(j)
        self.Grev[j].pop(i)

    def printSummary(self):
        """
        Print summary of Digraph object
        """
        print('Digraph nodes = ', self.numNodes())
        print('Digraph arcs = ', self.numArcs())
        print('Digraph density = ', self.density())


        if self.binattr is not None:
            for attrname in self.binattr.keys():
                print('Binary attribute', attrname, 'has', self.binattr[attrname].count(NA_VALUE), 'NA values')
        else:
            print('No binary attributes')
        if self.contattr is not None:
            for attrname in self.contattr.keys():
                print('Continuous attribute', attrname, 'has', sum([math.isnan(x) for x in self.contattr[attrname]]), 'NA values')
        else:
            print('No continuous attributes')
        if self.catattr is not None:
            for attrname in self.catattr.keys():
                print('Categorical attribute', attrname, 'has', self.catattr[attrname].count(NA_VALUE), 'NA values')
        else:
            print('No categorical attributes')

        if self.zone is not None:
            print('There are', self.max_zone, 'snowball sample waves, with', len(self.inner_nodes), 'nodes in inner waves')
        else:
            print('No snowball zones')
            
    @functools.cache # Memoize the twoPaths function (Python 3.9)
    def mixedTwoPaths(self, i, j):
        """
        Count directed mixed two-paths for (i, j): paths i -> v -> j for some v
        (where v is distinct from both i and j, which are also distinct)
        """
        return sum([(v in self.inIterator(j)) for v in self.outIterator(i)]) if i != j else 0

    @functools.cache # Memoize the twoPaths function (Python 3.9)
    def inTwoPaths(self, i, j):
        """
        Count directed in two-paths for (i, j): paths i <- v -> j for some v
        (where v is distinct from both i and j, which are also distinct)
        """
        return sum([(v in self.inIterator(j)) for v in self.inIterator(i)]) if i != j else 0
    
    @functools.cache # Memoize the twoPaths function (Python 3.9)
    def outTwoPaths(self, i, j):
        """
        Count directed out two-paths for (i, j): paths i -> v <- j for some v
        (where v is distinct from both i and j, which are also distinct)
        """
        return sum([(v in self.outIterator(j)) for v in self.outIterator(i)]) if i != j else 0