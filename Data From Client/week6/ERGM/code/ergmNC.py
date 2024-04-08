#
# File:    ergmNC.py
# Author:  Peng Wang
# Created: Jan 2023
#
"""
    Python implementation of psudolikelihood estimation of 
    Autologistic Actor Attribute Model (ALAAM) or 
    Exponential Random Graph Model (ERGM) parameters

    Network/Graph data structure based on implementation by 
    Alex Stivala https://github.com/stivalaa/EstimNetDirected
"""

import numpy as np         # used for matrix & vector data types and functions
from functools import partial
import statsmodels.api as sm



from utils import NA_VALUE,int_or_na
from Graph import Graph
from Digraph import Digraph
from BipartiteGraph import BipartiteGraph
from BipartiteGraph import BipartiteGraph,MODE_A,MODE_B

from changeStatisticsALAAM import *
from changeStatisticsALAAMbipartite import *
from changeStatisticsERGMDirected import *
from changeStatisticsERGMNondirected import *


def run_ple_network_attr(num_v,
                        edgelist_filename, 
                        param_func_list, 
                        labels,
                        output_filename,
                        outcome_bin_filename=None,
                        binattr_filename=None,
                        contattr_filename=None,
                        catattr_filename=None,
                        directed = False,
                        bipartite = False,
                        num_vA = 0):
    
    """
    Run estimation using psudolikelihood on specified network with binary 
    and/or continuous and/or categorical attributes.
    When outcome_bin_filename is supplied, the model is assumed to be 
    autologistic actor oriented models (ALAAM), otherwise the model is
    exponential random graph models (ERGM).
    
    Parameters:
        num_v - number of vertices/nodes in the network
        edgelist_filename - filename of edgelist 
        param_func_list   - list of change statistic functions corresponding
                            to parameters to estimate
        labels            - list of strings corresponding to param_func_list
                            to label output (header line)
        output_filename   - modelling output file
        outcome_bin_filename - filename of binary outcome (node per line)
                                variable for ALAAM
        binattr_filename - filename of binary attributes (node per line)
                        Default None, in which case no binary attr.
        contattr_filename - filename of continuous attributes (node per line)
                        Default None, in which case no continuous attr.
        catattr_filename - filename of categorical attributes (node per line)
                        Default None, in which case no categorical attr.
        directed        - Default False.
                        True for directed network else undirected.
        bipartite       - Default False.
                        True for two-mode network else one-mode.

    WARNING: output files are overwritten.
    """
    
    assert(len(param_func_list) == len(labels))

    if directed:
        if bipartite:
            raise Exception("directed bipartite network not suppored")
        G = Digraph(num_v, edgelist_filename, binattr_filename, contattr_filename,
                    catattr_filename)
    else:
        if bipartite:
            G = BipartiteGraph(num_v, num_vA, edgelist_filename, binattr_filename,
                               contattr_filename, catattr_filename)
        else:
            G = Graph(num_v,edgelist_filename, binattr_filename,
                      contattr_filename, catattr_filename)

    G.printSummary()
    
    num_p = len(labels) #number of parameters
    
    if outcome_bin_filename not in ['',None]:
        # ALAAM
        y = list(map(int_or_na, open(outcome_bin_filename).read().split()[1:]))
        assert(len(y) == num_v)
        print('positive outcome attribute = ', (float(y.count(1))/len(y))*100.0, '%')
        assert( all([i in [0,1,NA_VALUE] for i in y]) )
        
        if NA_VALUE in y:
            print('Warning: outcome variable has', y.count(NA_VALUE), 'NA values')
        # convert list to numpy vector
        x = np.zeros(num_v,num_p)
        y = np.array(y) 
        for i in range(num_v):
            for j in range(num_p):
                x[i, j] = param_func_list[j](G, y, i)
        
    else: 
        #ERGM
        
        if directed:
            x = np.zeros((num_v*(num_v-1),num_p))
            y = np.zeros(num_v*(num_v-1))
            index=0
            for i in range(num_v):
                for j in range(num_v):
                    if(i==j): 
                        continue
                    if(G.isArc(i, j)):
                        y[index] = 1
                    for k in range(num_p):
                        x[index, k] = param_func_list[k](G, i, j)
                    index += 1
        
        else:
            if bipartite:
                x = np.zeros((G.num_A_nodes*G.num_B_nodes),num_p)
                y = np.zeros(G.num_A_nodes*G.num_B_nodes)
                index=0
                for i in range(G.num_A_nodes):
                    for j in range(G.num_B_nodes):
                        if(G.isEdge(i, j)):
                            y[index] = 1
                        for k in range(num_p):
                            x[index, k] = param_func_list[k](G, i, j)
                        index += 1                
            else: #nondirected one mode network
                x = np.zeros((num_v*(num_v-1)/2,num_p))
                y = np.zeros(num_v*(num_v-1)/2)
                index=0
                for i in range(num_v):
                    for j in range(i+1,num_v,1):
                        if(G.isEdge(i, j)):
                            y[index] = 1
                        for k in range(num_p):
                            x[index, k] = param_func_list[k](G, i, j)
                        index += 1                
       
    # logistic regression using sm package                 
    model = sm.Logit(y,x)
    model.exog_names[:] = labels
    try:
        result = model.fit()
        if output_filename not in [None,'']:
            with open(output_filename, 'w') as f:
                f.write('Psuodolikelihood estiamtion results\n') 
                f.write(result.summary().as_text()) 
    except Exception as e:
        print ('Error in psuodolikelihood estimation.')





def run_example():
    """
    example run on a school network
    """
    run_ple_network_attr(
        num_v=150,
        edgelist_filename='../school/Influence_arclist.txt',
        param_func_list=[
            changeArc, changeReciprocity, changeAltInStars,
            changeAltOutStars,changeAltKTrianglesT,changeAltTwoPathsT,
            partial(changeSender, attrname = "Gender"), 
            partial(changeReceiver, attrname = "Gender"), 
            partial(changeInteraction, attrname = "Gender"), 
            partial(changeContinuousSender, attrname = "CompleteYears"), 
            partial(changeContinuousReceiver, attrname = "CompleteYears"), 
            partial(changeDiff, attrname = "CompleteYears"), 
            partial(changeContinuousSender, attrname = "k6_overall"),
            partial(changeContinuousReceiver, attrname = "k6_overall"), 
            partial(changeDiff, attrname = "k6_overall"), 
            partial(changeMatching, attrname = "Class"),
        ],
        labels=[
            "Density", "Reciprocity", "Popularity", "Activity", "Closure", "Brokerage",
            "Sender(Gender)", "Receiver(Gender)", "Homophily(Gender)",
            "Sender(CompleteYears)", "Receiver(CompleteYears)", "Heterophily(CompleteYears)",
            "Sender(K6)", "Receiver(K6)", "Heterophily(K6)",
            "Matching(Class)",
        ],
        output_filename='../school/modelling_result.txt',
        binattr_filename='../school/BIN2.txt',
        contattr_filename='../school/CONT4.txt',
        catattr_filename='../school/Class.txt',
        directed= True,
    )
    
    

def main():
    run_example()

    
    



### MAIN
if __name__ == '__main__':
    model = main()
    