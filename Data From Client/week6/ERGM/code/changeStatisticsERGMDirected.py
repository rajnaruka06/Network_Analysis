#!/usr/bin/env python
#
# File:    changeStatisticsDirected.py
# Author:  Maksym Byshkin, Alex Stivala
# Created: October 2017
#
#
#
# Functions to compute change statistics.  Each function takes a
# Digraph G (which contains graph itself, along with binary
# attributes and two-path matrices for fast computation) and returns
# the change statistic for adding the edge i,j
#
# These functions are adapted from the original PNet code by Peng Wang:
#
#   Wang P, Robins G, Pattison P. PNet: A program for the simulation and
#   estimation of exponential random graph models. University of
#   Melbourne. 2006.
#

import math
import numpy as np         # used for matrix & vector data types and functions
import re

from Digraph import Digraph

decay = 2.0  #decay factor for alternating statistics (lambda a python keyword)



def changeArc(G, i, j,  attrname = None):
    """
    change statistic for Arc
    """
    return 1

def changeReciprocity(G, i, j,  attrname = None):
    """
    change statistic for Reciprocity
    """
    return int(G.isArc(j, i))

def changeAltInStars(G, i, j,  attrname = None):
    """
    change statistic for alternating k-in-stars (popularity spread, AinS)
    """
    assert(decay > 1)
    return decay * (1 - (1-1/decay)**G.indegree(j))

def changeAltOutStars(G, i, j,  attrname = None):
    """
    change statistic for alternating k-out-stars (activity spread, AoutS)
    """
    assert(decay > 1)
    return decay * (1 - (1-1/decay)** G.outdegree(i))

def changeAltKTrianglesT(G, i, j,  attrname = None):
    """
    change statistic for alternating k-triangles AT-T (path closure)
    """
    assert(decay > 1)
    delta = 0
    for v in G.outIterator(i):
        assert(G.isArc(i, v))
        if v == i or v == j:
            continue
        if G.isArc(j, v):
            #delta += (1-1/decay)**G.mixedTwoPaths(i, v]
            delta += (1-1/decay)**G.mixedTwoPaths(i, v)
    for v in G.inIterator(i):
        assert(G.isArc(v, i))
        if v == i or v == j:
            continue
        if G.isArc(v, j):
            delta += (1-1/decay)**G.mixedTwoPaths(v, j)
    delta += decay * (1 - (1-1/decay)**G.mixedTwoPaths(i, j))
    return delta                 

def changeAltKTrianglesC(G, i, j,  attrname = None):
    """
    change statistic for alternating k-triangles AT-C (cyclic closure)
    """
    assert(decay > 1)
    delta = 0
    for v in G.inIterator(i):
        assert(G.isArc(v, i))
        if v == i or v == j:
            continue
        if G.isArc(j, v):
            delta += ( (1-1/decay)**G.mixedTwoPaths(i, v) +
                       (1-1/decay)**G.mixedTwoPaths(v, j) )
    delta += decay * (1 - (1-1/decay)**G.mixedTwoPaths(j, i))
    return delta                 

def changeAltTwoPathsT(G, i, j,  attrname = None):
    """
    change statistic for alternating two-paths A2P-T (multiple 2-paths)
    """
    assert(decay > 1)
    delta = 0
    for v in G.outIterator(j):
        assert(G.isArc(j, v))
        if v == i or v == j:
            continue
        delta += (1-1/decay)**G.mixedTwoPaths(i, v)
    for v in G.inIterator(i):
        assert(G.isArc(v, i))
        if v == i or v == j:
            continue
        delta += (1-1/decay)**G.mixedTwoPaths(v, j)
    return delta

def changeAltTwoPathsD(G, i, j,  attrname = None):
    """
    change statistic for alternating two-paths A2P-D (shared popularity)
    """
    assert(decay > 1)
    delta = 0
    for v in G.outIterator(i):
        #if(G.isArc(i, v)):
            if v == i or v == j:
                continue
            if G.OutTwoPathGraph.isArc(j,v):
                delta += (1-1/decay)**G.outTwoPaths(j,v)
    return delta

def changeAltTwoPathsTD(G, i, j,  attrname = None):
    """
    change statistic for altnernating two-paths A2P-TD (shared
    popularity + multiple two-paths), adjusting for multiple counting
    """
    return 0.5*(changeAltTwoPathsT(G, i, j) + changeAltTwoPathsD(G, i, j))

def changeSender(G, i, j,  attrname):
    """
    change statistic for Sender
    """
    return G.binattr[attrname][i]

def changeReceiver(G, i, j,  attrname):
    """
    change statistic for Receiver
    """
    return G.binattr[attrname][j]

def changeInteraction(G, i, j,  attrname):
    """
    change statistic for Interaction
    """
    return G.binattr[attrname][i] * G.binattr[attrname][j]

def changeMatching(G, i, j,  attrname):
    """
    change statistic for categorical matching
    """
    return G.catattr[attrname][i] == G.catattr[attrname][j]

def changeMatchingReciprocity(G, i, j,  attrname):
    """
    change statistic for categorical matching reciprocity
    """
    return G.catattr[attrname][i] == G.catattr[attrname][j] and G.isArc(j, i)


def changeMismatching(G, i, j,  attrname):
    """
    change statistic for categorical mismatching
    """
    return G.catattr[attrname][i] != G.catattr[attrname][j]

def changeMismatchingReciprocity(G, i, j,  attrname):
    """
    change statistic for categorical mismatching reciprocity
    """
    return G.catattr[attrname][i] != G.catattr[attrname][j] and G.isArc(j, i)



def changeSink(G, i, j,  attrname = None): pass
def changeSource(G, i, j,  attrname = None): pass
def changeIsolates(G, i, j,  attrname = None): pass
def changeTwoPath(G, i, j,  attrname = None): pass
def changeInTwoStars(G, i, j,  attrname = None): pass
def changeOutTwoStars(G, i, j,  attrname = None): pass
def changeTransitiveTriad(G, i, j,  attrname = None): pass
def changeCyclicTriad(G, i, j,  attrname = None): pass
def changeAltKTrianglesD(G, i, j,  attrname = None): pass
def changeAltKTrianglesU(G, i, j,  attrname = None): pass
def changeAltTwoPathsU(G, i, j,  attrname = None): pass
def changeLoop(G, i, j,  attrname = None): pass
def changeLoopInteraction(G, i, j,  attrname = None): pass
def changeMismatchingTransitiveTriad(G, i, j,  attrname = None): pass
def changeMismatchingTransitiveTies(G, i, j,  attrname = None): pass

def changeContinuousSender(G, i, j,  attrname): 
    return G.contattr[attrname][i]
    
def changeContinuousReceiver(G, i, j,  attrname):   
    return G.contattr[attrname][j]

def changeDiff(G, i, j,  attrname): 
    return abs(G.contattr[attrname][i] - G.contattr[attrname][j])
               
def changeDiffReciprocity(G, i, j,  attrname = None): pass
def changeDiffSign(G, i, j,  attrname = None): pass
def changeDiffDirSR(G, i, j,  attrname = None): pass
def changeDiffDirRS(G, i, j,  attrname = None): pass
def changeJaccardSimilarity(G, i, j,  attrname = None): pass
def changeGeoDistance(G, i, j,  attrname = None): pass
def changeLogGeoDistance(G, i, j,  attrname = None): pass
def changeEuclideanDistance(G, i, j,  attrname = None): pass
