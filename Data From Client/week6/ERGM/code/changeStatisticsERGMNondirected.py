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


from Graph import Graph

decay = 2.0  #decay factor for alternating statistics (lambda a python keyword)

def changeEdge(G,i,j, attrname = None): 
    return 1

def changeTwoStars(G,i,j, attrname = None): 
    return G.degree(i)+G.degree(j)

def changeAltStars(G, i, j, attrname = None):
    """
    change statistic for alternating-stars (popularity spread, AinS)
    """
    assert(decay > 1)
    return decay*(2-(1-1/decay)**G.degree(i)-(1-1/decay)**G.degree(j))

def changeAltKTriangles(G, i, j, attrname = None):
    """
    change statistic for alternating k-triangles AT (path closure)
    """
    assert(decay > 1)
    delta = 0
    for v in G.neighbourIterator(i):
        if v == i or v == j: continue
        if G.isEdge(j,v):
            delta+=(1-1/decay)**G.twoPaths(i,v) +(1-1/decay)**G.twoPaths(j,v)
    delta+=decay*(1- (1-1/decay)**G.twoPaths(i,j))
    return delta

def changeAltTwoPaths(G,i,j, attrname = None): 
    """
    change statistic for alternating two-paths A2P (multiple 2-paths)
    """    
    assert(decay > 1)
    delta = 0
    for v in G.neighbourIterator(i):
        if v == i or v == j: continue
        delta+=(1-1/decay)**G.twoPaths(j,v)
    for v in G.neighbourIterator(j):
        if v == i or v == j: continue
        delta+=(1-1/decay)**G.twoPaths(i,v)
    return delta 

def changeActivity(G, i, j, attrname):
    """
    change statistic for Activity
    """
    return G.bin_attr_dataframe[attrname][i]

