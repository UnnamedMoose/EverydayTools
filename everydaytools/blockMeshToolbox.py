# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 16:15:37 2018

@author: artur
"""

import numpy as np

def blockMeshSpacing(N, exp, retVec=False):
    """
    Calculates the spacing of N+1 points defining N cells as will be calculated by OpenFOAM's blockMesh.
    Returns the min and max values by default but may return the entire vector if
    required. The spacings are provided in the interval from 0 to 1 -> need to dimensionalise
    afterwards.
    """
    expCoeff = exp**(1./(N-1.))
    
    if np.abs(expCoeff - 1.0) < 1e-32:
        x = np.linspace(0,1,N)
    else:    
        x = np.zeros((N+1))
        x[-1] = 1.
        
        for i in range(1,N):
            x[i] = (1.-expCoeff**(i))/(1.-expCoeff**N)
    
    if retVec:
        return x[1]-x[0], x[-1]-x[-2], x
    else:
        return x[1]-x[0], x[-1]-x[-2]

def construct3dVertices(blockPoints,span):
    """
    Duplicate the entries in block vertices array, setting the z-dimension of the new level to span
    Check if the given list is for 2D or 3D vertices and correct accordingly assuming default value of 0
        for unspecified z
    """
    newBlockPoints = np.zeros((blockPoints.shape[0]*2,3))
    
    for i in range(blockPoints.shape[0]):
        if blockPoints.shape[1] == 2:
            newBlockPoints[i,0:2] = blockPoints[i,:]
        else:
            newBlockPoints[i,:] = blockPoints[i,:]
        
        newBlockPoints[i+blockPoints.shape[0],:] = np.append(blockPoints[i,0:2],span)
        
    return newBlockPoints

def make3dBlock(pts, nPts):
    """
    Take 4 edges of a block and return 8 vertices given the total number of nodes in one
    2D mesh vertex layer
    """
    return np.append(np.array(pts), np.array([pt+nPts for pt in pts]))

def makePatch(pts, nPts):
    """ Turn a 2D patch definition described by two vertex indices into 3D """
    return pts + [pt+nPts for pt in np.flipud(pts)]

def makePatches(pts, nPts):
    """ Turn a set of 2D patch definitions into 3D """
    return [makePatch(pt, nPts) for pt in pts]
