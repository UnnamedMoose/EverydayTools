import numpy as np
import matplotlib.pyplot as plt
import os

import blockMeshToolbox
import OpenFOAMiO as OFiO

def rotatePoint(point, theta, x0=0.):
    """
    rotates a specified point in 2D space about a given origin (defaults to [0,0])
    by an angle theta [in degrees, +ve clockwise]
    """
    theta *= -np.pi/180;
    x = point[0]*np.cos(theta)+point[1]*np.sin(theta)
    y = point[1]*np.cos(theta)-point[0]*np.sin(theta)
    return np.array([x,y])

"""
# TODO changes which would make the whole thing better
- for nose domain, instead of projecting along origin-offset vector, find point
    on the domain circle that intersects the normal vector at that x/c
- interpolating the curves along the s-parameter to have a more uniform point
    distribution would probably help with better meshing in OF
- the forward and aft arc points might get on the wrong side of the mesh vertices
    for particularly unusual foil geometries; maybe better to use average of projection
    vectors fwd and aft?
- front and back patch points are the same as the vertices defining 2D blocks,
    is there a way to unify them neatly?
"""

def makeFoilGrid(xc, xu, yu, xl, yl, case,
    dsNearZone = 0.1, # size of the near-field mesh zone
    Rdomain = 25., # radius of the domain
    span = 1., # span
    scale = 1.0, # for uniformly scaling the mesh
    xcNose = 0.04, # x/c from which to start the nose mesh block
    xcAftOverhang = 0.04, # extend offset domain by this much aft of TE for better cell quality
    spreadNoseAngle = 5.0, # multiplier for the angle between nose extrusion vectors
    spreadTeAngle = 2.5,
    #
    nCellsSpan = 1, # spanwise no. cells
    #
    nCellsWall = 50, # wall-normal no. cells in near- and far-field
    nCellsOuter = 75,
    #
    nCellsNose = 50,#35, # no. cells along the foil surface
    nCellsTe = 16,
    nCellsUpper = 150,
    nCellsLower = 150,
    #
    meshRefParam = 1., # for uniformly scaling default mesh size
    #
    # foil-normal directions; >1 to cluster points closer to the foil
    expOuter = 120.,
    expWall = 100.,
    # foil surface
    expLongUpperLe = 7.,#3.,
    expLongUpperTe = 7.,#3.,
    expLongLowerLe = 7.,#4.,
    expLongLowerTe = 7.,#4.,
    # at the near-field block edge; >1 to cluster points towards block vertices
    expLongUpperOffset = 0.7,#0.5,
    expLongLowerOffset = 0.7,#0.5,
    expNoseOffset = 0.8,
    expTeOffset = 1.6,
    # outer domain; >1 to cluster points towards block vertices
    expLongUpperOuter = 0.2,
    expLongLowerOuter = 0.2,
    expNoseOuter = 1.5,
    expTeOuter = 2.5,
    #
    plotDomain = False
    ):

    # ===
    # offset the foil surface to create fine mesh zone
    
    # calculate the surface normal gradients at each point
    # NOTE: normals are pointing inwards due to old notaion; no matter, just take the negative of the vector
    nu = np.zeros((xu.shape[0], 2))
    nl = np.zeros((xl.shape[0], 2))
    
    # TE
    nu[-1,0] = (xu[-2]-xu[-1])
    nu[-1,1] = (yu[-2]-yu[-1])
    nl[-1,0] = (xl[-2]-xl[-1])
    nl[-1,1] = (yl[-2]-yl[-1])
    
    # rotate and non-dimensionalise
    nu[-1,:] = rotatePoint(nu[-1,:],90.) / np.sum(nu[-1,:]**2.)**0.5
    nl[-1,:] = rotatePoint(nl[-1,:],-90.) / np.sum(nl[-1,:]**2.)**0.5
    
    # LE
    nu[0,0] = (xu[1]-xu[0])
    nu[0,1] = (yu[1]-yu[0])
    nl[0,0] = (xl[1]-xl[0])
    nl[0,1] = (yl[1]-yl[0])
    
    nu[0,:] = rotatePoint(nu[0,:],-90.) / np.sum(nu[0,:]**2.)**0.5
    nl[0,:] = rotatePoint(nl[0,:],90.) / np.sum(nl[0,:]**2.)**0.5
    
    # make sure the normal gradients for upper and lower surfaces are the same at the LE (should be v. close anyway for any foil)
    nu[0,:] = (nu[0,:]+nl[0,:])/2.
    nl[0,:] = nu[0,:]
    
    # internal points
    for i in range(1, xu.shape[0]-1):
        nu[i,:] =  (rotatePoint([xu[i]-xu[i-1],yu[i]-yu[i-1]],-90.) + rotatePoint([xu[i+1]-xu[i],yu[i+1]-yu[i]],-90.))/2.
        nu[i,:] /=  np.sum(nu[i,:]**2.)**0.5
        
    for i in range(1, xl.shape[0]-1):
        nl[i,:] =  (rotatePoint([xl[i]-xl[i-1],yl[i]-yl[i-1]],90.) + rotatePoint([xl[i+1]-xl[i],yl[i+1]-yl[i]],90.))/2.
        nl[i,:] /=  np.sum(nl[i,:]**2.)**0.5
    
    # shift the offsets along the normal vectors
    xuOffset = xu - nu[:,0]*dsNearZone
    yuOffset = yu - nu[:,1]*dsNearZone
    xlOffset = xl - nl[:,0]*dsNearZone
    ylOffset = yl - nl[:,1]*dsNearZone
    
    # ===
    # rearrange the offsets into arrays holding the actual airfoil (or near field edge) points
    iNose = np.argmin(np.abs(xcNose - xc))
    xNose = np.flipud(xu[:iNose ])
    xNose = np.append(xNose, xl[:iNose ])
    
    yNose = np.flipud(yu[:iNose ])
    yNose = np.append(yNose,yl[:iNose ])
    
    xUpper = xu[iNose-1:]
    yUpper = yu[iNose-1:]
    
    xLower = xl[iNose-1:]
    yLower = yl[iNose-1:]
    
    xNoseOffset = np.flipud(xuOffset[:iNose ])
    xNoseOffset = np.append(xNoseOffset, xlOffset[:iNose ])
    
    yNoseOffset = np.flipud(yuOffset[:iNose ])
    yNoseOffset = np.append(yNoseOffset, ylOffset[:iNose ])
    
    xUpperOffset = xuOffset[iNose-1:];
    yUpperOffset = yuOffset[iNose-1:];
    
    xLowerOffset = xlOffset[iNose-1:];
    yLowerOffset = ylOffset[iNose-1:];
    
    # extend the offset to alleviate issues with high opening angles at the TE
    vu = np.array([xUpperOffset[-1]-xUpperOffset[-2], yUpperOffset[-1]-yUpperOffset[-2]])
    vl = np.array([xLowerOffset[-1]-xLowerOffset[-2], yLowerOffset[-1]-yLowerOffset[-2]])
    vu /= np.sqrt(np.sum(vu**2))
    vl /= np.sqrt(np.sum(vl**2))
    
    xUpperOffset[-1] += vu[0]*xcAftOverhang
    yUpperOffset[-1] += vu[1]*xcAftOverhang
    xLowerOffset[-1] += vl[0]*xcAftOverhang
    yLowerOffset[-1] += vl[1]*xcAftOverhang
    
    # ===
    # closure of the trailing edge block using an arc
    xCentreTe = (xu[-1]+xl[-1])/2. # centred at the midpoint of TE
    yCentreTe = (yu[-1]+yl[-1])/2.
    
    # interpolation point dsNearZone away along average projection direction
    n = np.array([(nu[-1,0] + nl[-1,0]) / 2., (nu[-1,1] + nl[-1,1]) / 2.])
    n /= np.sqrt(np.sum(n**2))
    
    xTeOffset = xCentreTe - n[0]*dsNearZone
    yTeOffset = yCentreTe - n[1]*dsNearZone
    
    # ===
    # create the external domain
    # centre of a circle located at mid-chord of the foil - fwd part of the domain
    x0Circle = ((np.max(xu) + np.max(xl))/2. + (np.min(xu) + np.min(xl))/2.) / 2.
    
    # project the vectors from circle centre to ends of projected offsets onto the domain radius
    nuAft = np.array([xUpperOffset[-1]-x0Circle, yUpperOffset[-1]])
    nuFwd = np.array([xUpperOffset[0]-x0Circle, yUpperOffset[0]])
    nlAft = np.array([xLowerOffset[-1]-x0Circle, yLowerOffset[-1]])
    nlFwd = np.array([xLowerOffset[0]-x0Circle, yLowerOffset[0]])
    
    nuAft /= np.sqrt(np.sum(nuAft**2))
    nuFwd /= np.sqrt(np.sum(nuFwd**2))
    nlAft /= np.sqrt(np.sum(nlAft**2))
    nlFwd /= np.sqrt(np.sum(nlFwd**2))
    
    # modify the shape of domain wedges to spread them apart and reduce high-skew cells
    # close to the foil
    noseAngle = (1.-spreadNoseAngle)*np.arccos(np.dot(nuFwd, nlFwd))
    teAngle = (1.-spreadTeAngle)*np.arccos(np.dot(nuAft, nlAft))
    
    # ===
    # create a set of points used by the block mesh as block vertices and edge interpolation points
    blockPoints = np.array([
        [xUpper[0], yUpper[0]], # foil points
        [xUpper[-1], yUpper[-1]],
        [xLower[0], yLower[0]],
        [xLower[-1], yLower[-1]],
    
        [xUpperOffset[0], yUpperOffset[0]], # offset near-mesh zone points
        [xUpperOffset[-1], yUpperOffset[-1]],
        [xLowerOffset[0], yLowerOffset[0]],
        [xLowerOffset[-1], yLowerOffset[-1]],
    
        rotatePoint([nuFwd[0]*Rdomain+x0Circle, nuFwd[1]*Rdomain], -noseAngle/2/np.pi*180),
        rotatePoint([nlFwd[0]*Rdomain+x0Circle, nlFwd[1]*Rdomain], noseAngle/2/np.pi*180),
        rotatePoint([nuAft[0]*Rdomain+x0Circle, nuAft[1]*Rdomain], -teAngle/2/np.pi*180),
        rotatePoint([nlAft[0]*Rdomain+x0Circle, nlAft[1]*Rdomain], teAngle/2/np.pi*180),
    ])
    
    # ===
    # project the vertices along the span
    blockPoints = blockMeshToolbox.construct3dVertices(blockPoints, span)
    nVert2D = blockPoints.shape[0]/2
    
    blocks = [
        # upper surface
        OFiO.writeBlock(blockMeshToolbox.make3dBlock([1,5,4,0], nVert2D),
            [int(v**meshRefParam) for v in [nCellsWall, nCellsUpper, nCellsSpan]],
            [expWall, expWall, expWall, expWall,
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expLongUpperTe, 1/expLongUpperLe),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expLongUpperOffset, 1/expLongUpperOffset),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expLongUpperOffset, 1/expLongUpperOffset),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expLongUpperTe, 1/expLongUpperLe),
#            "((0.2 0.2 {}) (0.6 0.6 1) (0.2 0.2 {}))".format(expLongUpperTe, 1/expLongUpperLe),
#            "((0.2 0.2 {}) (0.6 0.6 1) (0.2 0.2 {}))".format(expLongUpperOffset, 1/expLongUpperOffset),
#            "((0.2 0.2 {}) (0.6 0.6 1) (0.2 0.2 {}))".format(expLongUpperOffset, 1/expLongUpperOffset),
#            "((0.2 0.2 {}) (0.6 0.6 1) (0.2 0.2 {}))".format(expLongUpperTe, 1/expLongUpperLe),
            1, 1, 1, 1],
            grading="edgeGrading", ret=True),
    
        # nose
        OFiO.writeBlock(blockMeshToolbox.make3dBlock([0,4,6,2], nVert2D),
            [int(v**meshRefParam) for v in [nCellsWall, nCellsNose, nCellsSpan]],
            [expWall, expWall, expWall, expWall,
            1,
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expNoseOffset, 1/expNoseOffset),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expNoseOffset, 1/expNoseOffset),
            1,
            1, 1, 1, 1],
            grading="edgeGrading", ret=True),
    
        # lower surface
        OFiO.writeBlock(blockMeshToolbox.make3dBlock([2,6,7,3], nVert2D),
            [int(v**meshRefParam) for v in [nCellsWall, nCellsLower, nCellsSpan]],
            [expWall, expWall, expWall, expWall,
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expLongLowerLe, 1/expLongLowerTe),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expLongLowerOffset, 1/expLongLowerOffset),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expLongLowerOffset, 1/expLongLowerOffset),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expLongLowerLe, 1/expLongLowerTe),
            1, 1, 1, 1],
            grading="edgeGrading", ret=True),
        
        # trailing edge
        OFiO.writeBlock(blockMeshToolbox.make3dBlock([3,7,5,1], nVert2D),
            [int(v**meshRefParam) for v in [nCellsWall, nCellsTe, nCellsSpan]],
            [expWall, expWall, expWall, expWall,
            1,
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expTeOffset, 1/expTeOffset),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expTeOffset, 1/expTeOffset),
            1,
            1, 1, 1, 1],
            grading="edgeGrading", ret=True),
    
        # upper surface domain
        OFiO.writeBlock(blockMeshToolbox.make3dBlock([5,10,9,4], nVert2D),
            [int(v**meshRefParam) for v in [nCellsOuter, nCellsUpper, nCellsSpan]],
            [expOuter, expOuter, expOuter, expOuter,
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expLongUpperOffset, 1/expLongUpperOffset),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expLongUpperOuter, 1/expLongUpperOuter),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expLongUpperOuter, 1/expLongUpperOuter),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expLongUpperOffset, 1/expLongUpperOffset),
#            "((0.2 0.2 {}) (0.6 0.6 1) (0.2 0.2 {}))".format(expLongUpperOffset, 1/expLongUpperOffset),
#            "((0.2 0.2 {}) (0.6 0.6 1) (0.2 0.2 {}))".format(expLongUpperOuter, 1/expLongUpperOuter),
#            "((0.2 0.2 {}) (0.6 0.6 1) (0.2 0.2 {}))".format(expLongUpperOuter, 1/expLongUpperOuter),
#            "((0.2 0.2 {}) (0.6 0.6 1) (0.2 0.2 {}))".format(expLongUpperOffset, 1/expLongUpperOffset),
            1, 1, 1, 1],
            grading="edgeGrading", ret=True),
        
        # nose domain
        OFiO.writeBlock(blockMeshToolbox.make3dBlock([4,9,8,6], nVert2D),
            [int(v**meshRefParam) for v in [nCellsOuter, nCellsNose, nCellsSpan]],
            [expOuter, expOuter, expOuter, expOuter,
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expNoseOffset, 1/expNoseOffset),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expNoseOuter, 1/expNoseOuter),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expNoseOuter, 1/expNoseOuter),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expNoseOffset, 1/expNoseOffset),
            1, 1, 1, 1],
            grading="edgeGrading", ret=True),
    
    
        # lower surface domain
        OFiO.writeBlock(blockMeshToolbox.make3dBlock([6,8,11,7], nVert2D),
            [int(v**meshRefParam) for v in [nCellsOuter, nCellsLower, nCellsSpan]],
            [expOuter, expOuter, expOuter, expOuter,
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expLongLowerOffset, 1/expLongLowerOffset),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expLongLowerOuter, 1/expLongLowerOuter),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expLongLowerOuter, 1/expLongLowerOuter),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expLongLowerOffset, 1/expLongLowerOffset),
            1, 1, 1, 1],                    
            grading="edgeGrading", ret=True),
    
        # trailing edge domain
        OFiO.writeBlock(blockMeshToolbox.make3dBlock([7,11,10,5], nVert2D),
            [int(v**meshRefParam) for v in [nCellsOuter, nCellsTe, nCellsSpan]],
            [expOuter, expOuter, expOuter, expOuter,
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expTeOffset, 1/expTeOffset),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expTeOuter, 1/expTeOuter),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expTeOuter, 1/expTeOuter),
            "((0.5 0.5 {}) (0.5 0.5 {}))".format(expTeOffset, 1/expTeOffset),
            1, 1, 1, 1],
            grading="edgeGrading", ret=True),
    ]
    
    # ===
    # define edges in 2D
    # discard 0th and -1st points to make interpolation work a bit better
    # TODO does it really help?
    edges = {
        "upperSurface": (np.vstack([xUpper[1:-1], yUpper[1:-1]]), 0, 1),
        "upperOffsetSurface": (np.vstack([xUpperOffset[1:-1], yUpperOffset[1:-1]]), 4, 5),
        "lowerSurface": (np.vstack([xLower[1:-1], yLower[1:-1]]), 2, 3,),
        "lowerOffsetSurface": (np.vstack([xLowerOffset[1:-1], yLowerOffset[1:-1]]), 6, 7),
        "noseSurface": (np.vstack([xNose[1:-1], yNose[1:-1]]), 0, 2),
        "noseOffsetSurface": (np.vstack([xNoseOffset[1:-1], yNoseOffset[1:-1]]), 4, 6),
        "teOffsetArc": (np.array([xTeOffset, yTeOffset]), 7, 5),
        "upperDomainArc": (np.array([0, Rdomain]), 10, 9),
        "fwdDomainArc": (np.array([-Rdomain+x0Circle, 0]), 9, 8),
        "lowerDomainArc": (np.array([0, -Rdomain]), 8, 11),
        "aftDomainArc": (np.array([Rdomain+x0Circle, 0]), 11, 10),
    
    #    "upperSurface": (np.vstack([xUpper, yUpper]), 0, 1),
    #    "upperOffsetSurface": (np.vstack([xUpperOffset, yUpperOffset]), 4, 5),
    #    "lowerSurface": (np.vstack([xLower, yLower]), 2, 3,),
    #    "lowerOffsetSurface": (np.vstack([xLowerOffset, yLowerOffset]), 6, 7),
    #    "noseSurface": (np.vstack([xNose, yNose]), 0, 2),
    #    "noseOffsetSurface": (np.vstack([xNoseOffset, yNoseOffset]), 4, 6),
    #    "teOffsetArc": (np.array([xTeOffset, yTeOffset]), 7, 5),
    #    "upperDomainArc": (np.array([0, Rdomain]), 10, 9),
    #    "fwdDomainArc": (np.array([-Rdomain+x0Circle, 0]), 9, 8),
    #    "lowerDomainArc": (np.array([0, -Rdomain]), 8, 11),
    #    "aftDomainArc": (np.array([Rdomain+x0Circle, 0]), 11, 10),
    
    }
    
    # ===
    # assemble the dictionary
    s = ""
    s += OFiO.writeHeader(True)
    s += OFiO.writeFoamFileLabel(name="blockMeshDict", ret=True)
    
    # TODO could automate the front-back bollocks by defining the edges in 2D
    
    # ---
    # write edge interpolation points
    for e in edges:
        if "Surface" in e:
            s += OFiO.writePointsList("{}Front".format(e),
                 np.vstack((edges[e][0], np.zeros((1,edges[e][0].shape[1])))).T, ret=True)
            s += OFiO.writePointsList("{}Back".format(e),
                 np.vstack((edges[e][0], span+np.zeros((1,edges[e][0].shape[1])))).T, ret=True)
    
    # ---
    # write block vertices
    s += "convertToMeters {:.6e};\n".format(scale)
    s += OFiO.writePointsList("vertices", blockPoints, ret=True)
    
    # ---
    # write the blocks
    s += "blocks\n"
    s += "(\n"
    for block in blocks:
        s += "\t{}\n".format(block)
    s += ");\n"
    
    # ---
    # write the edges
    s += "edges\n"
    s += "(\n"
    for e in edges:
        if "Surface" in e:
            s += "\t"+OFiO.writeEdge("spline", edges[e][1], edges[e][2],
                                     "{}Front".format(e), ret=True)
            s += "\t"+OFiO.writeEdge("spline", edges[e][1]+nVert2D, edges[e][2]+nVert2D,
                                     "{}Back".format(e), ret=True)
        if "Arc" in e:
            s += "\t"+OFiO.writeEdge("arc", edges[e][1], edges[e][2],
                                     np.append(edges[e][0], 0), ret=True)
            s += "\t"+OFiO.writeEdge("arc", edges[e][1]+nVert2D, edges[e][2]+nVert2D,
                                     np.append(edges[e][0], span), ret=True)
    s += ");\n"
    
    # ---
    # write the boundaries
    s += "boundary\n"
    s += "(\n"
    
    s += OFiO.writeBoundary("outer", blockMeshToolbox.makePatches([[10,9], [9,8], [8,11], [10,11]], nVert2D),
                            patchType="patch", ret=True)
    s += OFiO.writeBoundary("foil", blockMeshToolbox.makePatches([[0,1], [1,3], [3,2], [2,0]], nVert2D),
                            patchType="wall", ret=True)
    
    frontPatchFaces = [
        [1,5,4,0],
        [0,4,6,2],
        [2,6,7,3],
        [3,7,5,1],
        [5,10,9,4],
        [4,9,8,6],
        [6,8,11,7],
        [7,11,10,5],
    ]
    backPatchFaces = [[v+nVert2D for v in face] for face in frontPatchFaces]
    
    s += OFiO.writeBoundary("front", frontPatchFaces, patchType="empty", ret=True)
    s += OFiO.writeBoundary("back", backPatchFaces, patchType="empty", ret=True)
    
    s += ");\n"
    
    # ---
    # write the merge pairs
    s += "mergePatchPairs\n"
    s += "(\n"
    s += ");\n"
    
    # ---
    # export to the case
    with open(os.path.join(case, "system/blockMeshDict"), "w") as blockMeshDictFile:
        blockMeshDictFile.write(s)
    
    # ===
    # plot the domain in 2D
    if plotDomain:
        print("Expansion ratio at the wall {}".format(expWall**(1./(nCellsWall-1.))))
        ds = np.min(blockMeshToolbox.blockMeshSpacing(nCellsWall, expWall)) * dsNearZone
        print("Wall spacing {}".format(ds))
        
        plt.plot(xu, yu, "k--", xl, yl, "k--")
        plt.plot(xuOffset, yuOffset, "k:", xlOffset, ylOffset, "k:")
        
        plt.plot(xNose, yNose, "r", xUpper, yUpper, "b", xLower, yLower, "g")
        plt.plot(xNoseOffset, yNoseOffset, "r", xUpperOffset, yUpperOffset, "b", xLowerOffset, yLowerOffset, "g")
        
        for i in range(nVert2D):
            plt.plot(blockPoints[i,0], blockPoints[i,1], "k.")
            plt.text(blockPoints[i,0], blockPoints[i,1], i, size=14)
        
        i = -1
        for e in edges:
            if "Arc" in e:
                i += 1
                plt.plot(edges[e][0][0], edges[e][0][1], "r.")
                plt.text(edges[e][0][0], edges[e][0][1], i, size=14, color="r")
        
        t = np.linspace(0, 2.*np.pi, 101)
        plt.plot(Rdomain*np.cos(t)+x0Circle, Rdomain*np.sin(t), "k:")
        
        plt.show()

# ===
# UNIT TEST
if __name__ == "__main__":
    case = "./standardAirfoilCase"
    # test offsets - NACA66mod with t/c = 0.21 and m/c = 0.065
    xc = np.array([  0.00000000e+00,   2.52972302e-04,   5.11170138e-04,
             7.74813953e-04,   1.04413844e-03,   1.31939381e-03,
             1.60084715e-03,   1.88878404e-03,   2.18351024e-03,
             2.48535367e-03,   2.79466660e-03,   3.11182811e-03,
             3.43724690e-03,   3.77136444e-03,   4.11465860e-03,
             4.46764771e-03,   4.83089531e-03,   5.20501557e-03,
             5.59067949e-03,   5.98862219e-03,   6.39965131e-03,
             6.82465691e-03,   7.26462304e-03,   7.72064151e-03,
             8.19392816e-03,   8.68584244e-03,   9.19791093e-03,
             9.73185593e-03,   1.02896304e-02,   1.08734610e-02,
             1.14859021e-02,   1.21299027e-02,   1.28088932e-02,
             1.35268951e-02,   1.42886660e-02,   1.50998906e-02,
             1.59674385e-02,   1.68997180e-02,   1.79071736e-02,
             1.90030009e-02,   2.02042075e-02,   2.15332380e-02,
             2.30205723e-02,   2.47090867e-02,   2.66618463e-02,
             2.89771825e-02,   3.18211507e-02,   3.55088344e-02,
             4.07625050e-02,   5.00000000e-02,   8.95833333e-02,
             1.29166667e-01,   1.68750000e-01,   2.08333333e-01,
             2.47916667e-01,   2.87500000e-01,   3.27083333e-01,
             3.66666667e-01,   4.06250000e-01,   4.45833333e-01,
             4.85416667e-01,   5.25000000e-01,   5.64583333e-01,
             6.04166667e-01,   6.43750000e-01,   6.83333333e-01,
             7.22916667e-01,   7.62500000e-01,   8.02083333e-01,
             8.41666667e-01,   8.81250000e-01,   9.20833333e-01,
             9.60416667e-01,   1.00000000e+00])
    xu = np.array([  0.00000000e+00,  -1.60517783e-03,  -2.22593706e-03,
            -2.63101176e-03,  -2.92061388e-03,  -3.13405568e-03,
            -3.29152685e-03,  -3.40498354e-03,  -3.48212572e-03,
            -3.52823885e-03,  -3.54706232e-03,  -3.54132506e-03,
            -3.51300981e-03,  -3.46363127e-03,  -3.39427698e-03,
            -3.30572139e-03,  -3.19851830e-03,  -3.07296532e-03,
            -2.92920866e-03,  -2.76720200e-03,  -2.58677028e-03,
            -2.38757560e-03,  -2.16911223e-03,  -1.93075623e-03,
            -1.67168960e-03,  -1.39094197e-03,  -1.08733881e-03,
            -7.59497547e-04,  -4.05776245e-04,  -2.42695565e-05,
             3.87277455e-04,   8.31514217e-04,   1.31156833e-03,
             1.83116437e-03,   2.39473717e-03,   3.00765344e-03,
             3.67641849e-03,   4.40907139e-03,   5.21564852e-03,
             6.10888648e-03,   7.10528029e-03,   8.22667792e-03,
             9.50287629e-03,   1.09759405e-02,   1.27079799e-02,
             1.47962285e-02,   1.74052189e-02,   2.08464061e-02,
             2.58360300e-02,   3.47851134e-02,   7.43650814e-02,
             1.14716561e-01,   1.55291993e-01,   1.96000472e-01,
             2.36856726e-01,   2.77841649e-01,   3.18926216e-01,
             3.60076737e-01,   4.01262696e-01,   4.42465928e-01,
             4.83674867e-01,   5.24879339e-01,   5.66063575e-01,
             6.07207260e-01,   6.48317831e-01,   6.89450361e-01,
             7.30636043e-01,   7.71871494e-01,   8.13111879e-01,
             8.53488486e-01,   8.91614148e-01,   9.28662752e-01,
             9.65290160e-01,   1.00166351e+00])
    yu = np.array([ 0.        ,  0.00319315,  0.004805  ,  0.00608989,  0.00720914,
            0.00822565,  0.00917165,  0.01006638,  0.01092238,  0.01174847,
            0.01255114,  0.01333535,  0.01410502,  0.01486343,  0.01561327,
            0.01635687,  0.01709632,  0.01783344,  0.01856996,  0.01930747,
            0.02004748,  0.02079154,  0.02154105,  0.02229755,  0.02306253,
            0.02383757,  0.02462433,  0.0254246 ,  0.02624027,  0.02707344,
            0.02792643,  0.0288018 ,  0.0297025 ,  0.03063184,  0.03159371,
            0.03259256,  0.03363381,  0.03472386,  0.03587058,  0.03708368,
            0.03837554,  0.0397622 ,  0.041265  ,  0.04291344,  0.04475001,
            0.04683942,  0.04928888,  0.05230153,  0.05633085,  0.06282875,
            0.08543216,  0.1031238 ,  0.11793925,  0.13058609,  0.14129618,
            0.15022948,  0.15746291,  0.16301593,  0.16695981,  0.16930766,
            0.17002028,  0.16898637,  0.16597683,  0.16084636,  0.15377405,
            0.14507521,  0.13483098,  0.1225152 ,  0.1076245 ,  0.09025714,
            0.07118659,  0.05074654,  0.02920515,  0.00679226])
    xl = np.array([ 0.        ,  0.00211112,  0.00324828,  0.00418064,  0.00500889,
            0.00577284,  0.00649322,  0.00718255,  0.00784915,  0.00849895,
            0.0091364 ,  0.00976498,  0.0103875 ,  0.01100636,  0.01162359,
            0.01224102,  0.01286031,  0.013483  ,  0.01411057,  0.01474445,
            0.01538607,  0.01603689,  0.01669836,  0.01737204,  0.01805955,
            0.01876263,  0.01948316,  0.02022321,  0.02098504,  0.02177119,
            0.02258453,  0.02342829,  0.02430622,  0.02522263,  0.02618259,
            0.02719213,  0.02825846,  0.02939036,  0.0305987 ,  0.03189712,
            0.03330313,  0.0348398 ,  0.03653827,  0.03844223,  0.04061571,
            0.04315814,  0.04623708,  0.05017126,  0.05568898,  0.06521489,
            0.10480159,  0.14361677,  0.18220801,  0.22066619,  0.25897661,
            0.29715835,  0.33524045,  0.3732566 ,  0.4112373 ,  0.44920074,
            0.48715847,  0.52512066,  0.56310309,  0.60112607,  0.63918217,
            0.67721631,  0.71519729,  0.75312851,  0.79105479,  0.82984485,
            0.87088585,  0.91300392,  0.95554317,  0.99833649])
    yl = np.array([ 0.        , -0.00298213, -0.00437915, -0.00544533, -0.00634181,
           -0.00713129, -0.00784588, -0.00850458, -0.00911973, -0.00969994,
           -0.01025147, -0.01077903, -0.01128626, -0.01177615, -0.01225106,
           -0.012713  , -0.01316369, -0.01360453, -0.01403682, -0.01446165,
           -0.01488007, -0.01529297, -0.01570118, -0.01610549, -0.01650664,
           -0.01690535, -0.01730229, -0.01769816, -0.01809361, -0.01848938,
           -0.01888613, -0.01928466, -0.01968572, -0.02009018, -0.02049898,
           -0.02091315, -0.02133391, -0.02176261, -0.02220081, -0.02265046,
           -0.02311387, -0.02359392, -0.02409432, -0.02461995, -0.02517753,
           -0.02577665, -0.02643296, -0.02717892, -0.02808615, -0.0293734 ,
           -0.03284726, -0.03518085, -0.03727422, -0.03917386, -0.04068171,
           -0.041652  , -0.04203048, -0.04194298, -0.04157625, -0.04101972,
           -0.04024955, -0.0390939 , -0.03721265, -0.03448677, -0.03122782,
           -0.02794459, -0.02499887, -0.02228178, -0.01980884, -0.01761415,
           -0.01572236, -0.01353848, -0.01062282, -0.00679226])
    
    makeFoilGrid(xc, xu, yu, xl, yl, case,
        dsNearZone = 0.1, # size of the near-field mesh zone
        Rdomain = 25., # radius of the domain
        span = 1., # span
        scale = 1.0, # for uniformly scaling the mesh
        xcNose = 0.05, # x/c from which to start the nose mesh block
        xcAftOverhang = 0.04, # extend offset domain by this much aft of TE for better cell quality
        spreadNoseAngle = 5.0, # multiplier for the angle between nose extrusion vectors
        spreadTeAngle = 2.5,
        #
        nCellsSpan = 1, # spanwise no. cells
        #
        nCellsWall = 100, # wall-normal no. cells in near- and far-field
        nCellsOuter = 75,
        #
        nCellsNose = 35, # no. cells along the foil surface
        nCellsTe = 15,
        nCellsUpper = 150,
        nCellsLower = 150,
        #
        meshRefParam = 1., # for uniformly scaling default mesh size
        #
        # foil-normal directions; >1 to cluster points closer to the foil
        expOuter = 250.,
        expWall = 600.,
        # foil surface
        expLongUpperLe = 3.,
        expLongUpperTe = 3.,
        expLongLowerLe = 4.,
        expLongLowerTe = 4.,
        # at the near-field block edge; >1 to cluster points towards block vertices
        expLongUpperOffset = 0.5,
        expLongLowerOffset = 0.5,
        expNoseOffset = 0.8,
        expTeOffset = 1.2,
        # outer domain; >1 to cluster points towards block vertices
        expLongUpperOuter = 0.2,
        expLongLowerOuter = 0.2,
        expNoseOuter = 2.5,
        expTeOuter = 2.5,
        #
        plotDomain = True
    )