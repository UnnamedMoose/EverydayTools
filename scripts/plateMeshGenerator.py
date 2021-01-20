#!/bin/env/python

import numpy as np
import matplotlib.pyplot as plt
import os

# to set PYTHONPATH from the script level, if needed.
#import sys
#sys.path.append("../Python")

import blockMeshToolbox
import OpenFOAMiO as OFiO

# === !!! ===
# uniform mesh scaling parameter
meshRefParam = 1

# final destination case (needs to contain a proper OpenFOAM folder structure)
case = "../grid0_blockMesh"
# === !!! ===

# ===
def makePlateGrid(case, meshRefParam):
	scale = 1.
	
	length = 1
	span = 1
	
	dsEdge = 0.00125
	dsBl = 0.02
	dsDomain = 0.25
	
	nCellsEdgeLower = 10
	nCellsEdgeUpper = 45
	nCellsChord = 1500
	nCellsBl = 150
	nCellsInOut = 25
	nCellsSpan = 1
	
	#sizeRatioWall = 5e3 # for exact match of baseline grid but high skewness
	sizeRatioWall = 1e3
	sizeRatioChordLeUpper = 4
	sizeRatioChordOuter = 2.5
	sizeRatioChordPlate = 50
	sizeRatioArcLower = 1.8
	sizeRatioArcUpper = 20
	sizeRatioInOut = 25
	
	"""
	# Check wall spacing relative to the baseline grid.
	yWall 2.28828e-7
	eWall 1.06
	v = blockMeshToolbox.blockMeshSpacing(150, 5e3, retVec=True)[2]
	print("Exp ratio at the wall:", (v[2]-v[1]) / (v[1]-v[0]))
	print("Cell size at the wall:", v[1]*dsBl)
	# check cell size on the plate
	yChord 0.0024
	"""
	
	# ===
	# define the vertices and edges
	blockPoints2D = np.array([
		# leading edge
		[0., 0.],
		[dsEdge*0.025, 0.],
		[dsEdge, 0.],
		[dsEdge, dsBl], # not a perfect arc to have orthogonal cells over most of the plate
		[-dsBl, 0.],
		[-dsBl*np.cos(45/180.*np.pi), dsBl*np.sin(45/180.*np.pi)],
		# mid-chord
		[length/2., 0],
		[length/2., dsBl],
		# inlet
		[-dsDomain, 0],
		[-dsDomain, dsBl],
		[-dsDomain, dsDomain],
		# upper wall
		[-dsBl*np.cos(45/180.*np.pi), dsDomain],
		[0., dsDomain],
		[length/2., dsDomain],
	])
	
	edges = {
		"lowerLeArc": (np.array([-dsBl*np.cos(22.5/180.*np.pi), dsBl*np.sin(22.5/180.*np.pi)]), 5, 4),
		"upperLeArc": (np.array([-dsBl*np.cos(67.5/180.*np.pi), dsBl*np.sin(67.5/180.*np.pi)]), 3, 5),
	}
	
	# ===
	# project the vertices along the span
	blockPoints = blockMeshToolbox.construct3dVertices(blockPoints2D, span)
	nVert2D = blockPoints.shape[0]/2
	
	blocks = [
		# leading edge lower
		OFiO.writeBlock(blockMeshToolbox.make3dBlock([0, 1, 5, 4], blockPoints2D.shape[0]),
			[int(v*meshRefParam) for v in [nCellsEdgeLower, nCellsBl]] + [nCellsSpan],
			[1, 1./sizeRatioArcLower, 1./sizeRatioArcLower, 1,
			sizeRatioWall, sizeRatioWall, sizeRatioWall, sizeRatioWall,
			1, 1, 1, 1],
			grading="edgeGrading", ret=True),
		# leading edge upper
		OFiO.writeBlock(blockMeshToolbox.make3dBlock([1, 2, 3, 5], blockPoints2D.shape[0]),
			[int(v*meshRefParam) for v in [nCellsEdgeUpper, nCellsBl]] + [nCellsSpan],
			[sizeRatioChordLeUpper, 1./sizeRatioArcUpper, 1./sizeRatioArcUpper, sizeRatioChordLeUpper,
			sizeRatioWall, sizeRatioWall, sizeRatioWall, sizeRatioWall,
			1, 1, 1, 1],
			grading="edgeGrading", ret=True),
		# inlet lower
		OFiO.writeBlock(blockMeshToolbox.make3dBlock([8, 4, 5, 9], blockPoints2D.shape[0]),
			[int(v*meshRefParam) for v in [nCellsInOut, nCellsEdgeLower]] + [nCellsSpan],
			[1./sizeRatioInOut, 1./sizeRatioInOut, 1./sizeRatioInOut, 1./sizeRatioInOut,
			1, 1./sizeRatioArcLower, 1./sizeRatioArcLower, 1,
			1, 1, 1, 1],
			grading="edgeGrading", ret=True),
		# inlet upper
		OFiO.writeBlock(blockMeshToolbox.make3dBlock([5, 3, 12, 11], blockPoints2D.shape[0]),
			[int(v*meshRefParam) for v in [nCellsEdgeUpper, nCellsInOut]] + [nCellsSpan],
			[1./sizeRatioArcUpper, 1, 1, 1./sizeRatioArcUpper,
			sizeRatioInOut, sizeRatioInOut, sizeRatioInOut, sizeRatioInOut,
			1, 1, 1, 1],
			grading="edgeGrading", ret=True),
		# inlet h-grid
		OFiO.writeBlock(blockMeshToolbox.make3dBlock([9, 5, 11, 10], blockPoints2D.shape[0]),
			[int(v*meshRefParam) for v in [nCellsInOut, nCellsInOut]] + [nCellsSpan],
			[1./sizeRatioInOut, sizeRatioInOut, 1],
			grading="simpleGrading", ret=True),
		# chord upstream
		OFiO.writeBlock(blockMeshToolbox.make3dBlock([2, 6, 7, 3], blockPoints2D.shape[0]),
			[int(v*meshRefParam) for v in [nCellsChord/2., nCellsBl]] + [nCellsSpan],
			[sizeRatioChordPlate, sizeRatioWall, 1], 
			grading="simpleGrading", ret=True),
		# chord upstream upper
		OFiO.writeBlock(blockMeshToolbox.make3dBlock([3, 7, 13, 12], blockPoints2D.shape[0]),
			[int(v*meshRefParam) for v in [nCellsChord/2., nCellsInOut]] + [nCellsSpan],
			[sizeRatioChordPlate, sizeRatioChordOuter, sizeRatioChordOuter, sizeRatioChordPlate,
			sizeRatioInOut, sizeRatioInOut, sizeRatioInOut, sizeRatioInOut,
			1, 1, 1, 1],
			grading="edgeGrading", ret=True),
	]
	
	
	# ===
	# assemble the dictionary
	s = ""
	s += OFiO.writeHeader(True)
	s += OFiO.writeFoamFileLabel(name="blockMeshDict", ret=True)
	
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
			s += "\t"+OFiO.writeEdge("arc", int(edges[e][1]+nVert2D), int(edges[e][2]+nVert2D),
									 np.append(edges[e][0], span), ret=True)
	s += ");\n"
	
	# ---
	# write the boundaries
	s += "boundary\n"
	s += "(\n"
	
	s += OFiO.writeBoundary("inlet", blockMeshToolbox.makePatches([[10,9], [9,8]], nVert2D),
							patchType="patch", ret=True)
	s += OFiO.writeBoundary("plate", blockMeshToolbox.makePatches([[0,1], [1,2], [2,6]], nVert2D),
							patchType="wall", ret=True)
	s += OFiO.writeBoundary("lowerWall", blockMeshToolbox.makePatches([[4,8], [4,0]], nVert2D),
							patchType="patch", ret=True)
	s += OFiO.writeBoundary("upperWall", blockMeshToolbox.makePatches([[10,11], [11,12], [12,13]], nVert2D),
							patchType="patch", ret=True)
	
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
#	plt.figure()
#	for i in range(blockPoints2D.shape[0]):
#		plt.plot(blockPoints2D[i,0], blockPoints2D[i,1], "kp")
#		plt.text(blockPoints2D[i,0], blockPoints2D[i,1], i, size=14)
#	
#	plt.show()


# ===
makePlateGrid(case, meshRefParam)
