# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 20:18:29 2021

@author: ALidtke
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas
import re

def readRefrescoProbes(filename, fileType):
	""" Fast import method for probe-type or acoustics files produced by ReFRESCO.
	
	In theory, the file is still TecPlot-compatible, but takes forever to parse
	with a generic method in situations when many thousands of time steps are saved.
	Returns coordinates of all the probes (assuming they're constant in time)
	and a list of data frames with time traces per probe.
	
	Attributes
	----------
	* ``filename`` (`string`): path to the file to read and parse.
	* ``filetype`` (`string`): either 'probes' for MOPlane and similar or 'acoustics'.
	
	Returns
	----------
	* :class:`pandas.core.frame.DataFrame` with coordinates of each probe (assumed)
		to be fixed in space.
	* `list` of :class:`pandas.core.frame.DataFrame` containing data for each
		probe in the data file (time traces).
	"""
	
	# Figure out where to get the time steps from
	if fileType.lower() == "probes":
		timestepsFrom = "zoneName"
	elif fileType.lower() == "acoustics":
		timestepsFrom = "SOLUTIONTIME"
	else:
		raise ValueError("Incorrect file type specified, expecting 'probes' or 'acoustics'.")
	
	# Read raw text
	with open(filename, "r") as f:
		s = f.read()
	
	# Use regexes to find headers and sizes
	zones = re.findall("ZONE.*", s)
	nProbes = int(re.findall("I=[0-9]+", zones[0])[0].split("=")[1])
	if timestepsFrom == "zoneName":
		timesteps = np.array([
			int(re.findall("T=\".*\",", z)[0].split("\"")[1].split("_")[-1]) for z in zones])
	elif timestepsFrom == "SOLUTIONTIME":
		timesteps = np.array([
			int(re.findall("SOLUTIONTIME=[0-9]+", z)[0].split("=")[1]) for z in zones])
	
	# Labels for fields.
	variables = [v.replace("\"","") for v in re.findall("\"[a-zA-Z\s]+\"", re.findall("VARIABLES=.*", s)[0].split("=")[1])]
	fields = [f for f in variables if f not in ["CoordinateX", "CoordinateY", "CoordinateZ"]]
	
	# Figure out which data to get from which part of the file.
	iCoordsStart = variables.index("CoordinateX")
	fieldIndices = [i for i in range(len(variables)) if variables[i] not in ["CoordinateX", "CoordinateY", "CoordinateZ"]]
	
	# Convert all numerical values to a huge array in one go.
	s = np.array([[float(v) for v in l.strip().split()] for l in s.split("\n")
				  if re.match("-?[0-9]+", l.strip())])
	
	# Extract coordinates, assuming they're constant.
	coords = pandas.DataFrame(data=s[:nProbes, iCoordsStart:iCoordsStart+3], columns=["CoordinateX", "CoordinateY", "CoordinateZ"])
	
	# Chop the data into data frames per probe.
	values = []
	for i in range(nProbes):
		values.append(pandas.DataFrame(data=s[i::nProbes,fieldIndices], columns=fields))
		values[-1]["TimeStep"] = timesteps
	
	return coords, values