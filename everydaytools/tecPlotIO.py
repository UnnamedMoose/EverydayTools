# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 21:49:47 2021

@author: ALidtke
"""

import numpy as np
import pandas
import re
import copy

def _getZoneData(line):
	""" Extract TecPlot formatted zone data from a line, return as two lists of
	keys and matching values.
	
	Attributes
	----------
	* ``line`` (`string`): string representation of a line in a TecPlot file
	
	Returns
	----------
	* `list` with all the names of zone properties (sizes, SOLUTIONTIME, etc.)
	* `list` with corresponding values
	"""
	keys = re.findall("[\,\s]+[a-zA-Z]+=", line)
	vals = copy.deepcopy(line)
	for k in keys:
		vals = vals.replace(k, "SEPARATOR")
	vals = [v.strip("\"").strip("\r") for v in vals.split("SEPARATOR") if v]
	keys = [k.replace(",","").replace("=","").strip() for k in keys]
	return keys, vals

def readTecPlot(filename, getZoneProperties=False):
	""" Read a text-formatted TecPlot data file.
	
	By default return each zone as a pandas DataFrame and ignore zone properties
	Each zone is stored as an object inside a dict, but if only a single zone is
	present in the file, a DataFrame instance only is returned.
	
	Attributes
	----------
	* ``filename`` (`string`): path to the file to read and parse.
	* ``getZoneProperties`` (`bool`, optional): whether or not to return all
		the zone properties or just values themselves.
	
	Returns
	----------
	* `dict` of :class:`pandas.core.frame.DataFrame` containing data of each zone
		or a singe :class:`pandas.core.frame.DataFrame` if no zones are defined
		in the file.
	* `dict` of `dicts` with properties of each zone. Keys match those of data
		object if the file contains multiple zone. Otherwise, the type is just
		a `dict`.
	"""

	# read the file and split into lines, excluding any headers, comments and empty lines
	with open(filename,"r") as instream:
		s = instream.read().split("\n")

	# this will hold the data for each zone
	data = {}
	newZone = "defaultZone" # used when there are no zones defined in the file
	zoneProperties = {} # keywords and values specified in the definition of each zone

	# go over each line
	for l in s:
		if ((len(l.strip()) == 0) or (l[0] == '#')):
		  continue  
	  
		# if found a variables definition
		if ("variables" in l) or ("VARIABLES" in l):
			# safeguard against other information being in the same line as variables
			if "variables" in l:
				l = l.split("variables")[1]
			else:
				l = l.split("VARIABLES")[1]
			variables = [v for v in l.split('"') if (v and (v.strip() != ',') and
				('=' not in v) and ('variables' not in v) and ('VARAIBLES' not in v) and (v != "\r") and (v.strip()))]

		# start a new zone
		elif ("zone" in l) or ("ZONE" in l):
			# find any and all zone properties and its name
			l = l.replace("ZONE","").replace("zone","")

			# get zone properties
			keys, vals = _getZoneData(l)

			# distinguish between zone title and other optional properties
			for i in range(len(keys)):
				if keys[i] in ["T", "t"]:
					newZone = vals[i]
					vals.pop(i)
					keys.pop(i)
					break
			
			# Safeguard against empty zone names. Use index in the file instead.
			if newZone == "":
				newZone = len(data)

			# create a new list for incoming zone values nad store optional properties
			# avoid overwriting zones with the same name
			if newZone in data:
				initialZoneName = newZone
				iRepeat = 0
				while newZone in data:
					newZone = "{}{}".format(initialZoneName, iRepeat)
					iRepeat += 1

			data[newZone] = []
			zoneProperties[newZone] = dict(zip(keys, vals))

		# sometimes files have titles, ignore
		elif ("title" in l) or ("TITLE" in l):
			pass

		# this line must contain data, add it to the most recent zone
		# if there are no zones defined, create a new one with a default name
		elif l and len(l.strip()) > 0:
			try:
				data[newZone].append([float(v) for v in l.split()])

			# if no zones defined in the file
			except KeyError:
				data[newZone] = []
				data[newZone].append([float(v) for v in l.split()])

			# if there is some spurious text, e.g. if someone used fixed line width,
			# try to extract extra zone data, pass otherwise
			except ValueError:
				keys, vals = _getZoneData(l)
				for i in range(len(keys)):
					zoneProperties[newZone][keys[i]] = vals[i]

	# convert to numpy arrays
	for z in data:
		data[z] = np.array(data[z])

	# concatenate variables and data into pandas data frames.
	for z in data:
		if len(data[z]) > 0:
			data[z] = pandas.DataFrame(data[z], columns=variables)

	# if only the default zone is present return as an array
	# NOTE: for Python 3 keys are not a list and do nto support indexing, need to convert
	if (len(data.keys()) == 1):
		data = list(data.values())[0]

	# return
	if getZoneProperties:
		return data, zoneProperties
	else:
		return data

