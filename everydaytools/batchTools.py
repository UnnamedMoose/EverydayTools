# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 13:25:06 2019

@author: Artur Lidtke (alidtke@marin.nl)
"""

# because we live in a Python-3 world now
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import subprocess
import sys
#import datetime
#import re
import os
#import signal
import multiprocessing

#=================
# functions

def b2s(byteStr):
	""" Convert bytes to ascii string to avoid issues with implicit unicode conversion
	not recognising certain characters. """
	return byteStr.decode("ascii", "ignore")

def customPrint(text, c="k", endl=True):
	""" Prints some text to stdout in the given colour. Adds a new line at the end of text.
	Parameters
	----------
		@param text - the string to be printed.
		@param colourCode - colours of the printout
		@param addNl - add new line if True (default)
	"""
	
	colours = dict(zip( ["BLACK", "GRAY", "RED", "GREEN", "YELLOW", "BLUE", "MAGENTA", "CYAN", "WHITE",
					"k", "grey", "r", "g", "y", "b", "m", "c", "w"], np.append(range(-1,8), range(-1,8)) ))
	c = colours[c]
	
	if c >= 0:
		output = "\x1b[1;{:d}m{}\x1b[0m".format((30+c), text)
	else:
		output = text
	if endl:
		output += "\n"
	sys.stdout.write(output)

def runBatch(cmds, pwd, outStr = -1, errStr = -1, log=True, showPid=False):
	""" Runs a series of commands in a batch mode.\n
	Parameters
	----------
		@param cmds - list of bash commands passed as a list
		@param pwd - directory in which the commands are to be executed
		@param outStr - (optional) dictionary to hold the output of all calls
		@param errStr - (optional) dictionary to hold the error printouts of all calls
		@param log - (optional, default True) whether or not to log each command output
		@param showPitd (optional, default False) whether to show process PID before each line
	"""
	# constant definitions
	tab = '   '
	
	# ignored in outputs
	supressedWarnings = [
		'bash: no job control in this shell',
		"Inappropriate ioctl for device",
	]
	
	# storage for output logging
	if outStr == -1:
		outStr = {}
	if errStr == -1:
		errStr = {}
	
	# run each command
	for cmd in cmds:
		outStr[cmd], errStr[cmd] = [],[]
		
		if showPid:
			customPrint("PID{:d} ".format(multiprocessing.current_process().pid), "y", False)
		else:
			customPrint(tab, endl=False)
		customPrint("Executing: ", "b", False)
		customPrint(cmd, "k")
		
		# see where to put the log, if at all
		if log:
			logfilePath = os.path.join(pwd, "log."+cmd.replace(" ","_").replace("./","").replace("/","\\"))
		else:
			logfilePath = "/dev/null"
		
		# check if running in Spyder IDE or terminal
		if any("SPYDER" in name for name in os.environ):
			spyderConsole = True
		else:		
			spyderConsole = False
		
		with open(logfilePath, "w") as logfile:
			# if running in Spyder need to load environment variables in ~/.basrc
			# by invoking the interactive -i mode
			if spyderConsole:
				command = ["-i", cmd]
			else:
				command = [cmd]
			
			# open subprocess
			p = subprocess.Popen(command, cwd=pwd,
					shell=True, executable="/bin/bash", # added
					stdout=subprocess.PIPE, stderr=subprocess.PIPE)
			
			# handle outputs
			if log:
				outIter = iter(p.stdout.readline, "")
				for lineOut in outIter:
					logfile.write(lineOut)
					outStr[cmd].append(b2s(lineOut))
			
			errIter = [b2s(e) for e in iter(p.stderr.readline, "")]
			for lineErr in errIter:
				if lineErr.replace("\n", "") not in supressedWarnings:
					if log:
						logfile.write(lineErr)
					errStr[cmd].append(lineErr)
			
			p.wait() # wait for termination
	
		# see if got any unsupressed printout in the error stream and indicate if so, regardless of logging options
		if (len(errStr[cmd]) > 0):
			gotError = False
			errorLines = []
			for e in supressedWarnings:
				errorLines = [(e not in v) for v in errStr[cmd]]
				gotError = np.any(errorLines)
			
			if gotError:
				customPrint(tab + "Errors in: " + cmd, "r")
				for i in range(len(errStr[cmd])):
					if errorLines[i]:
						customPrint(tab*2 + errStr[cmd][i], endl=False)

def runParallel(allrunFunction, nProc, params):
	""" Run the specified function in parallel given allocated no. processors.
	params is a list of permutations of input parameters to the allrun function
	which are to be evaluated; should be specified as a list of tuples (even
	if the function only accepts a single value). Outputs from the allrun function
	will be returned as a sorted list corresponding to each parameter combination.
	
	For clear stdout messages, any runBatch calls in the allrunFunc should use
	the showPid parameter set to True.
	"""
	
	# create a pool of processes
	pool = multiprocessing.Pool(processes=nProc)
	
	# execute
	results = [pool.apply_async(allrunFunction, args=p) for p in params]
	
	# retrieve return values
	output = [p.get() for p in results]
	
	return output

#=================
#  unit tests
if __name__ == "__main__":
	baseCase = "standardAirfoilCase"
	workDir = "/home/alidtke/ReFRESCO/calcs/DTC_part0/grids_inProgress/gridpro/DTC_grid6_newScript/GridPro"
	
	# ====
	# UNIT TEST 0 - serial execution of bash commands
	cmds = [
		"rm -r testCase",
		"cp -r {} testCase".format(baseCase),
	]
	runBatch(cmds, pwd=workDir, log=False)
	
	cmds = [
		"blockMesh",
	]
	runBatch(cmds, pwd=os.path.join(workDir, "testCase"))
	
	# ===
	# UNIT TEST 1 - parallel subprocess management with pool
	def allrunTest(param):
		runBatch(["cp -r {} testCase_{:d}".format(baseCase, param)], pwd=workDir, log=False, showPid=True)
		runBatch(["blockMesh"], pwd=os.path.join(workDir, "testCase_{}".format(param)), showPid=True)
		return param
	
	caseNos = [(p,) for p in range(4)]
	outputs = runParallel(allrunTest, 2, caseNos)
	
