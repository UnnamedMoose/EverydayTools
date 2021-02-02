# -*- coding: utf-8 -*-
"""
Copyright by Artur K. Lidtke, Univ. of Southampton, UK, 2014

Created on Thu Dec 19 09:27:20 2013

@author: artur

This file contains a set of functions which import OpenFOAM data files and parse them to numpy arrays
for further manipulaiton and plotting
"""

# Python-specific files
import sys
import numpy as np
import imp
import re
import warnings
import os

try:
    imp.find_module('pandas')
    PANDAS_FOUND = True
    from pandas import read_csv
except ImportError:
    PANDAS_FOUND = False
#    print('\n WARNING in OpenFOAMiO: pandas module not found, using slower importing functions ... \n')

try:
    imp.find_module('vtk')
    VTK_FOUND = True
    import vtk
except ImportError:
    VTK_FOUND = False
#    print('\n WARNING in OpenFOAMiO: vtk module not found ... \n')


#TODO: rewrite the probe and plane import functions in such way as to have a few signal objects or similar
# TODO: add the bins option to import forces or write a separate function to handle that

def getNumbersFromString(s):
    """ Splits the passed string at spaces and extracts all numeric values
    Parameters
    ----------
        @param s - string to be split and have its numbers extracted
    Returns
    ----------
        @param vals - list of floats
    """
    vals = []
    for v in s.replace(',',' ').split():
        try:
            vals.append(float(v))
        except ValueError:
            pass
    return vals

#============
# FILE MANIPULATIONS
#============

regexFloats = '\-*\+*e?E?\d+\.?\d*' # all characters forming a float
regexFloat = '\-?\d+\.?\d*e?E?\-?\+?\d*' # single float, any notation

def readFile(filename, split=False):
    """ Read the file and return it as a string
    Parameters
    ----------
        @param filename - path to the file
        @param split - whether to split at newline or not, default False
    Returns
    ----------
        @param s - either a string (if split==False) or a list of strings (if split==True)
            representing the entire file or its subsequent lines, respectively
    """
    with open(filename,'r') as f:
        # concentrate the file as a string
        s = ''
        for line in f:
            s += line
    f.close()
    if split:
        s = s.split("\n")
    return s

def writeFile(filename,s):
    """ Write the file passed as a string, should have \n characters in it\n
    Parameters
    ----------
        @param filename - path to the file"""
    with open(filename,'w') as f:
        for line in s:
            f.write(line)
        f.close()

def modifyList(listName,newVals,filename):
    """ Read-in the given file, find a given OpenFOAM list in it and replace its values\n
    Parameters
    ----------
        @param listName - name of the list to be altered
        @param newVals - a list of new values (scalars or ints)
        @param filename - path to the target file"""
#    print(hasattr(newVals[0],'__getitem__'), newVals[0][0]
#    if hasattr(newVals[0],'__getitem__') and len(newVals[0]) == 3:
#        varIsVect = True
#    else:
#        varIsVect = False
    
    varIsVect = True
    try:
        (v for v in newVals[0][0])
    except IndexError:
        varIsVect = False
    except TypeError:
        varIsVect = False
    
    listRegex = listName+'\s*[\(\s*'+regexFloats+'\)]*;'
    
    s = readFile(filename)
        
    # search for the occurances of the list in question
    match = re.findall(listRegex,s)
    
    # check if exactly 1 match found
    if not(match):
        msg = ('\n\tNo match found, list ' + listName +
            ' in file '+ filename.split('/')[-1] + ' has not been changed!')
        warnings.warn(msg)
        return
    
    elif len(match) > 1:
        msg = ('\n\tMore than one match found for list ' + listName +
            ' in file '+ filename.split('/')[-1] + ', cannot proceed with substitution!')
        warnings.warn(msg)
        return
    
    # construct new list and replace
    newList = listName + '\n(\n'
    for val in newVals:
        if varIsVect:
            newList += '('+ str(val[0]) +' '+ str(val[1]) +' '+ str(val[2]) + ')\n'
        else:
            newList += str(val) + '\n'
    newList += ');'
    
    o = re.sub(listRegex, newList, s)
        
    # write the file
    writeFile(filename,o)

def modifyKeyword(keyword,newVal,valType,filename,regex = -1):
    """ Read-in the given file, find a given OpenFOAM keyword in it and replace its value\n
    Parameters
    ----------
        @param keyword - name of the keyword to be altered
        @param newVal - new value
        @param valType - s for string, f for float, i for int, v for vector, sf - string with optional float arg
        @param filename - path to the target file
        @param regex - override the default regular expression"""
        
    s = readFile(filename)
    
    if valType == 's':
        regex = '[a-zA-Z]+'
        strVal = keyword + ' ' + newVal + ';'
        keyword = keyword.replace('(','\(').replace(')','\)')
    elif (valType == 'i') or (valType == 'f'):
        regex = regexFloat
        strVal = keyword + ' ' + str(newVal) + ';'
    elif (valType == 'sf'):
        regex = '[a-zA-Z]+' + '\s*[' + regexFloat+']*'
        strVal = keyword + ' ' + str(newVal) + ';'
        keyword = keyword.replace('(','\(').replace(')','\)')
    else:
        regex = regexFloats
        strVal = keyword + ' ('
        for v in newVal:
            strVal += ' ' + v
        strVal += ' );'
    
    o = re.sub(keyword+'\s+'+regex+'\s*;', strVal, s)
    writeFile(filename,o)

def getList(listName,filename):
    """ Read-in the given file, find a given OpenFOAM list in it and return it\n
    Parameters
    ----------
        @param listName - name of the list to find
        @param filename - path to the target file"""
    
    s = readFile(filename)

    # see if the list is found for floats/ints
    match = re.findall(listName+'\s*[\(\s*'+regexFloats+'\)]*;',s)
    
    # check if the list is found exactly once
    if len(match) == 0:
        msg = ('\n\tList ' + listName + ' not found in file '
            + filename.split('/')[-1] + ', returning empty!')
        warnings.warn(msg)
        return []
    
    elif len(match) > 1:
        msg = ('\n\tMore than one list with name ' + listName + ' found in file '
                + filename.split('/')[-1] + ', returning empty!')
        warnings.warn(msg)
        return []
    
    match = match[0].replace(listName,'')
    
    varIsVect = False
    if len(re.findall('\(',match)) > 1: varIsVect = True

    vals = [float(v) for v in re.findall(regexFloat,match)]
    if varIsVect:
        return [[vals[i*3],vals[i*3+1],vals[i*3+2]] for i in range(len(vals)/3)]
    else:
        return vals

def getKeyword(keyword,filename):
    """ Read-in the given file, find a given OpenFOAM keyword in it and return it\n
    Parameters
    ----------
        @param keyword - name of the keyword to find
        @param filename - path to the target file"""
    
    s = readFile(filename)

    # see if the keyword is found for floats/ints
    match = re.findall(keyword+'\s+[\-{0,2}\.?e?E?0-9a-zA-Z]+\s*;',s)

    # check if the keyword is found exactly once
    if len(match) == 0:
        msg = ('\n\Keyword ' + keyword + ' not found in file '
            + filename.split('/')[-1] + ', returning -1!')
        warnings.warn(msg)
        return -1
    
    elif len(match) > 1:
        msg = ('\n\tMore than one keyword with name ' + keyword + ' found in file '
                + filename.split('/')[-1] + ', returning -1!')
        warnings.warn(msg)
        return -1
    
    match = match[0].replace(keyword,'').replace(';','')
    
    # value is a vector
    if len(re.findall('\(',match)) == 1:
        return [float(v) for v in re.findall(regexFloat,match)]
    elif utils.isNumber(match):
        # value is int
        if utils.isInt(float(match)):
            return int(float(match))
        # value is float        
        else:
            return float(match)
    else:
        # value is string
        return match
    
def getObjects(filename):
    """ Get the names of runtime objects isted in a dictionary\n
    Parameters
    ----------
        @param listName - path to the control file"""
    
    s = readFile(filename)
    
    # word and { separated with a \n and surrounded by arbitrary whitespaces
    objects = re.findall('(?:[\.0-9a-zA-Z_]+)'+'\s*\{',s)
    return [obj.replace('\n','').replace('{','').replace(' ','') for obj in objects]

#============
# READING
#============

if VTK_FOUND:
    def saveDataToVtk(x,targetPath,baseFileName,time='constant',scalars=[],scalarNames=[],
                      vectors=[],vectorNames=[]):
        """
        Saves the data for a control surface for a given time step; may only provide
        the control surface geometric description to save the shape itself
        """
        
        def mkVtkIdList(it):
            """
            Makes a vtkIdList from a Python iterable. I'm kinda surprised that
            this is necessary, since I assumed that this kind of thing would
            have been built into the wrapper and happen transparently, but it
            seems not.
            """
            vil = vtk.vtkIdList()
            for i in it:
                vil.InsertNextId(int(i))
            return vil
        
        if (len(scalars) != len(scalarNames)) or (len(vectors) != len(vectorNames)):
            raise Exception('Inconsistent number of fields and field names passed!')
        
        surf    = vtk.vtkPolyData()
        points  = vtk.vtkPoints()
        polys   = vtk.vtkCellArray()
        
        if len(scalars) != 0:
            scalarArrays = [vtk.vtkFloatArray() for i in scalars]
            for i in range(len(scalarNames)):
                scalarArrays[i].SetName(scalarNames[i])
    
        if len(vectors) != 0:
            vectorArrays = [vtk.vtkFloatArray() for i in vectors]
            for i in range(len(vectorNames)):
                vectorArrays[i].SetNumberOfComponents(3)
                vectorArrays[i].SetName(vectorNames[i])
        
        pointIndex = 0
        for j in range(len(x)):
            
            pts = []
            
            for k in range(len(x[j])):
                points.InsertPoint(pointIndex, x[j][k])
                pts.append(pointIndex)
                
                pointIndex += 1
            polys.InsertNextCell( mkVtkIdList( tuple(pts) ) )
        
        if len(scalars) != 0:
            for i in range(len(scalarNames)):
                for j in range(len(x)):
                    scalarArrays[i].InsertTuple1(j,scalars[i][j])
        
        if len(vectors) != 0:
            for i in range(len(vectorNames)):
                for j in range(len(x)):
                    vectorArrays[i].InsertTuple3(j,vectors[i][j][0],vectors[i][j][1],vectors[i][j][2])
    
        surf.SetPoints(points)
        surf.SetPolys(polys)
        
        if len(scalars) != 0:
            for i in range(len(scalarNames)):
                surf.GetCellData().AddArray(scalarArrays[i])
        
        if len(vectors) != 0:
            for i in range(len(vectorNames)):
                surf.GetCellData().AddArray(vectorArrays[i])
        
        w = vtk.vtkPolyDataWriter()
        w.SetInput(surf)
        w.SetFileName(targetPath+baseFileName+'_'+str(time)+'.vtk')
        w.Write()

def importAndAssemble(case,runtimeObject,filename,importFunction,
                       importFuncArgs = {}, nJoint = 0, nSkipFirst = 0, nSkipLast = 0,
                       quiet = False, getsAnArray=False):
    # path to the case
    path = os.path.join(case, "postProcessing", runtimeObject)
    
    # get the time directories
    timeDirs = np.array(os.walk(path).next()[1])
    timeDirs = timeDirs[np.argsort([float(v) for v in timeDirs])]
    
    # go over each time directory, import and assemble
    for j in range(0,len(timeDirs)):
        tmppath = path;
    
        if float(timeDirs[j])%1. == 0.:
            tmppath += str(int(timeDirs[j]))
        else:
            tmppath += str(float(timeDirs[j]))
    
        tmppath += '/' + filename
        
        # call the passed import function
#        xyz,time_part,data_part,deltaT_part = importFunction(tmppath)
        # need to first assign all the return values to a single variable to unpack
        retVals = importFunction(tmppath, **importFuncArgs)
        if getsAnArray:
            time_part = retVals[:,0]
            data_part = retVals[:,1:]
        else:
            time_part = retVals[0]
            data_part = retVals[1]
		
        otherInfo = False
        if not getsAnArray:
        	try: otherInfo = retVals[2:]
        	except IndexError:
		        pass
        
        if j == 0:
            time = time_part;
            data = data_part;
#            deltaT = deltaT_part # this is the same for all time folders but shifted by the initialisation time for each run

        else:
            # if there are some leftover folders, which shouldn't happen - just do the longest possible time trace
            if np.max(time_part) < np.max(time):
                print('\nWARNING: Detected incorrectly ordered time values: data for the file to be added is')
                print('         already contained in the previous time range. Subsequent time folders will be ignored...\n')
                break
            
            repeatedIndex = 0
            for k in range(0,time_part.shape[0]):
                if time_part[k] > time[-1]:
                    repeatedIndex = k;
                    # may want to skip a few more for smoothness sometimes
                    repeatedIndex += nJoint
                    break;
    
            time = np.append(time,time_part[repeatedIndex:],0);
            
            if type(data) == list:
                for i in range(len(data)):
                    data[i] = np.append(data[i],data_part[i][repeatedIndex:,:],0)
            
            elif type(data)  == np.ndarray:
                data = np.append(data,data_part[repeatedIndex:,:],0)
            
            else:
                msg = 'Unsupported return data type of the import function: {}!'.format(type(data))
                raise IOError(msg)
        
        if not quiet:
            print("Added values for ",tmppath,"...")

    # get rid of the unwanted first few records
    time = time[nSkipFirst:time.shape[0]-nSkipLast]
    if type(data) == list:
        for i in range(len(data)):
            data[i] = data[i][nSkipFirst:data[i].shape[0]-nSkipLast,:]
    
    elif type(data)  == np.ndarray:
        data = data[nSkipFirst:data.shape[0]-nSkipLast,:]
    
    if otherInfo:
        return time,data,otherInfo
    else:
        return time,data

def getTimeDirs(path, retTimeNames=False):
    """ returns paths to all the time folders available in path """
    # get the time directories
#    timeDirs = os.walk(path).next()[1]
#    timeDirs = np.sort(timeDirs)
    paths = []
    
    # go over each time directory, assemble possible paths
    dirs = os.walk(path).next()[1]
    timeDirs = []
    for d in dirs:
        try:
            timeDirs.append(int(d))
        except ValueError:
            pass
    timeDirs = np.sort(timeDirs)
    
    for j in range(0,len(timeDirs)):
        tmppath = path;
        if float(timeDirs[j])%1. == 0.:
            tmppath = os.path.join(tmppath, str(int(timeDirs[j])))
        else:
            tmppath = os.path.join(tmppath, str(float(timeDirs[j])))
        tmppath += '/'
        paths.append(tmppath)
    
    if retTimeNames:
        return timeDirs
    else:
        return paths

def iomportAndAssembleNoise(case,runtimeObject,filename,importFunction,
                       importFuncArgs = {}, nJoint = 0, nSkipFirst = 0, nSkipLast = 0, getsAnArray=False):
    """ equivalent to importAndAssemble but takes care of some perculiarities in the
    FW-H output files """
    
    # path to the case
    path = os.path.join(case, 'postProcessing', runtimeObject)
    
    # get the time directories
    timeDirs = os.walk(path).next()[1];
    timeDirs = np.sort(timeDirs);
    
    # go over each time directory, import and assemble
    for j in range(0,len(timeDirs)):
        tmppath = path;
    
        if float(timeDirs[j])%1. == 0.:
            tmppath += str(int(timeDirs[j]))
        else:
            tmppath += str(float(timeDirs[j]))
    
        tmppath += '/' + filename
        
        # call the passed import function
#        xyz,time_part,data_part,deltaT_part = importFunction(tmppath)
        # need to first assign all the return values to a single variable to unpack
        retVals = importFunction(tmppath, **importFuncArgs)
        if getsAnArray:
            time_part = retVals[:,0]
            data_part = retVals[:,1:]
        else:
            time_part = retVals[0]
            data_part = retVals[1]
		
        otherInfo = False
        if not getsAnArray:
        	try: otherInfo = retVals[2:]
        	except IndexError:
		        pass
        
        # keep individual time series for each receiver
        time_part_all = []
        
        # discard the part of the signal where there is no useful data for each receiver
        deltaT = otherInfo[1]
        for iRec in range(len(data_part)):
            # between the start and the min ret time value
            timeToDiscardTo = float(timeDirs[j]) + deltaT[iRec]
            
            # some time after the start of the next sub-run
            timeToDiscardAfter = time_part[-1]
            if j < len(timeDirs)-1:
                timeToDiscardAfter = float(timeDirs[j+1]) + deltaT[iRec] + (time_part[1]-time_part[0])*nJoint
            
            keep = (time_part>=timeToDiscardTo) * (time_part<=timeToDiscardAfter)
            
            time_part_all.append(time_part[keep])
            data_part[iRec] = data_part[iRec][keep,:]
        
        if j == 0:
            time = time_part_all
            data = data_part
#            deltaT = deltaT_part # this is the same for all time folders but shifted by the initialisation time for each run

        else:
            for iRec in range(len(data)):
                
                repeatedIndex = 0
                for k in range(0,time_part_all[iRec].shape[0]):
                    if time_part_all[iRec][k] > time[iRec][-1]:
                        repeatedIndex = k;
                        break;
    
                time[iRec] = np.append(time[iRec],time_part_all[iRec][repeatedIndex:],0);
                data[iRec] = np.append(data[iRec],data_part[iRec][repeatedIndex:,:],0)
        
        print("Added values for ",tmppath,"...")

    # get rid of the unwanted first few records
    for iRec in range(len(data)):
        time[iRec] = time[iRec][nSkipFirst:time[iRec].shape[0]-nSkipLast]
        data[iRec] = data[iRec][nSkipFirst:data[iRec].shape[0]-nSkipLast,:]
    
    if otherInfo:
        return time,data,otherInfo
    else:
        return time,data

def importSurfaceFieldData(filename):
    """
    Read surface data for all surface fields saved - may be noise or flow field data, whatever
    """
    
    # associates field type identifier with the number of elements
    fieldTypeIdentifiers = {'S':1,'V':3,'F':5}
    
    vertices = []
    normals = []
    areas = []
    
    time = []
    
    fieldNames = []
    fieldTypes = []
    fields = {}
        
    nFaces = 0
    nVertices = 0
    
    with open(filename) as f:
        data = f.read().split('\n')
        data = [x for x in data if x]
        
    for line in data:
        
        # read the header lines
        if line[0] == '#':
            line = line.replace('#','').replace('(','').replace(')','').split()
            
            # total no. faces
            if (line[0] == 'No.') and (line[1] == 'faces:'):
                nFaces = int(line[2])
                
            # vertices of an individual face
            elif line[0] == 'Vertices':
                face = [0 for i in range(int(line[1]))]
                for i in range(len(face)):
                    face[i] = (float(line[i*3+2]), float(line[i*3+3]), float(line[i*3+4]))
                vertices.append(face)
                nVertices += len(face)
                
                normals.append( (float(line[len(face)*3+7]), float(line[len(face)*3+8]), float(line[len(face)*3+9])) )
                areas.append( float(line[len(face)*3+11]) )
            
            elif line[0] == 'RadiationVectors':
                recI = line[3].replace(':','')
                fields['r'+recI] = np.zeros((1,nFaces,3))
                
                line = [float(v) for v in line[4:]]
                
                for faceI in range(nFaces):
                    for dimI in range(3):
                        fields['r'+recI][0,faceI,dimI] = line[3*faceI+dimI]
            
            # field names and types - allocate the storage containers
            elif (line[0] == 'Field') and (line[1] == 'names:'):
                fieldNames = line[2:]
        
            elif (line[0] == 'Field') and (line[1] == 'types:'):
                fieldTypes = line[2:]
                for fieldI in range(len(fieldNames)):
                    # for scalars have 0- as the 3rd dimension
                    if fieldTypes[fieldI] == 'S':
                        fields[fieldNames[fieldI]] = np.zeros((0,nFaces))
                    else:
                        fields[fieldNames[fieldI]] = np.zeros((0,nFaces,fieldTypeIdentifiers[fieldTypes[fieldI]]))
        
        # read the time data
        else:
            line = [float(elem) for elem in line.replace('(','').replace(')','').split()]
            
            time.append(line[0])
            
            # current index in the line list
            lineIndex = 1
            
            for fieldI in range(len(fieldNames)):
                # for a scalar field only take the split line segement and convert to np.array
                if fieldTypes[fieldI] == 'S':
                    fields[fieldNames[fieldI]] = np.append(fields[fieldNames[fieldI]],
                        np.array(line[lineIndex:lineIndex+nFaces])[..., np.newaxis].T, axis=0)
                    lineIndex += nFaces
                    
                # for vectors need to isolate the 3rd array dimension from the vectorised format
                else:
                    nDim = fieldTypeIdentifiers[fieldTypes[fieldI]]
                    tmpField = np.zeros((1,nFaces,nDim))
                    for faceI in range(nFaces):
                        for dimI in range(nDim):
                            tmpField[0,faceI,dimI] = line[lineIndex+nDim*faceI+dimI]
                    
                    fields[fieldNames[fieldI]] = np.append(fields[fieldNames[fieldI]], tmpField, axis=0)
                    lineIndex += nFaces*nDim
                
    return vertices,areas,normals,fieldNames,time,fields

def importOpenFoamResidualData(filename,fields):
    """ Imports residuals time history from an OpenFOAM log file for a set of field names
    Parameters
    ----------
        @param filename - path to the log file
        @param fields - list of field names as string objects
    Returns
    ----------
        @param time - time/iteration values for each complete time step found in the log
        @param fieldInitialInitialRes - dict with entries for each field, each entry is a
            list with len(time) values; these correspond to the first initial residual
            in each time step (measure of convergence for steady-state solvers)
        @param fieldFinalInitialRes - same as InitialInitial but with the last found initial
            residual (measure of convergence for unsteady solvers where time step convergence is needed)
        @param fieldNoIter - same as FinalInitial and InitialInitial but each time step entry is a list
            with length equal to the number of loops per a given variable; elements are the number of
            linear solver iterations called during each loop
        @param fieldInitialRes - same as NoIter but holds initial residuals instead of solver iters
        @param fieldFinalRes - same as NoIter but holds final residuals instead of solver iters
        @param initContErr - initial continuity error in each time step
        @param contErr - for each time step contains the values of continuity errors computed after
            each set of pressure iterations per time step
    """
    s = readFile(filename,split=True)
    
    time = []
    fieldInitialRes = dict(zip(fields,[[] for i in fields]))
    fieldInitialInitialRes = dict(zip(fields,[[] for i in fields])) # has only the first of all residuals per time step
    fieldFinalRes = dict(zip(fields,[[] for i in fields]))
    fieldFinalInitialRes = dict(zip(fields,[[] for i in fields]))  # only has the last of all residuals per time step
    fieldNoIter = dict(zip(fields,[[] for i in fields])) 
    initContErr = []
    contErr = []
    
    timeI = -1
    skipToNextTime = False
    gotFullTimeStep = False
    
    for line in s:
        if line and (line != '\n'):
            if (line.split()[0] == 'Time') and (line.split()[1] == '='):
                try:
                    time.append(float(line.split()[2]))
                    timeI += 1
                    
                    # temporary containers for this time step
                    initResTmp = dict(zip(fields,[[] for i in range(len(fields))]))
                    finalResTmp = dict(zip(fields,[[] for i in range(len(fields))]))
                    noIterTmp = dict(zip(fields,[[] for i in range(len(fields))]))
                    contErrTmp = []
                    
                    skipToNextTime = False
                except ValueError:
                    skipToNextTime = True
                    
                # check if entire record available for this time step
                gotFullTimeStep = False
            
            elif line.split()[0] == 'ExecutionTime':
                # move the temporary containers to the final storage
                if not skipToNextTime:
                    try:
                        for i in range(len(fields)):
                            fieldInitialRes[fields[i]].append(initResTmp[fields[i]])
                            fieldInitialInitialRes[fields[i]].append(initResTmp[fields[i]][0])
                            fieldFinalRes[fields[i]].append(finalResTmp[fields[i]])
                            fieldFinalInitialRes[fields[i]].append(initResTmp[fields[i]][-1])
                            fieldNoIter[fields[i]].append(noIterTmp[fields[i]])
                        contErr.append(contErrTmp)
                        initContErr.append(contErrTmp[0])
                        gotFullTimeStep = True
                    except:
                        pass
                
            else:
                if not skipToNextTime:
                    for i in range(len(fields)):
                        # see if this line has printout for this field
#                        try:
#                            if line.split(":")[1].startswith
                        index = line.find('Solving for '+fields[i])
                        if index > -1:
                            # if yes then go over each element in the line and see if it's a number, store if yes
                            vals = getNumbersFromString(line.replace(',',' '))
                            # in OF format the numbers will always be: initial res, final res and no. solver iterations
                            if len(vals) != 3:
                                skipToNextTime = True
                                break
                            initResTmp[fields[i]].append(vals[0])
                            finalResTmp[fields[i]].append(vals[1])
                            noIterTmp[fields[i]].append(vals[2])
                            # move on to the next line as this one has been handled
                            break
#                        except IndexError:
#                            pass # line to be ignored
                
                    # see if this line has continuity error information
                    index = line.find('time step continuity errors :')
                    if index > -1 and timeI > -1: # for unsteady solvers will get cont errs before time loop, ignore
                        vals = getNumbersFromString(line.replace(',',' '))
                        # take the sum local value - more representative of problems in critical parts of the domain
                        contErrTmp.append(vals[0])
        
    # truncate to the last complete time record
    if not gotFullTimeStep:
        minLen = len(time)
        for i in range(len(fields)):
            if len(time) > len(fieldInitialRes[fields[i]]):
                minLen = len(fieldInitialRes[fields[i]])
        
        time = time[:minLen]
        for i in range(len(fields)):
            fieldInitialRes[fields[i]] = fieldInitialRes[fields[i]][:minLen]
            fieldInitialInitialRes[fields[i]] = fieldInitialInitialRes[fields[i]][:minLen]
            fieldFinalRes[fields[i]] = fieldFinalRes[fields[i]][:minLen]
            fieldFinalInitialRes[fields[i]] = fieldFinalInitialRes[fields[i]][:minLen]
            fieldNoIter[fields[i]] = fieldNoIter[fields[i]][:minLen]
                
    return time,fieldInitialInitialRes,fieldFinalInitialRes,fieldNoIter,fieldInitialRes,fieldFinalRes,initContErr,contErr

if __name__ == "__main__":
    filename = '/home/artur/SharedFolders/iridis4_scratch/AUVglider/stage5_gridTest_ver15_refLvl2_kkl_U_0.31_AoA_-10_TI_0.5/log.run'
    fields = ['p','Ux','Uy','Uz','kl','kt','omega']
    time,fieldInitialInitialRes,fieldFinalInitialRes,fieldNoIter,fieldInitialRes,fieldFinalRes,initContErr,contErr \
        = importOpenFoamResidualData(filename, fields)
    print(len(time), len(fieldFinalInitialRes))

def importTabulatedData(filename,headerLines = 0,separator=' ',usePandas=True):
    """
    Load a set of non-descript tabulated data and return as a single array
    """
    
    if PANDAS_FOUND and usePandas:
        data = np.array(read_csv(filename,sep=separator,skiprows = headerLines, header=None))
        return data
        
    else:
        f = open(filename,'r')
        lengthKnown = 0
        linesSkipped = 0
        
        while True:
            line = f.readline();

            if line == '':
                break;

            linesSkipped += 1
            if linesSkipped > headerLines:

                if (line != '\n'):

                    if not(lengthKnown):
                        data = np.zeros((0,len(line.split())))
                        lengthKnown=1
    
                    data = np.append(data,np.array([[float(i) for i in line.split()]]),0)
        return data

def importOpenFoamClCdData(filename, returnIndividualLists=False):
    """
    Read in the lift, drag and moment coefficients in the OpenFOAM format
    """
    utils.convertEOL(filename);
    f = open(filename,'r')
    noDataPoints = 0;
    Cm = [];
    Cd = [];
    Cl = [];
    time = [];

    while True:
        line = f.readline();    # read line by line
    
        if line == '':    # check for EoF
            break;
        
        if PANDAS_FOUND:
            # count the header lines
            if (line != '\n') and (line.split()[0] == '#'):
                noDataPoints += 1
            else:
                # import the data
                data = np.array(read_csv(filename,sep='\t', skiprows = noDataPoints,header=None))
                break
            
        else:
        	if (line != '\n') and (line.split()[0] != '#'):    # skip empty lines
        		noDataPoints += 1;
        
        		time.append(float(line.split()[0]));
        		Cm.append(float(line.split()[1]));
        		Cd.append(float(line.split()[2]));
        		Cl.append(float(line.split()[3]));
    
    if returnIndividualLists:
        if PANDAS_FOUND:
            return data[:,0],data[:,1],data[:,2],data[:,3]
        else:
            return np.array(time),np.array(Cm),np.array(Cd),np.array(Cl)
    else:
        if PANDAS_FOUND:
            return data[:,0],np.vstack((data[:,1],data[:,2],data[:,3])).T
        else:
            return np.array(time),np.vstack((np.array(Cm),np.array(Cd),np.array(Cl))).T

def importOpenFoamScalarTimeData(filename):
    """
    Read in the value of a single scalar being dumped over time in the OpenFOAM format
    """
    utils.convertEOL(filename);
    f = open(filename,'r')
    noDataPoints = 0;
    val = [];
    time = [];
    PANDAS_FOUND = False;
    while True:
        line = f.readline();    # read line by line
        
        if line == '':    # check for EoF
            break;
    
        if PANDAS_FOUND:
        # count the header lines
            if (line != '\n') and (line.split()[0] == '#'):
                noDataPoints += 1
            else:
                # import the data
                data = np.array(read_csv(filename,sep='\t', skiprows = noDataPoints,header=None))
                break
    
        else:
            if (line != '\n') and (line.split()[0] != '#'):    # skip empty lines
                noDataPoints += 1;
                time.append(float(line.split()[0]));
                val.append([float(s) for s in line.split()[1:]]);
    
    if PANDAS_FOUND:
        return data[:,0],data[:,1:]
    else:
        return np.array(time),np.array(val)
    
def importOpenFoamForceData(filename):
    """
    Imports the force data, given the path to the file
    Distinguishes between 2.2.2+ and older versions (whether porous force is
    included or not - does not return it)
    """
    utils.convertEOL(filename);
    f = open(filename,'r')
    forces = [];
    moments = [];
    time = [];
    nHeaderLines = 0

    while True:
        line = f.readline();    # read line by line

        if line == '':    # check for EoF
            break;
        
        if PANDAS_FOUND:
            # count the header lines
            if (line != '\n') and (line.split()[0] == '#'):
                nHeaderLines += 1
            else:
                # import the data
                data = np.array(read_csv(filename,sep=';', skiprows = nHeaderLines, comment="#", header=None))

                # this array will store the actual values after the float conversion
                allData = np.zeros((data.shape[0],np.array([float(j) for j in data[0,0].replace('(',' ').replace(')',' ').replace(',',' ').split()]).shape[0]))

                # split each row and assign
                for i in range(data.shape[0]):
                    allData[i,:] = np.array([float(j) for j in data[i,0].replace('(',' ').replace(')',' ').replace(',',' ').split()])
                break
            
        else:
            if (line != '\n') and (line.split()[0] != '#'):    # skip empty lines
                line = line.replace('(',' ')
                line = line.replace(')',' ')
                line = line.replace(',',' ')
    
                time.append(float(line.split()[0]));
                forces.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3]),float(line.split()[4]),float(line.split()[5]),float(line.split()[6])]);
                
                # check line.split length and deduce whether porous force is included or not
                if len(line.split()) == 19: # 2.2.2+
                    moments.append([float(line.split()[10]),float(line.split()[11]),float(line.split()[12]),float(line.split()[13]),float(line.split()[14]),float(line.split()[15])]);
                elif len(line.split()) == 13: # older (no porous force component)
                    moments.append([float(line.split()[7]),float(line.split()[8]),float(line.split()[9]),float(line.split()[10]),float(line.split()[11]),float(line.split()[12])]);
                else:
                    print('Tried opening forces file with',len(line.split()),'entries. Please verify yoyr file...')
                    sys.exit()
                    
    if PANDAS_FOUND:
        time = allData[:,0]
        forces = np.transpose(np.vstack([allData[:,1],allData[:,2],allData[:,3],allData[:,4],allData[:,5],allData[:,6]]))
        if allData.shape[1] == 19:
            moments = np.transpose(np.vstack([allData[:,10],allData[:,11],allData[:,12],allData[:,13],allData[:,14],allData[:,15]]))
        elif allData.shape[1] == 13:
            moments = np.transpose(np.vstack([allData[:,7],allData[:,8],allData[:,9],allData[:,10],allData[:,11],allData[:,12]]))
        else:
            print('Tried opening forces file with',len(line.split()),'entries. Please verify yoyr file...')
            sys.exit()
        return time, np.hstack([np.array(forces),np.array(moments)])
    else:
        return np.array(time), np.hstack([np.array(forces),np.array(moments)])

def importOpenFoamPlaneData(filename):
    """
    Import the raw cut plane data
    """
    
    f = open(filename,'r')

    xyz = np.zeros((0,3))
    
    while True:
        line = f.readline();    # read line by line

        if line == '':    # check for EoF
            break;

        if (line != '\n'):    # skip empty lines
            if PANDAS_FOUND:
                # read in all the stuff from the header; terminate when done
                if line.split()[2] == 'POINT_DATA' or line.split()[2] == 'FACE_DATA':
                    fieldName = line.split()[1]
                    break
            else:
                if line.split()[2] == 'POINT_DATA':
                    fieldName = line.split()[1]
                    
                elif line.split()[1] == 'x':
                    field = np.zeros((0,len(line.split())-4)) # different for vector and scalar quantities
                    
                elif line.split()[0] != '#': # skip other comments if present
    
                    xyz = np.vstack([xyz,np.array([float(i) for i in line.split()[0:3]])])
    
                    localFieldVal = field.shape[1]*[0.]
                    for j in range(0,field.shape[1]):
                        localFieldVal[j] = float(line.split()[3+j])
    
                    field = np.vstack([field,np.array([localFieldVal])])

    if PANDAS_FOUND:
        # now import the field data using read_csv - much much faster
        data = np.array(read_csv(filename,sep=' ', skiprows = 2,header=None))
    
        return fieldName, data[:,0:3],data[:,3:data.shape[1]]
    else:
        return fieldName, xyz, field

def importOpenFoamProbeData(filename, fieldType='scalar', statusInterval=-1, usePandas = True, returnAsListOfArrs=False):
    """
    Reads in the field in OpenFOAM format and returns three objects: probe locations in xyz space
    as an Nby3 array, time values as nTby1 array and values of the field as a list of length N filled with either nTby1 or nTby3 arrays.
    All the data is arranged in a chronological manner.
    For large data sets this operation may take a considerable amount of time so may specify
    an integer denoting the interval of time values for which status will be printed on the scren.
    
    EDIT: changed the return type to a 3D array: (nTimes x nProbes x nTerms(0 or 3));
    use the flag to stick to the old return type
    """
    
#    f = open(filename,'r')
    
    if statusInterval > 0:
        noLines = 0;

    lines = readFile(filename).split('\n')
    iLine = 0
    
    while True:
#        line = f.readline();    # read line by line
        line = lines[iLine]
        iLine += 1

        if line == '':    # check for EoF
            break;
  
        if (line != '\n'):    # skip empty lines
        
            if line.split()[1] == 'x':
                # get the number of probes and allocate memory
                noProbes = len(line.split())-2;
                time = np.zeros((0))
                if (fieldType == 'scalar') or (fieldType == 's'):
                    field = [np.zeros((0)) for i in range(noProbes)]#[np.zeros((0))]*noProbes
                elif (fieldType == 'vector') or (fieldType == 'v'):
                    field = [np.zeros((0,3)) for i in range(noProbes)]#[np.zeros((0,3))]*noProbes
                    
                # process the probe x-locations
                x = np.array([float(i) for i in line.split()[2:noProbes+2]])

            elif line.split()[1] == 'y':
                y = np.array([float(i) for i in line.split()[2:noProbes+2]])
                    
            elif line.split()[1] == 'z':
                z = np.array([float(i) for i in line.split()[2:noProbes+2]])

            elif line.split()[0] != '#':
                if PANDAS_FOUND and usePandas:
                    if (fieldType == 'scalar') or (fieldType == 's'):
                        data = np.array(read_csv(filename,sep='\t', skiprows = 4,header=None))
                        time = data[:,0]
                        for i in range(noProbes):
                            field[i] = data[:,i+1]
                            
                    elif (fieldType =='vector') or (fieldType =='v'):
                        data = np.array(read_csv(filename,sep=';', skiprows = 4,header=None))
                        
                        # this array will store the actual values after the float conversion
                        allData = np.zeros((data.shape[0],np.array([float(j) for j in data[0,0].replace('(',' ').replace(')',' ').replace(',',' ').split()]).shape[0]))

                        # split each row and assign
                        for i in range(data.shape[0]):
                            allData[i,:] = np.array([float(j) for j in data[i,0].replace('(',' ').replace(')',' ').replace(',',' ').split()])
                        
                        # split column-wise to divide into probe signals
                        time = allData[:,0]
                        for i in range(noProbes):
                            field[i] = allData[:,i*3+1:i*3+4]
                    break
                else:
                    time = np.hstack([time,float(line.split()[0])])
                    
                    if statusInterval > 0:
                        noLines += 1;
                        if noLines >= statusInterval:
                            print('Processing values for time',time[-1],'...')
                            noLines = 0
                    
                    for i in range(0,noProbes):
                        
                        if (fieldType == 'scalar') or (fieldType == 's'):
#                            vals = [float(s) for s in (line.replace('(',' ').replace(')',' ').replace(',',' ')).split()]
#                
#                            time = np.append(time, vals[0])
#                            for i in range(0,noProbes):
#                                field[i] = np.append(field[i],vals[i+1])
                            field[i] = np.hstack([field[i],float(line.split()[i+1])]);
                        
                        elif (fieldType =='vector') or (fieldType =='v'):
#                            vals = [float(s) for s in (line.replace('(',' ').replace(')',' ').replace(',',' ')).split()]
#                
#                            time = np.append(time, vals[0])
#                            for i in range(0,noProbes):
#                                field[i] = np.vstack([field[i],np.array([vals[(i*3+1) : (i*3+1+3)]])])
                    
                            localFieldVal = 3*[0.]
                            for j in range(0,3):
                                localFieldVal[j] = float(line.split()[i*3+j+1].replace('(',' ').replace(')',' '))
    
                            field[i] = np.vstack([field[i],np.array([localFieldVal])])

    if returnAsListOfArrs:
        return time, field, np.transpose(np.vstack([x,y,z]))
    else:
        if fieldType == 'scalar':
            fRet = np.zeros((time.shape[0],x.shape[0]))
        else:
            fRet = np.zeros((time.shape[0],x.shape[0],3))
        
        for k in range(time.shape[0]):
            for i in range(x.shape[0]):
                fRet[k,i] = field[i][k]
        return time, fRet, np.transpose(np.vstack([x,y,z]))
    
def importPointListFromDict(filename,listName):
    """
    Import locations of points from a dictionary file and return them as an array.
    """
    f = open(filename,'r')
    points = [];
    while True:
        line = f.readline();    # read line by line
        if line == '':    # check for EoF
            break;
            
        if line != '\n':    # skip empty lines
        
            if line.split()[0] == listName:
                while(True):
                    line = f.readline();
                    if line.split()[0] == ');':
                        break;
                    elif line != '\n' and len(line.split()) > 1:
                        points.append([float(line.split()[1]),float(line.split()[2]),float(line.split()[3])])
    
    points = np.array(points);
    return points;

def importOpenFoamSurfaceSamplingData(filename):
    """
    Import the raw formatted data obtained from using the surface probe utility (essentially, a plain text file with data)
    """
    
    if PANDAS_FOUND:
        data = np.array(read_csv(filename,sep=' ',header=None))
        xyz = data[:,0:3]
        if data.shape[1] == 4:
            field = np.transpose(np.array([data[:,3]]))
        else:
            field = data[:,3:6]
        return xyz, field
        
    else:
        f = open(filename,'r')
    
        xyz = np.zeros((0,3))
        fieldType = -1
        
        while True:
            line = f.readline();    # read line by line
    
            if line == '':    # check for EoF
                break;
    
            if (line != '\n'):    # skip empty lines

                xyz = np.vstack([xyz,np.array([float(i) for i in line.split()[0:3]])])
                
                if fieldType == -1:
                    if len(line.split()) == 4:
                        fieldType = 's'
                        field = np.zeros((0,1))
                    elif len(line.split()) == 6:
                        fieldType = 'v'
                        field = np.zeros((0,3))
                
                if fieldType == 's':
                    field = np.append(field,np.array([[float(line.split()[3])]]),0)
                elif fieldType == 'v':
                    field = np.append(field,np.array([[float(i) for i in line.split()[3:6]]]),0)

        return xyz, field

def importNoiseTerms(filename):
    """ Imports noise data from an FWH file; the returned data is a list of length
    nProbes filled with (nTime,3) arrays """
    
    f = open(filename,'r')
    deltaT = []
    
    while True:
        line = f.readline();    # read line by line

        if line == '':    # check for EoF
            break;
  
        if (line != '\n'):    # skip empty lines
            if line.split()[1] == 'x':
                # get the number of probes and allocate memory
                noProbes = len(line.split())-2;
                time = np.array([])
                field = -1
                    
                # process the probe x-locations
                x = np.array([float(i) for i in line.split()[2:noProbes+2]])

            elif line.split()[1] == 'y':
                y = np.array([float(i) for i in line.split()[2:noProbes+2]])
                    
            elif line.split()[1] == 'z':
                z = np.array([float(i) for i in line.split()[2:noProbes+2]])
                
            elif line.split()[1] == 'deltaT':
                deltaT = np.array([float(i) for i in line.split()[2:noProbes+2]])

            elif line.split()[0] != '#':
                line = (line.replace('(',' ').replace(')',' ').replace(',',' ')).split()
                # check if saving 3 or 5 terms
                if field == -1 and ((len(line)-1)/noProbes)%3 == 0:
                    field = [np.zeros((0,3)) for i in range(noProbes)]
                elif field == -1 and ((len(line)-1)/noProbes)%5 == 0:
                    field = [np.zeros((0,5)) for i in range(noProbes)]
                
                # convert all to floats
                vals = [float(s) for s in line]
                
                time = np.append(time, vals[0])
                for i in range(0,noProbes):
                    if ((len(line)-1)/noProbes)%3 == 0:
                        field[i] = np.vstack([field[i],np.array([vals[(i*3+1) : (i*3+1+3)]])])
                    elif ((len(line)-1)/noProbes)%5 == 0:
                        field[i] = np.vstack([field[i],np.array([vals[(i*5+1) : (i*5+1+5)]])])
    f.close()
    return time, field, np.transpose(np.vstack([x,y,z])), deltaT
    
#============
# WRITING - INTERNAL
#============

def writeHeader(ret=False):
    s =  "/*--------------------------------*- C++ -*----------------------------------*\\\n"
    s += "| =========                 |                                                  |\n"
    s += "| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox            |\n"
    s += "|  \\    /   O peration     | Version:  2.2.2                                  |\n"
    s += "|   \\  /    A nd           | Web:      www.OpenFOAM.org                       |\n"
    s += "|    \\/     M anipulation  |                                                  |\n"
    s += "\*----------------------------------------------------------------------------*/\n"
    if ret:
        return s
    else:
        print(s)

def writeFoamFileLabel(ret=False,Version='2.0',Format='ascii',Class='dictionary',name='unnamedFile'):
    s =  'FoamFile\n'
    s += '{\n'
    s += '     version     {};\n'.format(Version)
    s += '     format     {};\n'.format(Format)
    s += '     class     {};\n'.format(Class)
    s += '     object     {};\n'.format(name)
    s += '}\n'
    if ret:
        s += writeSeparator(True)
        return s
    else:
        print(s)
        writeSeparator();

def writeSeparator(ret=False):
    s = '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n'
    if ret:
        return s
    else:
        print(s)

def writePointsList(name, points, ret=False):
    """
    Writes a list of points in the appropriate format
    """
    s = "{}\n".format(name)
    s += '(\n'
    for i in range(points.shape[0]):
        s += '\t({:.6e} {:.6e} {:.6e})\n'.format(points[i,0], points[i,1], points[i,2])
    s += ');\n'
    if ret:
        return s
    else:
        print(s)

def writeBlock(pointIndices, nElems, expRat, grading='simpleGrading', type='hex', ret=False):
    """
    Write a hex block for use inside blockMeshDict
    """
    retStr = type + ' ( ';
    for i in range(len(pointIndices)):
        retStr += str(pointIndices[i]) + ' '
    retStr += ') (' + str(nElems[0]) + ' ' + str(nElems[1]) + ' ' + str(nElems[2]) + ') '
    retStr += grading
    retStr += ' ( '
    for e in expRat:
        retStr += str(e) + ' '
    retStr += ')'
    if ret:
        return retStr
    else:
        print(retStr)

def writeBoundary(patch, faces, patchType='patch', additionalKeywords=[], ret=False):
    """
    Write a boundary definition for use inside blockMeshDict
    """
    retStr = "{}\n".format(patch)
    retStr += "{\n"
    retStr += "\ttype {};\n".format(patchType)
    if len(additionalKeywords) != 0:
        for i in range(len(additionalKeywords)):
            retStr += "\t{}\n".format(additionalKeywords[i])
            
    retStr += "\tfaces\n"
    retStr += "\t(\n"

    # TODO looking back at it, is this distinction necessary?
    if (type(faces[0])==type([])):
        # more then one set of points contributing to the boundary
        s = ''
        for i in range(len(faces)):
            s += '\t\t( '
            for j in range(len(faces[i])):
                s += str(faces[i][j]) + ' '
            s += ')\n'

    else:
        # boundary defined by a single set of points
        s = '\t\t( '
        for i in range(len(faces)):
            s += str(faces[i]) + ' '
        s += ')\n'

    retStr += s
    retStr += "\t);\n"
    retStr += "}\n"
    
    if ret:
        return retStr
    else:
        print(retStr)

def writeEdge(edgeType, start, end, points, ret=False):
    """
    write an edge for use in blockMeshDictFile
    """
    retStr = edgeType + ' ' + str(start) + ' ' + str(end) + ' '
    if (type(points) == type('word')): # include a pre-specified list of points
        retStr += '$' + points
    elif len(points.shape) == 2: # 2d numpy array of points
        for i in range(points.shape[0]):
            retStr += '( ' + str(points[i,0]) + ' ' + str(points[i,1]) + ' ' + str(points[i,2]) + ' )\n'
    else: # a single point as numpy 1d array
        retStr += '( ' + str(points[0]) + ' ' + str(points[1]) + ' ' + str(points[2]) + ' )\n'
    if not retStr.endswith("\n"):
        retStr += "\n"
    if ret:
        return retStr
    else:
        print(retStr)

#============
# WRITING - DICTIONARIES/FILES
#============

def writeProbeDict(path,filename,xyz,fields = ['U','p'],runtime=False,name = 'probes',outputControl='timeStep',writeInterval='10'):
    """
    Export a probes dictionary given the path, filename and the probe locations
    By default samples for velocity and pressure
    """
    path += filename;
    sys.stdout = f = open(path,'w')
    
    writeHeader();

    if not(runtime):
        writeFoamFileLabel(name='probesDict')
    print('')
    writeSeparator();
    print('')
    if runtime:
        print(name)
        print('{')
        print('    type probes;')
        print('    functionObjectLibs ( "libsampling.so");')
        print('')
        print('    outputControl ',outputControl,';')
        print('    outputInterval ',writeInterval,';')
        print('')
    print('    fields')
    print('    (')
    for i in range(0,len(fields)):
        print('        ',fields[i])
    print('    );')
    print('')
    print('    probeLocations')
    print('    (')
    
    for i in range(0,xyz.shape[0]):
        print('        (',xyz[i,0],' ',xyz[i,1],' ',xyz[i,2],')')
    
    print('    );')
    if runtime:
        print('}')
    print('')
    
    writeSeparator();

    f.close();
    sys.stdout = sys.__stdout__;

def writeSetsDict(path,filename,name,fields,xyz,setNames,patches,outputControl='timeStep',writeInterval='10',
                        setTypeName = 'sets',typeName = 'patchCloud', libname = 'libsampling',
                        additionalKeywords = []):
    tab = '    '
    
    path += filename;
    sys.stdout = f = open(path,'w')
    
    writeHeader();
    print('')
    writeSeparator();
    print('')
    print(name)
    print('{')
    print(tab,'type '+setTypeName+';')
    print(tab,'functionObjectLibs ( "'+libname+'.so");')
    print('')
    print(tab,'outputControl ',outputControl,';')
    print(tab,'outputInterval ',writeInterval,';')
    print('')
    print(tab,'setFormat   raw;')
    print(tab,'interpolationScheme cellPoint;')
    print('')
    print(tab,'fields')
    print(tab,'(')
    for i in range(0,len(fields)):
        print(tab*2,'    ',fields[i])
    print(tab,');')
    print('')
    print(tab,'sets')
    print(tab,'(')
    
    for iName in range(len(setNames)):
        print(tab*2,setNames[iName])
        print(tab*2,'{')
        print(tab*3,'type ' + typeName+';')
        for keyword in additionalKeywords[iName]:
            print(tab*3,keyword)
        
        if len(patches) > 0:
            print(tab*3,'patches')
            print(tab*3,'(')
            for patch in patches:
                print(tab*4,patch)
            print(tab*3,');')
        
        if len(xyz) > 0:
            print(tab*3,'points')
            print(tab*3,'(')
            for i in range(0,xyz[iName].shape[0]):
                print(tab*4,'(',xyz[iName][i,0],' ',xyz[iName][i,1],' ',xyz[iName][i,2],')')
            print(tab*3,');')
        print(tab*2,'}')
    
    print(tab,');')
    print('}')
    print('')
    
    writeSeparator();

    f.close();
    sys.stdout = sys.__stdout__;

def writeSamplingLinesDict(path,filename,xyzStart,xyzEnd,N,fields = ['U','p'],runtime=False,
                           name = 'lineInSpace',outputControl='timeStep',writeInterval='10',
                           appendXloc=True,typeName = 'sets', libname = 'libsampling'):
    """
    Export a sampling line dictionary given the path, filename and the line locations
    By default samples for velocity and pressure
    """
    path += filename;
    sys.stdout = f = open(path,'w')
    
    writeHeader();

    if not(runtime):
        writeFoamFileLabel(name='sampleDict')
    print('')
    writeSeparator();
    print('')
    if runtime:
        print(name)
        print('{')
        print('    type ',typeName,';')
        print('    functionObjectLibs ( "',libname+'.so");')
        print('')
        print('    outputControl ',outputControl,';')
        print('    outputInterval ',writeInterval,';')
        print('')
        print('    interpolationScheme cellPoint;')
        print('    setFormat   raw;')
        print('')
    print('    fields')
    print('    (')
    for i in range(0,len(fields)):
        print('        ',fields[i])
    print('    );')
    print('')
    print('    sets')
    print('    (')
    
    for i in range(0,xyzStart.shape[0]):
        if appendXloc:
            print('        ',name+'_'+str(xyzStart[i,0]))
        else:
            print('        ',name+'_'+str(i))
        print('         {')
        print('             type        uniform;')
        print('             axis        distance;')
        print('             start       (',xyzStart[i,0],xyzStart[i,1],xyzStart[i,2],');')
        print('             end         (',xyzEnd[i,0],xyzEnd[i,1],xyzEnd[i,2],');')
        print('             nPoints     ',N,';')
        print('         }')
    
    print('    );')
    if runtime:
        print('}')
    print('')
    
    writeSeparator();

    f.close();
    sys.stdout = sys.__stdout__;
    
def savePointsToOpenFOAMFormat(path,filename,x,y,z):
    """
    Save a lsit of points denoted by vectors of x,y,z locations to a file in a specified
    directory
    """
    path += filename;
    sys.stdout = f = open(path,"w")
    
    writeHeader();
    
    for i in range(len(x)-1,-1,-1):
        print('(',x[i],' ',y[i],' ',z[i],')')
    
    writeSeparator();
    
    f.close() 
    sys.stdout = sys.__stdout__;

def savePointToOpenFOAMFormat(path,filename,x,y,z):
    """
    Save a single point denoted by x,y,z location to a file in a specified directory
    """
    path += filename;
    sys.stdout = f = open(path,"w")
    
    writeHeader();

    print('(',x,' ',y,' ',z,')')
    
    writeSeparator();
    
    f.close() 
    sys.stdout = sys.__stdout__;

def savePointsToObj(filename, points): # wrapper -> too lazy to update the references
    pointsToObj(filename, points)

def pointsToObj(filename, points):
    """
    Save the locations of the points to an .obj file format.
    """
    sys.stdout = f = open(filename,"w")
    
    for j in range(0,points.shape[0]):
        print('v ',points[j,0],points[j,1],points[j,2])
    print('\n')

    for j in range(0,points.shape[0]):
        print('p ',j+1)
    
    f.close() 
    sys.stdout = sys.__stdout__;


#filepath = '/home/artur/OpenFOAM/artur-2.2.2/run/rotatingProbeTest/postProcessing/acoustics/0/myAcoustics.dat'
#importOpenFoamProbeData(filepath,'s')
#filepath = '/home/artur/OpenFOAM/artur-2.2.2/run/rotatingProbeTest/postProcessing/myProbes/0.000979154/U'
#importOpenFoamProbeData(filepath,'v',2)
#print(importOpenFoamPlaneData3('inputDataFiles/p_plane.raw')
#x,f=importOpenFoamSurfaceSamplingData('/home/artur/OpenFOAM/artur-2.2.2/run/NACA0012_LES/case_simple_2d/postProcessing/surfaceSamplingProbes/7502/lowerSurface_U.xy')
#print(importPointListFromDict('/home/artur/OpenFOAM/artur-2.2.2/run/NACA0012_LES/case_simple_2d/system/surfaceSamplingProbesDict','points').shape
"""
import matplotlib.pyplot as plt

N = [357]#90,179,357,713]#,1425]
for n in N:
    filename = 'inputDataFiles/driver-'+str(n)+'-1zn-narrow2d.p2dfmt'
    
    f = open(filename,'r')
    
    linesSkipped = 0
    xy = np.array([])
    
    while True:
        line = f.readline();
    
        if line == '':
            break;
    
        linesSkipped += 1
        if (linesSkipped > 2) and (line != '\n'):
            xy = np.append(xy,np.array([float(i) for i in line.split()]))
    
    f.close()
    
    print(n,':',xy.shape
    
    x = xy[0:n]
    y = xy[len(xy)-n:len(xy)]
    Y =[]
    for i in range(97):
        Y.append( xy[len(xy)/2+i*n] )
        print(xy[len(xy)/2+i*n],','
    print(len(Y)
    plt.plot(Y)
    plt.show()
    plt.plot(x,y,'x-')
    #plt.show()

print('['
for i in range(len(x)):
    print('[',x[i],',',y[i],'],'
print(']'
"""
