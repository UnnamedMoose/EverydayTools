# -*- coding: utf-8 -*-
"""
Created on Sun Feb 09 16:55:29 2014

@author: Artur
"""

import sys
import time
import numpy as np

#============
# HELPER FUNCTIONS
#============

def igInt(i,length=7):
    """
    returns a string denoting an integer of appropriate length
    """
    return '%*i' % (length,i)
    
def cutStrAndAlignRight(s,length=8):
    """
    Cuts the string down to the specified length and if it's shorter then it
    aligns it to the right
    """
    
    if len(s) >= length:
        return s[0:8]
    else:
        return ' '*(length-len(s))+s
        
def writeGeneralExpr(val,Type):
    """
    format an expression to put into the general section of the iges file; type can
    be 'str', 'int' or 'dim'
    """
    if Type == 'str':
        if len(val) > 0:
            return str(len(val))+'H'+val
        else:
            return ''
    elif Type == 'int':
        return str(val)
    elif Type == 'dim':
        return str(val)+'D0'
    else:
        raise IOError('Unrecognised general expression entry type '+Type+' ...')
        
def formatParamStr(s,generalLine,length=64):
    """
    modify so that len(last_line) = length (default=72)
    Add the general line pointer at the end of each line
    """
   
    # check the remainder and append with spaces as required, divide into elements
    s = s.split(',')

    retStr = ''
    newLine = ''
    
    # iterate over each element
    for i in s:
        if (';' in i): # last entry
            # fits in -> add it and start a new line
            if (len(newLine)+len(i)) <= (length-1):
                newLine += i
                newLine += ' '*(length-len(newLine)) + '%8d' % generalLine
                retStr += newLine
            # too long to fit in -> finish current line and then add to the next one
            # and finish
            else:
                newLine += ' '*(length-len(newLine)) + '%8d' % generalLine
                retStr += newLine
                newLine = i
                newLine += ' '*(length-len(newLine)) + '%8d' % generalLine
                retStr += newLine
                
        else: # intermediate entries
            # fits in -> easy, just add and continue
            if (len(newLine)+len(i)) <= (length-1):
                newLine += i + ','
            # start a new line when needed
            else:
                newLine += ' '*(length-len(newLine)) + '%8d' % generalLine
                retStr += newLine
                newLine = i + ','

    return retStr    

def makeSection(secStr,secLabel,length=72,usedLength=72):
    """
    divides the section into lines, appends them with an appropriate ending;
    returns the number of lines and the rearranged string
    """
    nLines = len(secStr)/usedLength

    if (len(secStr)%usedLength) > 0: # remainder
        nLines += 1
    
    retStr = ''
    
    for i in range(int(nLines)):
        nTailSpaces = length - len(secStr[i*usedLength:(i+1)*usedLength])
        retStr += secStr[i*usedLength:(i+1)*usedLength] + ' '*nTailSpaces + secLabel + igInt(i+1) + '\n'
    
    return retStr, nLines

def makeTerminator(nS,nG,nD,nP):
    """
    Creates the terminator expression summarising the number of lines in each section
    of the file
    """
    return makeSection('S' + igInt(nS) + 'G' + igInt(nG) + 'D' + igInt(nD) + 'P' + igInt(nP),'T')

#============
# IGES ENTITY BASE CLASS
#============

class igesEntity(object):
    """
    Basic class that contains the methods for storing, manipulating and saving the
    iges entity data, i.e. a line, curve, surface, or whatever.
    """
    
    def __init__(self,entity=0,generalLine=0,paramLine=0,structure=0,lineFontPattern=0,level=0,view=0,
                 transformMatrix=0,labelDisp=0,statusNumber=[0]*4,lineWeight=0,colour=0,
                 noParamLines=0,form=0,name='',index=0,params=[]):
        if len(statusNumber) != 4:
            raise IOError('Status number should be vector with 4 elements ...')
        self.entity=entity
        self.generalLine=generalLine
        self.paramLine=paramLine
        self.structure=structure
        self.lineFontPattern=lineFontPattern
        self.level=level
        self.view=view
        self.transformMatrix=transformMatrix
        self.labelDisp=labelDisp
        self.statusNumber=statusNumber
        self.lineWeight=lineWeight
        self.colour=colour
        self.noParamLines=noParamLines
        self.form=form
        self.name=name
        self.index=index
        self.params=params
        
    def dirStr(self):
        """
        Return a single line string -> this is split in the top level methods
        """
        retStr = '%8d%8d%8d%8d%8d%8d%8d%8d' % (self.entity,self.paramLine,self.structure,
                                                  self.lineFontPattern,self.level,self.view,
                                                  self.transformMatrix,self.labelDisp)
        
        retStr += ('%2d%2d%2d%2d' % (self.statusNumber[0],self.statusNumber[1],
                                   self.statusNumber[2],self.statusNumber[3]))#.replace(' ','0')
        
        retStr += '%8d%8d%8d%8d%8d' % (self.entity,self.lineWeight,self.colour,self.noParamLines,
                                                   self.form)
                                                   
        retStr += ' '*8*2
        
        retStr += cutStrAndAlignRight(self.name)
        
        retStr += '%8d' % self.index
        
        return retStr
    
    def paramStr(self):
        """
        return parameters as a single line string -> split elsewhere
        """
        retStr = str(self.entity)
        for i in range(len(self.params)):
            retStr += ',' + str(self.params[i])
        retStr += ';'
        
        return retStr

#============
# IGES OBJECT CLASS
#============

class igesObject(object):
    """
    Class holding multiple iges entities and managing the IO operations
    """
    def  __init__(self,objects,exportProdId='',name='',natSystem='',exporter='Iges Toolbox',
                  nIntBits=32,maxFloatPow=38,floatDigits=6,maxDoubPow=99,doubleDigits=15,
                  importProdId='',spaceScale=1.0,unitsName='M',maxLineGrads=1,maxLineWeight=.08,
                  minRes=.01,maxCoordVal=1e4,author='',organisation='',igesVer=6,draftStd=0):
        self.objects=objects
        
        self.exportProdId=exportProdId
        self.name=name
        self.natSystem=natSystem
        self.exporter=exporter
        
        self.nIntBits=nIntBits
        self.maxFloatPow=maxFloatPow
        self.floatDigits=floatDigits
        self.maxDoubPow=maxDoubPow
        self.doubleDigits=doubleDigits
        
        self.importProdId=importProdId
        self.spaceScale=spaceScale
        
        units=['M','MM']
        unitsLabels=[6,2]
        for i in range(len(units)):
            if units[i] == unitsName:
                unitsFlag=unitsLabels[i]
                break
            else:
                unitsFlag = False
        
        if unitsFlag:
            self.unitsFlag=unitsFlag
            self.unitsName=unitsName
        else:
            raise IOError('Unrecognised units type: '+unitsName+' ...')
            
        self.maxLineGrads=maxLineGrads
        self.maxLineWeight=maxLineWeight
        
        self.date = time.strftime("%Y%m%d.%H%M%S")
        self.lastMod = self.date
        
        self.minRes=minRes
        self.maxCoordVal=maxCoordVal
        self.author=author
        self.organisation=organisation
        self.igesVer=igesVer
        self.draftStd=draftStd
        
        self.userSubset=''
        
    def generalStr(self):
        """
        Print the header, or the general section, of the iges file
        """
        retStr = ',,' # use standard delimiters
        retStr += writeGeneralExpr(self.exportProdId,'str') + ','
        retStr += writeGeneralExpr(self.name,'str') + ','
        retStr += writeGeneralExpr(self.natSystem,'str') + ','
        retStr += writeGeneralExpr(self.exporter,'str') + ','
        
        retStr += writeGeneralExpr(self.nIntBits,'int') + ','
        retStr += writeGeneralExpr(self.maxFloatPow,'int') + ','
        retStr += writeGeneralExpr(self.floatDigits,'int') + ','
        retStr += writeGeneralExpr(self.maxDoubPow,'int') + ','
        retStr += writeGeneralExpr(self.doubleDigits,'int') + ','
        
        retStr += writeGeneralExpr(self.importProdId,'str') + ','
        retStr += writeGeneralExpr(self.spaceScale,'dim') + ','
        retStr += writeGeneralExpr(self.unitsFlag,'int') + ','
        retStr += writeGeneralExpr(self.unitsName,'str') + ','
        retStr += writeGeneralExpr(self.maxLineGrads,'int') + ','
        
        retStr += writeGeneralExpr(self.maxLineWeight,'dim') + ','
        retStr += writeGeneralExpr(self.date,'str') + ','
        
        retStr += writeGeneralExpr(self.minRes,'dim') + ','
        retStr += writeGeneralExpr(self.maxCoordVal,'dim') + ','
        retStr += writeGeneralExpr(self.author,'str') + ','
        retStr += writeGeneralExpr(self.organisation,'str') + ','
        retStr += writeGeneralExpr(self.igesVer,'int') + ','
        retStr += writeGeneralExpr(self.draftStd,'int') + ',,' # don't specify last modified
        retStr += writeGeneralExpr(self.userSubset,'str') + ';'
        
        return retStr
        
    def __str__(self):
        """
        Divide all the stuff into the appropriate format of lines, count them
        and create the cross references, then create the sections and return
        """
        retStr,NS = makeSection(' ','S')
        
        tmpStr,NG = makeSection(self.generalStr(),'G')
        retStr += tmpStr
        
        N = len(self.objects)
        
        paramStrs = ['']*N
        paramStartLines = [1]*N
        noLines = [0]*N

        tmpStr = ''
        tmpStr2 = ''
        for i in range(N):
            # get the string describing the paraeters of each object
            self.objects[i].generalLine = 2*i+1
            paramStrs[i] = formatParamStr(self.objects[i].paramStr(),self.objects[i].generalLine)
            tmpStr += paramStrs[i]
            
            # how many lines this entry will take
            nL = len(paramStrs[i])/72
            if (len(paramStrs[i])%72) > 0:
                nL += 1
            noLines[i] = nL
            
            # start line in the parameters sections
            if i > 0:
                paramStartLines[i] = paramStartLines[i-1]+noLines[i-1]
                
            # update the object properties
            self.objects[i].paramLine = paramStartLines[i]
            self.objects[i].noParamLines = noLines[i]
            
            tmpStr2 += self.objects[i].dirStr()

        tmpStr2,ND = makeSection(tmpStr2,'D')
        retStr += tmpStr2

        tmpStr,NP = makeSection(tmpStr,'P')
        retStr += tmpStr

        retStr += makeTerminator(NS,NG,ND,NP)[0]
        
        return retStr
        
    def addObj(self,obj):
        self.objects.append(obj)

#============
# LINE - TYPE 110
#============
class lineEntity(igesEntity):
    
    def __init__(self, p1, p2, name='',noAssocPtrs=0,assocPtrs=0):      
        # set the parameter fields
        if (len(p1)!=3) or (len(p2)!=3):
            raise IOError('Point definitions must specify 3 coordinates! Received: '+
                len(p1)+' and '+len(p2))
            
        self.p1=np.array(p1)
        self.p2=np.array(p2)
        self.noAssocPtrs=noAssocPtrs
        self.assocPtrs=assocPtrs
        
        # call the default base constructor and set the fields appropriately
        super(lineEntity,self).__init__(entity=110,name=name,params=[self.p1[0],self.p1[1],
            self.p1[2],self.p2[0],self.p2[1],self.p2[2],self.noAssocPtrs,self.assocPtrs])
        
    def paramStr(self):
        # update the parameter values and use derived method to print the parameters
        self.params=[self.p1[0],self.p1[1],self.p1[2],self.p2[0],self.p2[1],self.p2[2],
                     self.noAssocPtrs,self.assocPtrs]
        
        return super(lineEntity,self).paramStr()

#============
# B-SPLINE - TYPE 126
#============
class bSplineEntity(igesEntity):
    
    def __init__(self, M, pts, name='', T=[], W=[], noAssocPtrs=0, assocPtrs=0):      
        # set the parameter fields
            
        self.M = M # degree of the basis function
        self.pts = np.array(pts)
        self.noAssocPtrs = noAssocPtrs
        self.assocPtrs = assocPtrs
        
        # assume the user doesn't know what they're doing with weights and knots
        # and just do a best guess; can be overriden if need be
        if len(T) == 0:
            self.T = np.hstack([[0]*self.M,[0,0.5,0.5,0.5,1],[1]*self.M])
        else:
            self.T = np.array(T)
        
        if len(W) == 0:
            self.W = np.ones(self.pts.shape[0])
        else:
            self.W = np.array(W)
        
        self.K = self.pts.shape[0]-1
        self.props = [
            0, # 0-non-planar, 1-planar
            0, # 0-open, 1-closed
            1, # 0-rational, 1-polynomial
            0, # 0-non-periodic, 1-periodic
            ]
        
        # call the default base constructor and set the fields appropriately
        super(bSplineEntity,self).__init__(entity=126,name=name,params=self.paramList())
    
    def paramList(self):
#        N = 1+self.K-self.M # no. segments
#        A = N+2*self.M
        
#        self.T = np.zeros(N+2*self.M)
#        T = np.hstack([[0]*self.M,np.linspace(0,1,N-1),[1]*self.M])
#        self.T = np.hstack([[0]*self.M,[0,0.5,0.5,0.5,1],[1]*self.M])
#        self.W = np.ones(self.pts.shape[0])
        
#        print self.M, N, A, self.K
#        print T
#        print W
        
        # wrap parameters in a single list
        paramsList = [self.K,self.M]+self.props
        
        # knots
        for i in range(self.T.shape[0]):
            paramsList.append(float(self.T[i]))
        
        # weights
        for i in range(self.W.shape[0]):
            paramsList.append(self.W[i])
        
        # points
        for i in range(self.pts.shape[0]):
            for j in range(self.pts.shape[1]):
                paramsList.append(self.pts[i,j])
                
        # normal (not required if non-planar)
        paramsList += [0.0,1.0,0.0]
        paramsList += [0.0,1.0,0] # ??
        
        paramsList += [self.noAssocPtrs,self.assocPtrs]

        return paramsList
    
    def paramStr(self):
        # update the parameter values and use derived method to print the parameters
        self.params = self.paramList()
        return super(bSplineEntity,self).paramStr()

if __name__ == "__main__":
	line0 = lineEntity([0.,0,0],[1,0,0])
	line1 = lineEntity([0.,0,0],[1,1,1])

	testPts = np.array([
		   [ 2.        ,  1.        ,  1.        ],
		   [ 2.        ,  1.33333333,  1.        ],
		   [ 1.83333333,  1.83333333,  1.        ],
		   [ 2.        ,  2.        ,  1.        ],
		   [ 2.16666667,  2.16666667,  1.        ],
		   [ 2.66666667,  2.        ,  1.        ],
		   [ 3.        ,  2.        ,  1.        ]])

	#testPts = np.array([[0,0,0],[1,0,0],[1,1,0]])

	spline0 = bSplineEntity(3, testPts, "someCurve")

	igesObj = igesObject([line0, line1, spline0])

	with open("test.iges","w") as f: f.write(str(igesObj))
#	print igesObj
