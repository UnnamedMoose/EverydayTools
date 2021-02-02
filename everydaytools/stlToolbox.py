# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 11:41:51 2013

@author: artur
"""

import numpy as np
import sys
import copy

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.patches import FancyArrowPatch
#from mpl_toolkits.mplot3d import proj3d

# ==============
# HELPER FUNCTIONS
# ==============

def cross(v1,v2):
    """
    returns the cross-product of two 3D vectors
    """
    return np.array([ (v1[1]*v2[2]-v2[1]*v1[2]),
                     -(v1[0]*v2[2]-v2[0]*v1[2]),
                     (v1[0]*v2[1]-v2[0]*v1[1])]);
    
def makeVector(start,end):
    """
    returns a vector in 3D given its start and end points
    """
    return np.array( [end[0]-start[0],end[1]-start[1],end[2]-start[2]] );
    
def vecLen(v):
    """
    returns the length of a 3D vector
    """
    return (v[0]**2.+v[1]**2.+v[2]**2.)**0.5;

# ==============
# face CLASS IMPLEMENTATION
# ==============

class face:
    """
    Holds the cartesian coordinates of a triangular face;
    Construct from points and optionally a normal (calculated by default)
    
    Methods:
        getNormal - calculates the normal of the face using the right-hand rule
    """
    x = np.zeros(3);
    y = np.zeros(3);
    z = np.zeros(3);
    
    def getNormal(self):
        v21 = makeVector([self.x[1],self.y[1],self.z[1]], [self.x[0],self.y[0],self.z[0]])
        v23 = makeVector([self.x[1],self.y[1],self.z[1]], [self.x[2],self.y[2],self.z[2]])
        
        self.normal = cross(v23,v21)
    
    def __init__(self,x,y,z,normal = -1):  # constructor
        self.x = x;
        self.y = y;
        self.z = z;
        if normal == -1:
            self.getNormal();
        else:
            self.normal = normal;
#        print(self.normal)
        
    def __str__(self):
        m = np.sqrt(self.normal[0]**2.+self.normal[1]**2.+self.normal[2]**2.)
        retStr = "  facet normal %.8e %.8e %.8e\n" % (self.normal[0]/m, self.normal[1]/m, self.normal[2]/m)
        retStr += "    outer loop\n"
        retStr += "      vertex   %.8e %.8e %.8e\n" % (self.x[0],self.y[0],self.z[0])
        retStr += "      vertex   %.8e %.8e %.8e\n" % (self.x[1],self.y[1],self.z[1])
        retStr += "      vertex   %.8e %.8e %.8e\n" % (self.x[2],self.y[2],self.z[2])
        retStr += "    endloop\n"
        retStr += "  endfacet"
        return retStr

# ==============
# stlObject CLASS IMPLEMENTATION
# ==============

class stlObject:
    """
    Holds a list of faces and a name of the object
    Construct with a list of faces (empty by default) and a name (='' by default)
    
    Methods:
        preview - plots the object in 3D space, may preview face centres, vertices and normals
        toString - used primarily for saving to file but may be used to print on the screen if default
                filename (=-1) is used
        translate - Moves this stlObject in cartesian space along the principle axes.
                returns a copy of this object
        rotate - rotates this stlObject around the principle axes by a specified amount.
                Also returns a copy of this object
        addSurfaceFromGrid - Takes a uniform (not necessarily uniformly distributed) grid of points,
                        triangulates its rectangular faces and adds them to the stl.
                        Can specify whether the current faces should be overwritten or not.
        addSolid - appends a given stlObject to the current one
    """
    
    name = '';
    faces = np.zeros(0);
    
    def __init__(self,faces=[],name=''):
        self.faces = faces;
        self.name = name;
      
#    def preview(self, showCentres=False, showPoints=False, showNormals=False, scaleNormals=1.):
#        """
#        plots the object in 3D space, may preview face centres, vertices and normals
#        """        
#        class Arrow3D(FancyArrowPatch):
#            def __init__(self, xs, ys, zs, *args, **kwargs):
#                FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
#                self._verts3d = xs, ys, zs
#        
#            def draw(self, renderer):
#                xs3d, ys3d, zs3d = self._verts3d
#                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
#                self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
#                FancyArrowPatch.draw(self, renderer)
#        
#        faces = self.faces;
#        
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        for i in range(0,len(faces)):
##            faces[i].getNormal() TODO no need since this gets done in the constructor
#            if showNormals:
#                x = [np.mean(faces[i].x), np.mean(faces[i].x)+faces[i].normal[0] * scaleNormals]
#                y = [np.mean(faces[i].y), np.mean(faces[i].y)+faces[i].normal[1] * scaleNormals]
#                z = [np.mean(faces[i].z), np.mean(faces[i].z)+faces[i].normal[2] * scaleNormals]
##                print(faces[i].normal)
#            
#                a = Arrow3D(x, y, z, mutation_scale=20, lw=1, arrowstyle="-|>", color="k");    # normals
#                ax.add_artist(a)
#            
#            if showCentres:
#                ax.scatter(np.mean(faces[i].x),np.mean(faces[i].y),np.mean(faces[i].z),c='green'); # centre of each triangle
#            
#            xp = np.append(faces[i].x,faces[i].x[0])
#            yp = np.append(faces[i].y,faces[i].y[0])
#            zp = np.append(faces[i].z,faces[i].z[0])
#            ax.plot(xp,yp,zp,c='red');      # edges of each face
#            
#            if showPoints:
#                ax.scatter(xp,yp,zp,c='blue');  # corner points of each face
#            
#        ax.set_xlabel('x')
#        ax.set_ylabel('y')
#        ax.set_zlabel('z')
#        plt.show()
    
    def toString(self,filename = -1):
        """
        used primarily for saving to file but may be used to print on the screen if default
        filename (=-1) is used
        """
        if filename != -1:
            sys.stdout = f = open(filename,"w")
        
        retString = 'solid %s\n' % self.name
        for i in range(0,len(self.faces)):
            retString += self.faces[i].__str__()
            retString += '\n'
        retString +=  'endsolid'# %s' % self.name
        
        print(retString)
        if filename != -1:
            f.close() 
    
        # redirect the output back to the screen
        sys.stdout = sys.__stdout__;
   
    def translate(self,dx,dy,dz):
        """
        Moves this stlObject in cartesian space along the principle axes.
        returns a copy of this object
        """
        for i in range(0,len(self.faces)):
            self.faces[i].x += dx;
            self.faces[i].y += dy;
            self.faces[i].z += dz;
        return copy.deepcopy(self)
         
    def rotate(self,CoR,thetaX,thetaY,thetaZ):
        """
        rotates this stlObject around the principle axes by a specified amount.
        Also returns a copy of this object
        """

        # translate the object to account for the CoR
        self.translate(-CoR[0],-CoR[1],-CoR[2])
        
        for i in range(0,len(self.faces)):
            for j in range(0,self.faces[0].x.shape[0]):
                x = self.faces[i].x[j];
                y = self.faces[i].y[j];
                z = self.faces[i].z[j];
                #r = (self.faces[i].x[j]**2.+self.faces[i].y[j]**2.+self.faces[i].z[j]**2.)**.5;
                
                # x-axis rotation
                y1 = y*np.cos(float(thetaX)/180.*np.pi)-z*np.sin(float(thetaX)/180.*np.pi)
                z1 = z*np.cos(float(thetaX)/180.*np.pi)+y*np.sin(float(thetaX)/180.*np.pi)
                
                # y-axis rotation
                z2 = z1*np.cos(float(thetaY)/180.*np.pi)-x*np.sin(float(thetaY)/180.*np.pi)
                x1 = x*np.cos(float(thetaY)/180.*np.pi)+z1*np.sin(float(thetaY)/180.*np.pi)
                
                # z-axis rotation
                x2 = x1*np.cos(float(thetaZ)/180.*np.pi)-y1*np.sin(float(thetaZ)/180.*np.pi)
                y2 = y1*np.cos(float(thetaZ)/180.*np.pi)+x1*np.sin(float(thetaZ)/180.*np.pi)
                
                self.faces[i].x[j] = x2;
                self.faces[i].y[j] = y2;
                self.faces[i].z[j] = z2;
            
        # translate the object back to conform with the original coordinate system
        self.translate(CoR[0],CoR[1],CoR[2])
        return copy.deepcopy(self);
    
    def scale(self,origin,sx,sy,sz):
        # WRONG!!! -> make a line from the origin to the point of interest and move the point along that line
        scaledObject = self;
        # translate the object to account for the CoR
        scaledObject = scaledObject.translate(-origin[0],-origin[1],-origin[2])
        for i in range(0,len(scaledObject.faces)):
            scaledObject.faces[i].x =scaledObject.faces[i].x*sx;
            scaledObject.faces[i].x =scaledObject.faces[i].x*sx;
            scaledObject.faces[i].x =scaledObject.faces[i].x*sx;
        scaledObject = scaledObject.translate(origin[0],origin[1],origin[2])
        return scaledObject
        
    def addSurfaceFromGrid(self,x,y,z,overwrite = True):
        """
        Takes a uniform (not necessarily uniformly distributed) grid of points,
        triangulates its rectangular faces and adds them to the stl.
        Can specify whether the current faces should be overwritten or not.
        """
        if overwrite:
            self.faces = [];

        N = x.shape[0]
        M = x.shape[1]
        for j in range(0,N-1):
            for i in range(0,M-1):
                # each rectangle is divided into 2 triangles
                # triangles with no area are not added => can duplicate points in the grid to
                # add just a triangle instead of a rectangle
                if ~((x[j,i]==x[j+1,i+1]) and (y[j,i]==y[j+1,i+1]) and (z[j,i]==z[j+1,i+1])or
                    (x[j,i]==x[j+1,i] and y[j,i]==y[j+1,i] and z[j,i]==z[j+1,i]) or
                    (x[j+1,i]==x[j+1,i+1] and y[j+1,i]==y[j+1,i+1] and z[j+1,i]==z[j+1,i+1])):
                    self.faces.append(face(np.array([x[j,i],x[j+1,i+1],x[j+1,i]]),
                                      np.array([y[j,i],y[j+1,i+1],y[j+1,i]]),
                                      np.array([z[j,i],z[j+1,i+1],z[j+1,i]])))
                                      
                if ~((x[j,i]==x[j+1,i+1]) and (y[j,i]==y[j+1,i+1]) and (z[j,i]==z[j+1,i+1])or
                    (x[j,i]==x[j,i+1] and y[j,i]==y[j,i+1] and z[j,i]==z[j,i+1]) or
                    (x[j,i+1]==x[j+1,i+1] and y[j,i+1]==y[j+1,i+1] and z[j,i+1]==z[j+1,i+1])):
                    self.faces.append(face(np.array([x[j,i],x[j,i+1],x[j+1,i+1]]),
                                      np.array([y[j,i],y[j,i+1],y[j+1,i+1]]),
                                      np.array([z[j,i],z[j,i+1],z[j+1,i+1]])))
                                      
    def addSolid(self,addedSolid):
        """
        appends a given stlObject to the current one
        """
        newFaces = (len(self.faces)+len(addedSolid.faces))*[0.];
        
        for i in range(0,len(self.faces)):
            newFaces[i] = face(self.faces[i].x,self.faces[i].y,self.faces[i].z)
        for i in range(0,len(addedSolid.faces)):
            newFaces[i+len(self.faces)] = face(addedSolid.faces[i].x,addedSolid.faces[i].y,addedSolid.faces[i].z)
        
        self.faces = newFaces;

# ==============
# IMPORT FUNCTIONS
# ==============

def importStlObject(filename):
    try:
        f = open(filename,'r')
        facetIndex = -1
        vertexIndex = -1
        vertices = [0]*3
        faces = []
        name = ''

        while True:
            line = f.readline() # read line by line

            if line == '': # check for EoF
                break
                
            if (line != '\n') or (line.split()[0] == 'endsolid'): # skip empty lines
                if line.split()[0] == 'solid':
                    name = line.split()[1]
    
                elif line.split()[0] == 'facet':
                    normal = [float(line.split()[2]),float(line.split()[3]),float(line.split()[4])]
                    facetIndex += 1
    
                elif line.split()[0] == 'vertex':
                    vertexIndex += 1
                    vertices[vertexIndex] = [float(line.split()[1]),float(line.split()[2]),float(line.split()[3])] 
                    
                    if vertexIndex == 2:
                        vertexIndex = -1
                        faces.append(face(np.array([vertices[0][0],vertices[1][0],vertices[2][0]]),
                                  np.array([vertices[0][1],vertices[1][1],vertices[2][1]]),
                                  np.array([vertices[0][2],vertices[1][2],vertices[2][2]]),normal))
                                  
        return stlObject(faces, name)
    except Exception:
        print('Error while reading the input file.. Terminating prematurely...\n')
        sys.exit()

if __name__ == "__main__":
	"""
	N = 5
	x = np.zeros((N,N))
	y = np.zeros((N,N))
	z = np.zeros((N,N))
	
	for i in range(N):
		x[:,i] = np.linspace(-0.1,1.1,N)
		y[i,:] = np.linspace(-0.1,1.1,N)
		z[i,:] = np.linspace(-0.1,1.1,N)
	
	obj1 = stlObject()
	obj1.addSurfaceFromGrid(x,y,z)
	
	#obj1.addSurfaceFromGrid(np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0]]),
	#                np.array([[0.,0.,0.],[-1.,0.,1.],[-1.,0.,1.]]),
	#                        np.array([[2.,2.,2.],[1.,1.,1.],[0.,0.,0.]]))
	    
	#obj2 = copy.deepcopy(obj1).translate(1)
	#obj1.addSolid(obj2)
	#obj1.translate(1,1,1)
	#obj1.rotate([0.,0.,0.],90,180,0)
	obj1.preview(showCentres=True, showPoints=True, showNormals=True)
	obj1.toString('testSurface.stl');
	"""
	
	obj1 = importStlObject("../../ReFRESCO/calcs/testCases_acoustics/testCase1_meshSelection/inputs/controlSurface_box_small.stl")
#	obj1.preview(showCentres=False, showNormals=True, scaleNormals=0.05)