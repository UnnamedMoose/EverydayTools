# -*- coding: utf-8 -*-
"""
Copyright by Artur K. Lidtke, Univ. of Southampton, UK, 2016
 
Created on Wed Jan  6 13:27:47 2016
"""
 
import numpy as np

#================
# new implementation

#http://diyhpl.us/~bryan/papers2/IGES5-3_forDownload.pdf
#http://soliton.ae.gatech.edu/people/jcraig/classes/ae4375/notes/b-splines-04.pdf
#http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-basis.html

def blendingPolys(T,M,density,interpolateEnd=False):
    """ Computes the blending polynomials using recursive Cox-de Boor formulation
    Parameters
    ----------
        @param T - knot vector
        @param M - degree of the curve
        @param density - number of t values stretching across the valid curve region
            (curve only defined where sum of blending polynomials == 1.0)
    Returns
    ----------
        @param t - np.array(density,) valid values of the free t parameter at which the curve is defined
        @param B - numpy.array((N,t.shape[0])) array containing the blending polynomial
            values at each t, defined for N basis functions which may exist for the given
            T vector and M
    """
    # define the valid range over which the curve is defined
    t = np.linspace(T[M],T[-M-1],density)
    
    # define blending polynomials recursively for each degree k
    B = []
    for k in range(M+1):
        N = T.shape[0]-k-1 # number of basis functions for this degree
    
        # holds values of each basis function defined over t
        B.append( np.zeros((N,t.shape[0])) )
        
        # evaluate each basis function
        for i in range(N):
            # 0th degree is just a series of steps
            if k == 0:
                B[-1][i,:] = np.ones(t.shape) * (t>=T[i]) * (t<T[i+1])
            
            # use recursive Cox-de Boor formula for higher degrees
            else:
                # the conditional statements take care of a situation where some of the
                # knot values are repeated
                d = T[i+k] - T[i]
                if d > 0:
                    B[-1][i,:] += (t-T[i])*B[-2][i,:]/d
                
                d = T[i+k+1]-T[i+1]
                if d > 0:
                    B[-1][i,:] += (T[i+k+1]-t)*B[-2][i+1,:]/d
    
    # make sure the other end point gets interpolated if required
    if interpolateEnd or np.abs(np.mean(T[-M-1])-T[-1]) < 1e-24:
#    if (B[-1][-1,-2] > B[-1][-1,-3]) and B[-1][-1,-2] >= 0.5:
        B[-1][-1,-1] = 1.0

    # only return the values for the highest required degree since that's what the user wants
    return t,B[-1]

def spline(pts,M=3,density=101,interpolateEnds=True,interpolatePoints=False,T=[],retAll=False):
    """
    Creates a uniform B-spline of arbitrary degree
    Parameters
    ----------
        @param pts - np.array(Npoints,Ndim) control vertices in N-dimensional space
    Optional
    ----------
        @param M - order of the spline (1-linear, 2-parabolic, 3-cubic, etc.), default 3
        @param density - number of points on the spline
        @param interpolateEnds - interpolate ends
        @param interpolatePoints - interpolates ends and every third point, using the remaining
            ones to control the gradients (analogous to most CAD implementations); this only
            really works with M=3, for other M results may vary (use at own risk)
        @param T - override knot values, none of the interpolation arguments will be used
        @param retAll - return blending polynomials, knots, and free parameter as arrays
            alongside the spline points
    """
    # define the knot vector
    if len(T) > 0:
        pass
    
    elif interpolatePoints:
        # interpolate ends and repeat knots M times at control points
        Nknots = pts.shape[0]+2*M-(M-1)
        T = [0]*(M+1)
        for i in range(M+1,Nknots-(M+1),M):
            T += [T[-1]+1]*M
        T += [T[-1]+1]*(M+1)
        T = np.array(T)
    
    elif interpolateEnds:
        # use M+1 repeated values at the start and end
        T = np.hstack([[0]*M, np.arange(pts.shape[0]-M+1), [pts.shape[0]-M]*M])
    
    else:
        # use uniformly distributed knot values
        T = np.arange(pts.shape[0]+2*M-(M-1))

    T = np.array(T,dtype=np.float64)
    
    # compute blending polynomials
    t,B = blendingPolys(T,M,density,interpolatePoints)
    
    # construct the curve from the points and blending polynomials
    curve = np.zeros((t.shape[0],pts.shape[1]))

    for i in range(pts.shape[0]):
        for j in range(pts.shape[1]):
            curve[:,j] += pts[i,j]*B[i,:]
    
    if retAll:
        return curve,B,T,t
    else:
        return curve

def fitSplineThroughPoints(data,f=0.3,density=101,retAll=False):
    """ Fits a spline through a set of points in Cartesian space.
    Works best with more-or-less uniformly distributed points since these don't overconstrain
    the spline and thus kinks are avoided.
    
    Paramters
    ---------
        @param data - np.array(Npts,Ndimensions) of points to be interpolated
    Optional
    ---------
        @param f - controls how close the gradient points are to the points about
            which they are defined (0-on top of base points, 1-on the neighbour);
            works best with f about 0.2-0.4
        @param density - number of points on the spline
        @param retAll - return full b-spline representation, including knots and basis functions,
            as well as the control vertices created by this function
    """
    M=3
    
    # construct interpolation points
    pts = data[0,:]

    # first gradient control point - aligned with the 2nd data point
    s1 = data[1,:]-data[0,:]
    pts = np.vstack([pts, data[0,:] + f*s1])
    
    # add control points inside the curve
    for i in range(1,data.shape[0]-1):
        s0 = s1
        s1 = data[i+1,:]-data[i,:]
        
        pts = np.vstack([pts, data[i,:] - f*(s0+s1)/2., data[i,:], data[i,:] + f*(s0+s1)/2.])
    
    # add the end
    pts = np.vstack([pts, data[-1,:] - f*s1, data[-1,:]])
    
    # run the spline through the points
    if retAll:
        xyz,B,T,t = spline(pts,M=M,density=density,interpolatePoints=True,retAll=retAll)
        return xyz,B,T,t,pts
    else:
        return spline(pts,M=M,density=density,interpolatePoints=True,retAll=retAll)

def fitSplineThroughAirfoil(dataUpper,dataLower,f=0.2,fLeUpper=0.9,fLeLower=0.9,density=1001):
    """ Fits splines through the upper and lower surface of an airfoil.
    In the middle of the foil adjusts the control vertices and knots so that the spline
    passes through the specified points and remains continuous and smooth. At the leading
    edge assumes the foil surface is perpendicular to the mean thickness line and merges
    with continuous gradient into the line defined by points 1 and 2 (looking from the LE).
    
    Parameters
    ----------
        @param data - 2D or 3D np.array with foil offsets, rows - points, cols - (x,y) or (x,y,z);
            data ordered from the leading to the trailing edge
    
    Arguments
    ----------
        @param f - controls how close the gradient points are to the points about
            which they are defined (0-on top of base points, 1-on the neighbour);
            works best with f about 0.2-0.4
        @param fLeUpper/fLeLower - control how much the control points at the LE are moved
            away from the interpolation point at that location (0-on the interpolation point,
            1 - at the intersection of straight lines bounding the LE arc); works best with
            about 0.9 but needs pts 1 and 2 in the data arrays to be close to the LE
        @param density - number of points on the spline
    """
    M=3
    
    # point on the thickness line
    xyMid = (dataUpper[1,:]+dataLower[1,:])/2.
    
    # unit vector perpendicular to the thickness line at the LE
    n = xyMid - dataUpper[0,:]
    n /= np.sqrt(np.sum(n**2.))
    n = np.array([-n[1],n[0]])
    
    # upper
    # line through points 1 and 2
    a = dataUpper[2,:]-dataUpper[1,:]
    a = a[1]/a[0]
    b = dataUpper[1,1] - a*dataUpper[1,0]
    
    # where the perpendicular vector and guide line meet
    xLeArcUpper = b/(n[1]/n[0]-a)
    yLeArcUpper = a*xLeArcUpper + b
    
    # lower
    a = dataLower[2,:]-dataLower[1,:]
    a = a[1]/a[0]
    b = dataLower[1,1] - a*dataLower[1,0]
    xLeArcLower = b/(n[1]/n[0]-a)
    yLeArcLower = a*xLeArcLower + b
    
    # add the zeroth point and arc control point at the LE
    ptsUpper = np.array([dataUpper[0,:], dataUpper[0,:] + fLeUpper*([xLeArcUpper,yLeArcUpper]-dataUpper[0,:])])
    ptsLower = np.array([dataLower[0,:], dataLower[0,:] + fLeLower*([xLeArcLower,yLeArcLower]-dataLower[0,:])])
    
    # make the LE part tangent to the line defined by points 1 and 2 (like an arc)
    s1 = dataUpper[2,:]-dataUpper[1,:]
    for i in range(1,dataUpper.shape[0]-1):
        s0 = s1
        s1 = dataUpper[i+1,:]-dataUpper[i,:]
        ptsUpper = np.vstack([ptsUpper, dataUpper[i,:] - f*(s0+s1)/2., dataUpper[i,:], dataUpper[i,:] + f*(s0+s1)/2.])
    ptsUpper = np.vstack([ptsUpper, dataUpper[-1,:] - f*s1, dataUpper[-1,:]])
    
    s1 = dataLower[2,:]-dataLower[1,:]
    for i in range(1,dataLower.shape[0]-1):
        s0 = s1
        s1 = dataLower[i+1,:]-dataLower[i,:]
        ptsLower = np.vstack([ptsLower, dataLower[i,:] - f*(s0+s1)/2., dataLower[i,:], dataLower[i,:] + f*(s0+s1)/2.])
    ptsLower = np.vstack([ptsLower, dataLower[-1,:] - f*s1, dataLower[-1,:]])
    
    # fit splines
    xyFitUpper = spline(ptsUpper,M=M,density=density,interpolatePoints=True)
    xyFitLower = spline(ptsLower,M=M,density=density,interpolatePoints=True)
    
    return xyFitUpper,xyFitLower

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    #-----------------
    # test blending polynomials
    M = 3
    #T = np.array([0,0,0,1,2,2,2,3,4,4,4])
    #T = np.array([0,1,2,3,4,5,6,7,8])
    #T = np.array([0,1,2,3,4,6,7,8,9])
    T = np.array([ 0. ,  0. ,  0. ,  0. ,  0.5,  0.5,  0.5,  1. ,  1. ,  1. ,  1. ])
    
    plt.figure()
    t,B = blendingPolys(T,M,101)
    for i in range(B.shape[0]):
        plt.plot(t,B[i,:])
    plt.plot(t,np.sum(B,axis=0))
    
    #-----------------
    # test spline
    M = 3
    pts = np.array([
    [3,3,3],
    [3.5,3.25,3],
    [2.5,3.75,3],
    [3,4,3],
    [3.5,4.25,3],
    [4.5,4.25,3],
    [5.0,4,3]
#           [ 2.        ,  1.        ,  1.        ],
#           [ 2.        ,  1.33333333,  1.        ],
#           [ 1.83333333,  1.83333333,  1.        ],
#           [ 2.        ,  2.        ,  1.        ],
#           [ 2.16666667,  2.16666667,  1.        ],
#           [ 2.66666667,  1.8        ,  1.        ],
#           [ 3.        ,  2.        ,  1.        ],
#           [3.333333,2.2,1.],
#           [3.6667, 3.66667,1.],
#           [4.0,4.0,1]
           ])
    
    xyz,B,T,t = spline(pts,M,density=101,interpolateEnds=False,interpolatePoints=True,retAll=True)
    W = np.ones(pts.shape[0])
    
    plt.figure()
    plt.plot(pts[:,0],pts[:,1],'m--.')
    plt.plot(xyz[:,0],xyz[:,1],'k-')
    
    #  test IGES output
    """
    import sys
    sys.path.append("/home/artur/Dropbox/Python/CADlibs")
    import igesToolbox
    
#    splines = [
#        igesToolbox.bSplineEntity(3,pts+[0,0,0],T/T[-1],W),
#        igesToolbox.bSplineEntity(3,pts+[0,0,1],T/T[-1],W),
#        igesToolbox.bSplineEntity(3,pts+[0,0,2],T/T[-1],W),
#        ]
    splines = []
    for i in range(3):
        splines.append( igesToolbox.bSplineEntity(3,pts+[0,0,i],T/T[-1],W) )
    igesFile = igesToolbox.igesObject(splines)
            
    with open('/home/artur/Dropbox/AUVglider/Python/testCurve.iges',"w") as f:
        f.write(str(igesFile))
    """
    #---------------
    # test fitting to data
    
    data = np.array([[  1.00000000e+00,   0.00000000e+00],
       [  9.50530000e-01,   1.19600000e-02],
       [  9.01040000e-01,   2.51900000e-02],
       [  8.51390000e-01,   3.87200000e-02],
       [  8.01590000e-01,   5.18700000e-02],
       [  7.51620000e-01,   6.41900000e-02],
       [  7.01500000e-01,   7.51800000e-02],
       [  6.51260000e-01,   8.43100000e-02],
       [  6.00900000e-01,   9.10000000e-02],
       [  5.50460000e-01,   9.47300000e-02],
       [  5.00000000e-01,   9.65600000e-02],
       [  4.49520000e-01,   9.68500000e-02],
       [  3.99040000e-01,   9.57100000e-02],
       [  3.48570000e-01,   9.30900000e-02],
       [  2.98120000e-01,   8.89700000e-02],
       [  2.47710000e-01,   8.32900000e-02],
       [  1.97360000e-01,   7.58100000e-02],
       [  1.47090000e-01,   6.62400000e-02],
       [  9.69600000e-02,   5.38100000e-02],
       [  7.19900000e-02,   4.61700000e-02],
       [  4.71100000e-02,   3.71800000e-02],
       [  2.24100000e-02,   2.59200000e-02],
       [  1.01900000e-02,   1.87300000e-02],
       [  5.44000000e-03,   1.46700000e-02],
       [  3.14000000e-03,   1.20600000e-02],
       [  0.00000000e+00,   0.00000000e+00],
       [  6.86000000e-03,  -1.00600000e-02],
       [  9.56000000e-03,  -1.18700000e-02],
       [  1.48100000e-02,  -1.44500000e-02],
       [  2.75900000e-02,  -1.84800000e-02],
       [  5.28900000e-02,  -2.45400000e-02],
       [  7.80100000e-02,  -2.92100000e-02],
       [  1.03040000e-01,  -3.31300000e-02],
       [  1.52910000e-01,  -3.93200000e-02],
       [  2.02640000e-01,  -4.39700000e-02],
       [  2.52290000e-01,  -4.74900000e-02],
       [  3.01880000e-01,  -5.00900000e-02],
       [  3.51430000e-01,  -5.18900000e-02],
       [  4.00960000e-01,  -5.28700000e-02],
       [  4.50480000e-01,  -5.30500000e-02],
       [  5.00000000e-01,  -5.24400000e-02],
       [  5.49540000e-01,  -5.09300000e-02],
       [  5.99100000e-01,  -4.81600000e-02],
       [  6.48740000e-01,  -4.31100000e-02],
       [  6.98500000e-01,  -3.63000000e-02],
       [  7.48380000e-01,  -2.83900000e-02],
       [  7.98410000e-01,  -2.00300000e-02],
       [  8.48610000e-01,  -1.18000000e-02],
       [  8.98960000e-01,  -4.51000000e-03],
       [  9.49470000e-01,   6.80000000e-04],
       [  1.00000000e+00,   0.00000000e+00]])
    
    #-----------------
    # run the spline through the points
    xyFit = fitSplineThroughPoints(data,f=0.3,density=1001)
    plt.figure()
    plt.plot(data[:,0],data[:,1],'k.')
    plt.plot(xyFit[:,0],xyFit[:,1],'k')

    #-----------------
    # test airfoil interpolation - fixed LE radius issues
    dataUpper = np.flipud(data[:26,:])
    dataLower = data[25:,:]
    xyFitUpper,xyFitLower = fitSplineThroughAirfoil(dataUpper,dataLower)
    plt.figure()
    plt.plot(data[:,0],data[:,1],'k.')
    plt.plot(xyFitUpper[:,0],xyFitUpper[:,1],'k')
    plt.plot(xyFitLower[:,0],xyFitLower[:,1],'k')

    #-----------------
    plt.show()