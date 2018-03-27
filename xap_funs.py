#############################################################################################
def posterior_stats(s,ps):
    """
    Inputs:
    s    array of source intensities
    ps   array of corresponding posterior probability distribution

    Output:
    mean and variance of s, defined as

    mean = Int(s*ps*ds)/Int(ps*ds)
    var  = Int(s*s*ps*ds)/Int(ps*ds) - mean*mean

    where ds is array step size (s[1]-s[0])

    mean,var = posterior_stats(s,ps)
    """

    import numpy as np

    ds = s[1]-s[0]
    smean = np.sum(s*ps*ds)/np.sum(ps*ds)
    s2    = np.sum(s*s*ps*ds)/np.sum(ps*ds)

    svar  = s2 - smean * smean
    return smean,svar

#############################################################################################

def pmf(k,mu):
    """
    Inputs:
    k    sample number
    mu   sample mean

    Returns:
    Probability of obtaining sample k from Poisson Distribution with mean mu.
    Replaces poisson.pmf from scipy.stats
    """

    from sherpa.ui import lgam
    import numpy as np

    Pk = k*np.log(mu)-lgam(k+1) - mu
    return np.exp(Pk)

#############################################################################################

def simple_int(y,x):
    """
    Simple Numerical Integration Routine
    Inputs:
    y   array of y values
    x   array of evenly-spaced x values
    Output:
    Integral of y(x)

    If array length < 2, result=0
    If array length = 2, use Basic Trapezoidal rule (Press eqn 4.1.3)
    If array length = 3, use Basic Simpson's Rule (Press eqn 4.1.4)
    If array length = 4, use Simpson's 3/8 Rule (Press eqn. 4.1.5)
    If array length = 5, use Boole's Rule (Press eqn. 4.1.6)
    If array length > 5, use Extended Simpson's Rule (Press eqn. 4.1.12)
    """

    nx = len(x)
    if nx < 2:
        return 0.0

    h = x[1]-x[0]

    if nx == 2:
        h = h/2.0
        return h*(y[0]+y[1])

    if nx == 3:
        h = h/3.0
        return h*(y[0]+4.0*y[1]+y[2])

    if nx == 4:
        h = h/8.0
        return h*(3.0*y[0]+9.0*y[1]+9.0*y[2]+3.0*y[3])

    if nx == 5:
        h = h/45.0
        return h*(14.0*(y[0]+y[4]) + 64.0*(y[1]+y[3]) +24.0*y[2])

    if nx > 5:
        # Remember the slice [n1:n2] does not include index n2
        return h*(5.0/12.0)*(y[0]+y[-1]) + h*(13.0/12.0)*(y[1]+y[-2]) + h*sum(y[2:-2])

#############################################################################################

def get_F_C_exp(evtfile,sregs,breg,psfs,exps,exposure):
    """
    Inputs:
    evtfile   Event list file name (used to determine counts in regions)
    sregs     List of source region fits file names
    breg      Background region fits file name
    psfs      List of source psf fits image file names

    Compute array of psf fractions F and vector of total counts C, such that
    F[i,j] is PSF fraction of source j in source region i
    C[i] is total counts in source region i
    Here, the last source region is actually the background region, so
    F[n,j] is the PSF fraction of source j in the background region and
    C[n] is the total counts in the background region.
    Array F and vector C are returned.

    In this version, observation exposure is accounted for, either via exposure maps
    stack exps (one per source/background region) or header keyword.

    """

    import numpy as np
    import pycrates as pc
    import region as re

    # First determine sizes of F and C:

    ldim=len(sregs)
    ndim=ldim+1

    C=np.zeros(ndim)

    # If no psfs are provided, assume source ecf=1

    F=np.identity(ndim)

    # Now build C. First the source regions:

    for i in np.arange(0,ldim):
        evtfilter="%s[sky=region(%s)]" % (evtfile,sregs[i])
        evts=pc.read_file(evtfilter)
        crtype=pc.get_crate_type(evts)
        if crtype == 'Table':
            C[i]=len(pc.get_colvals(evts,0)) # assuming event list has at least 1 column
        if crtype == 'Image':
            C[i]=np.sum(pc.copy_piximgvals(evts))
            
    # and now the background region:

    evtfilter="%s[sky=region(%s)]" % (evtfile,breg)
    evts=pc.read_file(evtfilter)
    crtype=pc.get_crate_type(evts)
    if crtype == 'Table':
        C[ldim]=len(pc.get_colvals(evts,0)) # assuming event list has at least 1 column
    if crtype == 'Image':
        C[ldim]=np.sum(pc.copy_piximgvals(evts))


    # Next, build F. If psfs are specified, use them to generate the ecf's

    if len(psfs)>0 :
        
        # All but the last row and all but the last column of F contain 
        # the ecf's of source j in region i:

        for i in np.arange(0,ldim):                                         # row loop
            for j in np.arange(0,ldim):                                     # column loop
                imgfilter="%s[sky=region(%s)]" % (psfs[j],sregs[i])
                F[i,j]=np.sum(pc.copy_piximgvals(pc.read_file(imgfilter)))
    
        # All but the last column of the last row of F contain the ecf's of 
        # source j in the background region:
                
        for j in np.arange(0,ldim):
            imgfilter="%s[sky=region(%s)]" % (psfs[j],breg)
            F[ldim,j]=np.sum(pc.copy_piximgvals(pc.read_file(imgfilter)))

    

    # The last column in F contains region areas. All but the last are source regions:

    for i in np.arange(0,ldim):
        F[i,ldim]=re.regArea(re.regParse("region(%s)" % sregs[i]))

    # And the last row, last column entry is the background region area.

    F[ldim,ldim]=re.regArea(re.regParse("region(%s)" % breg))

    # Finally, modify by exposure. If exps are specified, compute average map value in
    # each region:

    ereg = np.ones(ndim)
    if len(exps) > 0 :

        # average expmap in each source region

        for i in np.arange(0,ldim):
            imgfilter = "%s[sky=region(%s)]" % (exps[i],sregs[i])
            evals     = pc.copy_piximgvals(pc.read_file(imgfilter))
            enums     = evals.copy()
            enums[enums>0.0]=1.0
            ereg[i]   = np.sum(evals)/np.sum(enums)

        # Average expmap in background region

        imgfilter = "%s[sky=region(%s)]" % (exps[ldim],breg)
        evals     = pc.copy_piximgvals(pc.read_file(imgfilter))
        enums     = evals.copy()
        enums[enums>0.0]=1.0
        ereg[ldim]   = np.sum(evals)/np.sum(enums)


    # otherwise, use exposure from header for all regions

    else:
        ereg = ereg*exposure

    F = F*ereg.reshape(ndim,1)
    
    return F,C

#############################################################################################

def get_F_C_img(imgfile,sregs,breg,psfs):
    """
    Inputs:
    imgfile   Image file name (used to determine counts in regions)
    sregs     List of source region fits file names
    breg      Background region fits file name
    psfs      List of source psf fits image file names

    Compute array of psf fractions F and vector of total counts C, such that
    F[i,j] is PSF fraction of source j in source region i
    C[i] is total counts in source region i
    Here, the last source region is actually the background region, so
    F[n,j] is the PSF fraction of source j in the background region and
    C[n] is the total counts in the background region.
    Array F and vector C are returned.

    Just like get_F_C, but just uses image not event list to get counts.
    
    """

    import numpy as np
    import pycrates as pc
    import region as re

    # First determine sizes of F and C:

    ldim=len(sregs)
    ndim=ldim+1

    C=np.zeros(ndim)
    F=np.zeros([ndim,ndim])

    # Now build C. First the source regions:

    for i in np.arange(0,ldim):
        imgfilter="%s[sky=region(%s)]" % (imgfile,sregs[i])
        cr=pc.read_file(imgfilter)
        C[i]=np.sum(pc.copy_piximgvals(cr))
#        cr.get_dataset().snip()

    # and now the background region:

    imgfilter="%s[sky=region(%s)]" % (imgfile,breg)
    cr=pc.read_file(imgfilter)
    C[ldim]=np.sum(pc.copy_piximgvals(cr))
#    cr.get_dataset().snip()

    # Next, build F. All but the last row and all but the last column of F contain the ecf's of source j in region i:

    for i in np.arange(0,ldim):                                         # row loop
        for j in np.arange(0,ldim):                                     # column loop
            imgfilter="%s[sky=region(%s)]" % (psfs[j],sregs[i])
            cr=pc.read_file(imgfilter)
            F[i,j]=np.sum(pc.copy_piximgvals(cr))
#            cr.get_dataset().snip()
    
    # All but the last column of the last row of F contain the ecf's of source j in the background region:

    for j in np.arange(0,ldim):
        imgfilter="%s[sky=region(%s)]" % (psfs[j],breg)
        cr=pc.read_file(imgfilter)
        F[ldim,j]=np.sum(pc.copy_piximgvals(cr))
#        cr.get_dataset().snip()

    # The last column in F contains region areas. All but the last are source regions:

    for i in np.arange(0,ldim):
        F[i,ldim]=re.regArea(re.regParse("region(%s)" % sregs[i]))

    # And the last row, last column entry is the background region area.

    F[ldim,ldim]=re.regArea(re.regParse("region(%s)" % breg))

    return F,C

#############################################################################################

def get_s_sigma(F,C):
    """
    Solve matrix equation C = F dot s for source intensity vector s.

    Inputs:
    F[i,j]     Array of encircled energy fractions and region areas
    C[i]       Vector of region counts

    Output:
    s[i]       Vector of MLE estimates of source and background intensities
    sigma_s[i] Vector of errors on MLE estimates, assuming Gaussian statistics

    1/30/2014
    Change sigma calculation to use covariance matrix method

    5/5/2014
    Go back to old propagation of errors technique to calculate sigmas
    """

    import numpy as np
    import numpy.linalg as la

    # Solve equation by inverting matrix F:

    Finv = la.inv(F)

    s = np.dot(Finv,C)   # dot is matrix multiplation

    # To get errors, need to square Finv:

    Finv2 = Finv*Finv

    sigma_s = np.sqrt(np.dot(Finv2,C))

    # Follow Equations 5 & 8 in paper

#    sigma_s=np.zeros(len(s))
#    for i in range(len(sigma_s)):
#        for j in range(len(sigma_s)):
#            sigma_s[i]+=F[j,i]*F[j,i]/C[j]

#    sigma_s = np.sqrt(1.0/sigma_s)

    return s,sigma_s


#############################################################################################

def marginalize_posterior(joint_posterior,source_number,stepsize):
    """
    Marginalize Joint Posterior Distribution by summing joint posterior hypercube over
    all axes except that for the specified source.

    Input:

    joint_posterior     N-dimensional hypercube of unnormalized joint posterior distribution
                        evaluated on a mesh of source and background intensities.
    source_number       Index of source of interest
    stepsize            Step size of mesh for this source.

    Returns 1-dimensional vector of marginalized pdf for this source, normalized such that 
    sum(marginalized pdf)*stepsize = 1.0
    """

    import numpy as np

    # Roll axes in input array until source of interest in slowest-moving (axis 0)

    mpost = np.rollaxis(joint_posterior,source_number,0)

    nparam=mpost.ndim

    i=1
    while i<nparam:
        mpost=np.add.reduce(mpost,-1)        # Reduce by last axis in rolled hypercube
        i+=1

    return mpost/(sum(mpost)*stepsize)

#############################################################################################

def pdf_summary(s,spdf,CL):
    """
    Compute summary statistics for probability distribution spdf, measured at points s.

    Input:

    s           array of evenly spaced data points
    spdf        probability distribution values at s
    CL          desired confidence level enclosed by reported percentiles
    
    Output:

    smode       value of s corresponding to mode of spdf distribution
    slo         lower percentile of CL
    shi         upper percentile of CL
    zmode       Boolean true if mode max(spdf) is spdf[0]
    zCL         Boolean true if lower bound of spdf encountered before CL achieved

    The maximum value of spdf is determined, and points one either side are used to approximate
    a quadratic. The coefficients of the quadratic are used to compute the value of s corresponding
    to the peak of spdf (which may noot be a sampled value). This value of s is reported as the mode.
    If the maximum value of spdf is spdf[0], s[0] is reported as the mode and the Boolean zmode is 
    set to True.

    spdf is integrated between points on either side of spdf[max], and the percentiles are extended 
    alternately above and below spdf[max] until the integral exceeds the desired CL. If spdf[0] is 
    encountered before CL is achieved, the process continues with extension of the upper percentile 
    only until the desired CL is achieved, and the Boolean zCL is set to True.
    """

    import numpy as np
    import numpy.linalg as la

    #    from scipy.integrate import simps

    # Try to interpolate sampled values on a finer grid

    try:
        peakind=np.where(spdf==max(spdf))
        speak=s[peakind[0][0]]
        dels=s[-1]-speak
        smin=max(0.0,s[0]-dels)
        smax=s[-1]+dels
        sint=np.arange(smin,smax,dels/100.0)
        pint=np.exp(interpolate(s,np.log(spdf),sint))
    except:
        sint=s
        pint=spdf

    zmode = False
    zCL   = False

    i0 = np.where(pint==max(pint))[0][0]
    if i0==0:
        src_mode = sint[i0]
        zmode = True

    im1 = i0 - 1
    if im1 <=0 :
        zCL = True
        im1=0

    ip1 = i0 + 1

    # If there are 3 independent points, determine coefficients of quadratic that passes through im1, i0, ip1
    # and use these to get a better estimate for the mode (vertex of quadratic)

    if (ip1-im1)==2:
        aa = np.array([[sint[im1]**2,sint[im1],1.0],[sint[i0]**2,sint[i0],1.0],[sint[ip1]**2,sint[ip1],1.0]])
        bb = pint[im1:ip1+1]
        abc= np.dot(la.inv(aa),bb)

        # This is the x coordinate corresponding to the vertex of the quadratic:

        if (abc[0] != 0.0):
            src_mode = -abc[1]/(2.0*abc[0])
        
    # Now start integrating, out from central 3 points, util integral exceeds CL
    
    i_plus = i0
    i_minus = i0
    CLreal = 0.0

    while CLreal<CL and i_minus>0 and i_plus<(len(pint)-1):
    
        p_plus = pint[i_plus+1]
        p_minus = pint[i_minus-1]
    
        if(p_minus>p_plus):
            i_minus -= 1
            #            CLreal = simps(spdf[i_minus:i_plus+1],s[i_minus:i_plus+1],even='first')
            CLreal = simple_int(pint[i_minus:i_plus+1],sint[i_minus:i_plus+1])
            
        else:
            i_plus += 1
            #            CLreal = simps(spdf[i_minus:i_plus+1],s[i_minus:i_plus+1],even='first')
            CLreal = simple_int(pint[i_minus:i_plus+1],sint[i_minus:i_plus+1])


    if((CLreal<CL) and (i_minus==0)):
        zCL = True
        while((CLreal<CL) and (i_plus<(len(pint)-1))):
            i_plus += 1
            #            CLreal = simps(spdf[i_minus:i_plus+1],s[i_minus:i_plus+1],even='first')
            CLreal = simple_int(pint[i_minus:i_plus+1],sint[i_minus:i_plus+1])

    return src_mode,sint[i_minus],sint[i_plus],CLreal,zmode,zCL

#############################################################################################

class Intensity_Range:
    """
    Compute grid points for given intensity and sigma

    Methods:
       __init__(self,nsig,minlimit,nvals)
          set nsig    = number of sigma for half-size
              minlimit= lowest allowed value of intensity
              nvals   = number of grid points
       __call__(self,inten,sig_inten)
          compute nvals points at inten +/- nsig*inten_sig, but not below minlimit
          returns array of grid values and step size
    Attributes:
       nsig    = number of sigma for half-size
       minlimit= lowest allowed value of intensity
       nvals   = number of grid points
    """

    def __init__(self,nsig=5.0,minlimit=1.0e-10,nvals=50):
        self.nsig=float(nsig)
        self.minlimit=float(minlimit)
        self.nvals=float(nvals)

    def __call__(self,inten,inten_sig):
        import numpy as np
        nsig,minlimit,nvals=self.nsig,self.minlimit,self.nvals
        xmin=max(inten - nsig*inten_sig,minlimit)
        # replace statement below with alternate calculation 2013-03-06
        # xmax=inten + nsig*inten_sig
        xmax = xmin + 2.0 * nsig * inten_sig
        dx=(xmax-xmin)/nvals
        tmpgrid=np.arange(xmin,xmax,dx)
        if(len(tmpgrid)>nvals):
            tmpgrid=tmpgrid[0:int(nvals)]
        return tmpgrid,dx

    def attributes(self):
        print 'Intensity Grid Attributes:'
        print 'Grid Half-size number of MLE sigma:\t%f' % self.nsig
        print 'Minimum allowable grid value:\t\t%e' % self.minlimit
        print 'Number of grid points (per dimension):\t%.0f\n' % self.nvals

#############################################################################################

class Gamma_Prior:
    """
    Compute Gamma Prior Distribution intensity vector, using alpha, beta specified in attributes
    The prior distribution unnormalized and is defined as gp(s) = s**(alpha-1)*exp(-beta*s)
 
    Methods:

    __init__(self,alpha,beta)
        initialize alpha, beta

    __call__(self,s)
        returns array of gp for input array of s values

    __attributes(self)
        print alpha and beta
    """

    def __init__(self,alpha,beta):
        self.alpha = float(alpha)
        self.beta  = float(beta)

    def __call__(self,s):
        import numpy as np
        import math as m

        # return a flat prior for alpha=1, beta=0

        if(self.alpha==1.0 and self.beta==0.0):
            return np.ones(len(s))
        
        # Otherwise, evaluate full Gamma Distribution to avoid overflows

        return np.exp(self.alpha*np.log(self.beta)+(self.alpha-1.0)*np.log(s)-self.beta*s-m.lgamma(self.alpha))
    
    def attributes(self):
        print 'Gamma Prior Attribute alpha:\t%f' % self.alpha
        print 'Gamma Prior Attribute beta:\t%f' % self.beta

#############################################################################################

# akima.py

# Copyright (c) 2007-2012, Christoph Gohlke
# Copyright (c) 2007-2012, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Interpolation of data points in a plane based on Akima's method.

Akima's interpolation method uses a continuously differentiable sub-spline
built from piecewise cubic polynomials. The resultant curve passes through
the given data points and will appear smooth and natural.

:Authors:
  `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`__,
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 2012.09.24 FAP revisions to eliminate c extension module and run in xap.py

Requirements
------------

* `Python 2.7 or 3.2 <http://www.python.org>`__
* `Numpy 1.6 <http://numpy.scipy.org>`__

References
----------

(1) A new method of interpolation and smooth curve fitting based
    on local procedures. Hiroshi Akima, J. ACM, October 1970, 17(4), 589-602.

Examples
--------

>>> def example():
...     '''Plot interpolated Gaussian noise.'''
...     x = numpy.sort(numpy.random.random(10) * 100)
...     y = numpy.random.normal(0.0, 0.1, size=len(x))
...     x2 = numpy.arange(x[0], x[-1], 0.05)
...     y2 = interpolate(x, y, x2)
...     from matplotlib import pyplot
...     pyplot.title("Akima interpolation of Gaussian noise")
...     pyplot.plot(x2, y2, "b-")
...     pyplot.plot(x, y, "ro")
...     pyplot.show()
>>> example()

"""

def interpolate(x, y, x_new, axis=-1, out=None):
    """
    Return interpolated data using Akima's method.
    
    This Python implementation is inspired by the Matlab(r) code by
    N. Shamsundar. It lacks certain capabilities of the C implementation
    such as the output array argument and interpolation along an axis of a
    multidimensional data array.
    
    Parameters
    ----------
    
    x : array like
    1D array of monotonically increasing real values.
    
    y : array like
    N-D array of real values. y's length along the interpolation
    axis must be equal to the length of x.
    
    x_new : array like
    New independent variables.
    
    axis : int
    Specifies axis of y along which to interpolate. Interpolation
    defaults to last axis of y.
    
    out : array
    Optional array to receive results. Dimension at axis must equal
    length of x.
    
    Examples
    --------
    
    >>> interpolate([0, 1, 2], [0, 0, 1], [0.5, 1.5])
    array([-0.125,  0.375])
    >>> x = numpy.sort(numpy.random.random(10) * 10)
    >>> y = numpy.random.normal(0.0, 0.1, size=len(x))
    >>> z = interpolate(x, y, x)
    >>> numpy.allclose(y, z)
    True
    >>> x = x[:10]
    >>> y = numpy.reshape(y, (10, -1))
    >>> z = numpy.reshape(y, (10, -1))
    >>> interpolate(x, y, x, axis=0, out=z)
    >>> numpy.allclose(y, z)
    True
    
    """
    
    import numpy
    
    x = numpy.array(x, dtype=numpy.float64, copy=True)
    y = numpy.array(y, dtype=numpy.float64, copy=True)
    xi = numpy.array(x_new, dtype=numpy.float64, copy=True)
    
    if axis != -1 or out is not None or y.ndim != 1:
        raise NotImplementedError("implemented in C extension module")
    
    if x.ndim != 1 or xi.ndim != 1:
        raise ValueError("x-arrays must be one dimensional")
    
    n = len(x)
    if n < 3:
        raise ValueError("array too small")
    if n != y.shape[axis]:
        raise ValueError("size of x-array must match data shape")
    
    dx = numpy.diff(x)
    if any(dx <= 0.0):
        raise ValueError("x-axis not valid")
    
    #   Remove extrapolation error test
    
    #    if any(xi < x[0]) or any(xi > x[-1]):
    #        raise ValueError("interpolation x-axis out of bounds")
    
    m = numpy.diff(y) / dx
    mm = 2.0 * m[0] - m[1]
    mmm = 2.0 * mm - m[0]
    mp = 2.0 * m[n - 2] - m[n - 3]
    mpp = 2.0 * mp - m[n - 2]
    
    m1 = numpy.concatenate(([mmm], [mm], m, [mp], [mpp]))
    
    dm = numpy.abs(numpy.diff(m1))
    f1 = dm[2:n + 2]
    f2 = dm[0:n]
    f12 = f1 + f2
    
    ids = numpy.nonzero(f12 > 1e-12 * numpy.max(f12))[0]
    b = m1[1:n + 1]
    
    b[ids] = (f1[ids] * m1[ids + 1] + f2[ids] * m1[ids + 2]) / f12[ids]
    c = (3.0 * m - 2.0 * b[0:n - 1] - b[1:n]) / dx
    d = (b[0:n - 1] + b[1:n] - 2.0 * m) / dx ** 2
    
    bins = numpy.digitize(xi, x)
    bins = numpy.minimum(bins, n - 1) - 1
    
    #
    #   Try this to fix bad interpolations below first point
    #
    
    bins = numpy.maximum(bins,0)
    
    bb = bins[0:len(xi)]
    wj = xi - x[bb]
    
    return ((wj * d[bb] + c[bb]) * wj + b[bb]) * wj + y[bb]

#############################################################################################
def get_Fitness(pdfs,nsteps=1000):
    """
    Determine cumulative fitness functions for a range of pdfs, input as a list with

    pdfs[i][0] = start time of cell
    pdfs[i][1] = array of intensity values s at which pdf is evaluated
    pdfs[i][2] = array of pdf values for that cell

    The pdf is normalized such that sum(pdf)*(s[1]-s[0]) = 1

    """

    import numpy as np

    mins=[]
    maxes=[]
    npdfs = len(pdfs)
    F = np.zeros(npdfs)

    # First, find range of all the s arrays

    for i in range(0,len(pdfs)):
        mins.append(pdfs[i][1][0])
        maxes.append(pdfs[i][1][-1])

    # and use that to define new intensity grid s


    s0 = np.min(mins)
    s1 = np.max(maxes)
    ds = (s1-s0)/nsteps
    sint = np.arange(s0,s1,ds)

    # Build fint, the integrand of F, which is the product of the regridded pdfs. Start with fint set to 1
    # and work backwards, so that F includes the last pdf only the first time through the loop, then the 
    # last two, etc. until it includes them all.

    fint = np.ones(len(sint))

    for i in range(0,npdfs):
        j = npdfs - i - 1
        pint=np.exp(interpolate(pdfs[j][1],np.log(pdfs[j][2]),sint))
        pint /= (sum(pint)*ds)
        fint *= pint
        F[j] = sum(fint) * ds

    return np.log10(F)

#############################################################################################
def get_blocks(pdf_list,ncprior):
    """
    Determine change-points for Bayesian Blocks
    Input:
    pdf_list: List of pdf data for each cell, [start time of cell, array of intensity bins, array of pdfs]
    ncprior:  Penalty factor. log10 of prior probability of having N blocks. Input ncprior is assumed >0
              and is subtracted from the fitness function for each block. 
    Output:
    change_points: list of start times of cells that begin new blocks in optimum partition
    """

    import numpy as np

    # Make sure pdf list is time-sorted

    pdf_list.sort()

    ncells = len(pdf_list)

    # The optimal partition for the starting case of the first cell only has a best fitness function of 
    # -ncprior, since the marginalized likelihood is 1 for  single normalized pdf. The location of the 
    # first change-point is the beginning of the list, or index 0

    best = np.array(-ncprior)
    last = np.array(0)

    # Now need to construct A(r)

    for R in range(1,ncells):        # Skip the first cell since we already know the results for it
        F = get_Fitness(pdf_list[0:R+1])
        A = np.append(0,best) + F - ncprior
        best = np.append(best,A.max())
        last = np.append(last,A.argmax())
        #print "R = %d" % R
        #print "F: "+" ".join(' %5.3g' % v for v in F)
        #print "A: "+" ".join(' %5.3g' % v for v in A)
        #print "best: "+" ".join(' %5.3g' % v for v in best)
        #print "last: "+" ".join(' %3d' % v for v in last)

    # Once all ncells have been considered, reconstruct change-points from 'last' array:

    change_points = []
    cpindex = last[-1]

    while cpindex > 0 :
        change_points.insert(0,cpindex)
        cpindex = last[cpindex-1]

    # above gets everything except the first one

    change_points.insert(0,last[0])

    return change_points


            

