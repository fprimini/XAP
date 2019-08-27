#! /usr/bin/env python

# 06/14/2012  Updated to run under CIAO 4.4. Scipy and Pyfits dependencies removed.
# 06/19/2012  Updated to use akima interpolation provided by Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>
# 09/24/2012  Updated to remove _akima c library in akima interpolation routine.
# 06/13/2014  Updated to allow separate Gamma Distribution alpha/beta parameters via stacks for each source
#             and background
# 06/13/2014  Updated to allow command line parameter input
# 08/23/2019  Updated to python 3
# 08/26/2019  Clean up and rename to xap_mcmc.py. Incorporate relevant functions from xap_funs.py and xap_mcmc_funs.py and
#             deprecate those files (at least for this code).

import sys

def info(type, value, tb):
   if hasattr(sys, 'ps1') or not sys.stderr.isatty():
      # we are in interactive mode or we don't have a tty-like
      # device, so we call the default hook
      sys.__excepthook__(type, value, tb)
   else:
      import traceback, pdb
      # we are NOT in interactive mode, print the exception...
      traceback.print_exception(type, value, tb)
      print
      # ...then start the debugger in post-mortem mode.
      pdb.pm()

sys.excepthook = info

from pycrates import *
from numpy import *
from paramio import *
from region import *
from crates_contrib.utils import *
import math as m

#############################################################################

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

    # First determine sizes of F and C:

    ldim=len(sregs)
    ndim=ldim+1

    C=zeros(ndim)

    # If no psfs are provided, assume source ecf=1

    F=identity(ndim)

    # Now build C. First the source regions:

    for i in arange(0,ldim):
        evtfilter='{}[sky=region({})]'.format(evtfile,sregs[i])
        evts=read_file(evtfilter)
        crtype=get_crate_type(evts)
        if crtype == 'Table':
            C[i]=len(copy_colvals(evts,0)) # assuming event list has at least 1 column
        if crtype == 'Image':
            C[i]=sum(copy_piximgvals(evts))
        evts = None
        
    # and now the background region:

    evtfilter='{}[sky=region({})]'.format(evtfile,breg)
    evts=read_file(evtfilter)
    crtype=get_crate_type(evts)
    if crtype == 'Table':
        C[ldim]=len(copy_colvals(evts,0)) # assuming event list has at least 1 column
    if crtype == 'Image':
        C[ldim]=sum(copy_piximgvals(evts))
    evts = None

    # Next, build F. If psfs are specified, use them to generate the ecf's

    if len(psfs)>0 :
        
        # All but the last row and all but the last column of F contain 
        # the ecf's of source j in region i:

        for i in arange(0,ldim):                                         # row loop
            for j in arange(0,ldim):                                     # column loop
                imgfilter='{}[sky=region({})]'.format(psfs[j],sregs[i])
                imgcr=read_file(imgfilter)
                F[i,j]=sum(copy_piximgvals(imgcr))
                imgcr = None
                
        # All but the last column of the last row of F contain the ecf's of 
        # source j in the background region:
                
        for j in arange(0,ldim):
            imgfilter='{}[sky=region({})]'.format(psfs[j],breg)
            imgcr=read_file(imgfilter)
            F[ldim,j]=sum(copy_piximgvals(imgcr))
            imgcr = None

    # The last column in F contains region areas. All but the last are source regions:

    for i in arange(0,ldim):
        F[i,ldim]=regArea(regParse('region({})'.format(sregs[i])))

    # And the last row, last column entry is the background region area.

    F[ldim,ldim]=regArea(regParse('region({})'.format(breg)))

    # Finally, modify by exposure. If exps are specified, compute average map value in
    # each region:

    ereg = ones(ndim)
    if len(exps) > 0 :

        # average expmap in each source region

        for i in arange(0,ldim):
            imgfilter = '{}[sky=region({})]'.format(exps[i],sregs[i])
            imgcr     = read_file(imgfilter)
            evals     = copy_piximgvals(imgcr)
            enums     = evals.copy()
            enums[enums>0.0]=1.0
            ereg[i]   = sum(evals)/sum(enums)
            imgcr = None
            
        # Average expmap in background region

        imgfilter = '{}[sky=region({})]'.format(exps[ldim],breg)
        imgcr     = read_file(imgfilter)
        evals     = copy_piximgvals(imgcr)
        enums     = evals.copy()
        enums[enums>0.0]=1.0
        ereg[ldim]   = sum(evals)/sum(enums)
        imgcr = None

    # otherwise, use exposure from header for all regions

    else:
        ereg = ereg*exposure

    F = F*ereg.reshape(ndim,1)
    
    return F,C

#############################################################################

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

    import numpy.linalg as la

    # Solve equation by inverting matrix F:

    Finv = la.inv(F)

    s = dot(Finv,C)   # dot is matrix multiplation

    # To get errors, need to square Finv:

    Finv2 = Finv*Finv

    sigma_s = sqrt(dot(Finv2,C))

    return s,sigma_s


#############################################################################################

class Apertures:
    """
    Class of user-defined models to compute model counts in source or
    background apertures. Model parameters are source and background
    intensities. Data are raw counts in apertures. The vector of model
    counts is computed from the vector of intensities by application of
    the ECF/Exposure matrix F (see eq. 7 and Table 1 of Primini & Kashyap,
    2014, ApJ, 796, 24.)

    Methods:
    __init__(self,F)
      define F matrix
    __call__(self, params,iap)
      compute model counts for vector aperture for F and model intensities given
      in params array.
    Attributes:
      Print F.
    """

    def __init__(self,F):
        self.F=F

    def __call__(self,params,iap):
        F=self.F
        nelem=len(params)
        mvec=zeros(nelem)
        for i in range(nelem):
            for j in range(nelem):
                mvec[i]=mvec[i]+F[i][j]*params[j]
        return mvec

    def attributes(self):
        print ('ECF/Exposure Matrix:')
        print (self.F)
        print ('Length of F:')
        print (len(self.F))
    
###################################################################################################

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

        # return a flat prior for alpha=1, beta=0

        if(self.alpha==1.0 and self.beta==0.0):
            return ones(len(s))
        
        # Otherwise, evaluate full Gamma Distribution to avoid overflows

        return exp(self.alpha*log(self.beta)+(self.alpha-1.0)*log(s)-self.beta*s-m.lgamma(self.alpha))
    
    def attributes(self):
        print ('Gamma Prior Attribute alpha:\t{:f}'.format(self.alpha))
        print ('Gamma Prior Attribute beta:\t{:f}'.format(self.beta))

#############################################################################################


def run_mcmc(F,C,s,sigma_s,ndraws,nburn,scale):

    # Wrapper for Sherpa commands to get draws
    #
    # Inputs:
    #   F      - ECF/Exposure Matrix
    #   C      - Raw aperture counts vector
    #            Related to vector of model intensities for source
    #            and background M by matrix equation
    #            C = F x M
    #   s      - MLE intensities vector
    #   sigma_s  Errors on s
    #   ndraws - Number of draws
    #   nburn  - Number of draws to skip at start of sampling
    #   scale  - Initial scale for covariance matrix in get_draws
    #
    # Outputs:
    #   parnames,stats,accept,params - parameter names and vectors from
    #   get_draws with first nburn samples eliminated

    import sherpa.astro.ui as shp

    # Create an instance of the Apertures class and load it as the user model
    
    apertures = Apertures(F)
    shp.load_user_model(apertures,"xap")

    # Make a dummy independent variable array to go with C, and load it and C
    # as the dataset

    ix=arange(len(C))
    shp.load_arrays(1,ix,C,shp.Data1D)

    # Define parameter names. The last element in C is always the background.

    parnames=[]
    for i in range(len(C)-1):
        parnames.append('Source_{}'.format(i))
    parnames.append('Background')

    # and add user pars, using s, sigma_s to establish guesses and bounds for parameters.
    # Offset values of s by 5% to force re-fit for maximum likelihood.

    shp.add_user_pars("xap",parnames,1.05*s,parmins=1.05*s-5.0*sigma_s,parmaxs=1.05*s+5.0*sigma_s)

    # Set model, statistic, and minimization method

    shp.set_model(xap)
    shp.set_stat("cash")
    shp.set_method("moncar")

    # Finally, run fit and covar to set bounds for get_draws

    # First, set hard lower limit for source intensity to avoid negative values


    for i in range(len(xap.pars)):
        xap.pars[i]._hard_min=0.0

    import logging
    logger = logging.getLogger("sherpa")

#    logger.setLevel(logging.ERROR)

    shp.fit()
    shp.covar()


    # Check that covar returns valid matrix:

    cvmat = shp.get_covar_results().extra_output
    if any(isnan(cvmat.reshape(-1))) or any(diag(cvmat)<0.0):
        print ("WARNING: covar() returned an invalid covariance matrix. Attempting to use conf().")
        # Try to use conf() to set diagonal elements of covar matrix. If that doesn't work, use sigma_s
        shp.conf()
        conf_err=array(shp.get_conf_results().parmaxes)
        if not all(conf_err>0.0):
            print ("WARNING: conf() returned invalid errors. Using MLE errors.")
            conf_err=sigma_s
        cvmat = diag(conf_err*conf_err)

    shp.get_covar_results().extra_output=cvmat
    shp.set_sampler_opt('scale',scale)
    stats,accept,intensities=shp.get_draws(niter=ndraws)

    # and return all but the first nburn values. If nburn>=ndraws,
    # return a single draw.

    nburn1=min(nburn,ndraws-1)
    return parnames,stats[nburn1:],accept[nburn1:],intensities[:,nburn1:]

###################################################################################################

def write_draws(filename,parnames,stats,accept,intensities):

    # Use crates to save draws in a fits file

    # Inputs:
    # filename - output draws file name
    # parnames - names of source/backgrounds
    # stats    - vector of stats from get_draws
    # accept   - vector of draws acceptances
    # intensities - 2-d table of draws for each aperture
    
    tab=TABLECrate()
    tab.name="Draws"
    add_colvals(tab,'Stat',stats)
    add_colvals(tab,'Accept',accept)

    for i in range(len(parnames)):
        add_colvals(tab,parnames[i],intensities[i])
    
    ds=CrateDataset()
    ds.add_crate(tab)
    ds.write(filename,clobber=True)

    return

###################################################################################################

def write_draws_mpdfs(filename,parnames,intensities,ndrmesh):

    # Estimate mpdfs from draws, and use crates to save mpdfs in a fits file

    # Inputs:
    # filename - output draws mpdfs file name
    # parnames - names of source/backgrounds
    # intensities - 2-d table of draws for each aperture
    # ndrmesh  - number of grid points in mpdfs

    cds=CrateDataset()
    for i in range(len(parnames)):
        sdraws=intensities[i]

        # Compute Gamma Dist alpha, beta from mean and variance
        # of intensity values in draws

        smean = sum(sdraws)/len(sdraws)
        svar  = sum(sdraws*sdraws)/len(sdraws) - smean*smean
        alpha = smean*smean/svar
        beta  = smean/svar
        
        smin  = max(min(sdraws),1.0e-10)        # should never happen, but just in case, make sure sdraws>0
        smax  = max(sdraws)
        ds    = (smax-smin)/ndrmesh
        sgrid = arange(smin,smax,ds)

        # First Gamma Dist 

        gps   = Gamma_Prior(alpha,beta)
        draws_mpdf = gps(sgrid)

        tab=TABLECrate()
        tab.name=parnames[i]+"_"+"Gamma_Dist"
        add_colvals(tab,'Inten',sgrid)
        add_colvals(tab,'MargPDF',draws_mpdf)
        set_key(tab,'alpha',alpha)
        set_key(tab,'beta',beta)
        set_key(tab,'smean',smean)
        set_key(tab,'svar',svar)
        set_key(tab,'Stepsize',ds)
        cds.add_crate(tab)

    cds.write(filename,clobber=True)

    return

#############################################################################################

def usage():
    print ("Usage: xap_mcmc.py @@<CIAO-style parameter file>")
    print ("Default parameter file is xap_mcmc.par")
    return

#############################################################################################

def main(argv):

    # Get Inputs from parameter file:

    try:
        fp=paramopen("xap_mcmc.par","wL",argv)
    except:
        usage()
        sys.exit(2)

    evtfile=pget(fp,"infile")
    drawsfile=pget(fp,"drawsfile")
    drawsmpdffile=pget(fp,"drawsmpdffile")
    breg=pget(fp,"breg")
    srcstack=pget(fp,"srcstack")
    scr=read_file(srcstack)
    sregs=get_colvals(scr,0)

    # Read psfstack and test for null values

    psfs = array([])
    psfstack=pget(fp,"psfstack")
    try:
        pcr=read_file(psfstack)
        psfs=get_colvals(pcr,0)
    except:
        print( "Unable to find PSF files.\nSetting source region ecfs to 1.0\n and background ecf to 0.0\n")

    # Read expstack and test for null values

    exps = array([])
    expstack=pget(fp,"expstack")
    try:
        ecr=read_file(expstack)
        exps=get_colvals(ecr,0)
    except:
        print( "Unable to find exposure maps.\nOutput units will be \"counts/sec\".\n")

    #if len(sregs) != len(psfs):
    #    print( "Error: Source and PSF Stacks Inconsistent")

    CL_desired   = pgetd(fp,"CL")
    intenstack = pget(fp,"intenstack")
    nmesh        = pgeti(fp,"nmesh")
    ndraws       = pgeti(fp,"ndraws")
    nburn        = pgeti(fp,"nburn")
    scale        = pgetd(fp,"scale")
    ndrmesh      = pgeti(fp,"ndrmesh")
    clob         = pgetb(fp,"clobber")
    verb         = pgeti(fp,"verbose")

    paramclose(fp)

    # Get EXPOSURE from event list header

    exposure = 1.0
    try:
        exposure= get_keyval(read_file(evtfile),'exposure')
    except:
        print( "Unable to find EXPOSURE keyword in header to {}\nOutput units will be \"counts\".".format(evtfile))

    # Get MJD_OBS from event list header

    mjdobs = -1.0
    try:
        mjdobs= get_keyval(read_file(evtfile),'MJD_OBS')
    except:
        print( "Unable to find MJD_OBS keyword in header to {}\nObservation Time will be set to -1.0".format(evtfile))

    if verb>0:
        print( "Getting Events from {}\n".format(evtfile))
        print( "Exposure:\t{}".format(exposure))
        print( "Source Regions:")
        for i in arange(0,len(sregs)):
            print( sregs[i])

        print( "\nBackground Regions:")
        print( breg)
        print( "\nPSF Images:")
        
        for i in arange(0,len(psfs)):
            print( psfs[i])
        print( "\n")

    # Now set up F and C arrays:

    F,C = get_F_C_exp(evtfile,sregs,breg,psfs,exps,exposure)

    number_of_params = len(C)          # This is source plus background
    number_of_sources = number_of_params -1        # Number of sources only

    if verb>0:
        for i in arange(0,number_of_sources):
            print( "Counts in Region {}:\t\t{}".format(i,C[i]))

        print( "\nCounts in Background Region:\t{}".format(C[number_of_sources]))

        print( "\nF:")
        print( F)
        print( "\n")

    # Solve for MLE intensities and errors:

    s,sigma_s = get_s_sigma(F,C)

    # Finally, run mcmc:

    print( "Running MCMC...")

    parnames,stats,accept,intensities=run_mcmc(F,C,s,sigma_s,ndraws,nburn,scale)

    write_draws(drawsfile,parnames,stats,accept,intensities)

    # Write equivalent of mpdfs files, but using draws to estimate mpdfs

    write_draws_mpdfs(drawsmpdffile,parnames,intensities,ndrmesh)

if __name__ == "__main__":
    main(sys.argv)
