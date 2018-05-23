#   Functions to support xap_mcmc.py
#
#   Apertures:
#       Sherpa User Model to compute source and background intensities from
#       counts in source and background apertures.
#
#   run_mcmc:
#
#       Wrapper function for Sherpa fit and MCMC routines.
#
#   write_draws:
#
#       Function to output draws fits file.
#
#   write_draws_mpdfs:
#
#       Estimate mpdfs from draws, and use crates to save mpdfs in a fits file

###################################################################################################

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
        import numpy as np
        F=self.F
        nelem=len(params)
        mvec=np.zeros(nelem)
        for i in range(nelem):
            for j in range(nelem):
                mvec[i]=mvec[i]+F[i][j]*params[j]
        return mvec

    def attributes(self):
        print 'ECF/Exposure Matrix:'
        print self.F
        print 'Length of F:'
        print len(self.F)
    
###################################################################################################

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
    import numpy as np

    # Create an instance of the Apertures class and load it as the user model
    
    apertures = Apertures(F)
    shp.load_user_model(apertures,"xap")

    # Make a dummy independent variable array to go with C, and load it and C
    # as the dataset

    ix=np.arange(len(C))
    shp.load_arrays(1,ix,C,shp.Data1D)

    # Define parameter names. The last element in C is always the background.

    parnames=[]
    for i in range(len(C)-1):
        parnames.append('Source_%d' % i)
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
    if np.any(np.isnan(cvmat.reshape(-1))) or np.any(np.diag(cvmat)<0.0):
        print "WARNING: covar() returned an invalid covariance matrix. Attempting to use conf()."
        # Try to use conf() to set diagonal elements of covar matrix. If that doesn't work, use sigma_s
        shp.conf()
        conf_err=np.array(shp.get_conf_results().parmaxes)
        if not np.all(conf_err>0.0):
            print "WARNING: conf() returned invalid errors. Using MLE errors."
            conf_err=sigma_s
        cvmat = np.diag(conf_err*conf_err)

    shp.get_covar_results().extra_output=cvmat
#    initscale = 0.1
#    import pdb;pdb.set_trace()
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
    
    import pycrates as pc
    import crates_contrib.utils as pcu
    
    tab=pc.TABLECrate()
    tab.name="Draws"
    pcu.add_colvals(tab,'Stat',stats)
    pcu.add_colvals(tab,'Accept',accept)

    for i in range(len(parnames)):
        pcu.add_colvals(tab,parnames[i],intensities[i])
    
    ds=pc.CrateDataset()
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

    import numpy as np
    import pycrates as pc
    import crates_contrib.utils as pcu
    import xap_funs as xfuns
    import sys
    sys.path.append("/Users/fap/anaconda/lib/python2.7/site-packages/")
    from sklearn.neighbors import KernelDensity
    
    cds=pc.CrateDataset()
    for i in range(len(parnames)):
        sdraws=intensities[i]

        # Compute Gamma Dist alpha, beta from mean and variance
        # of intensity values in draws

        smean = np.sum(sdraws)/len(sdraws)
        svar  = np.sum(sdraws*sdraws)/len(sdraws) - smean*smean
        alpha = smean*smean/svar
        beta  = smean/svar
        
        smin  = max(min(sdraws),1.0e-10)        # should never happen, but just in case, make sure sdraws>0
        smax  = max(sdraws)
        ds    = (smax-smin)/ndrmesh
        sgrid = np.arange(smin,smax,ds)

        # First Gamma Dist 

        gps   = xfuns.Gamma_Prior(alpha,beta)
        draws_mpdf = gps(sgrid)

        # Now do KDE

        vals_kde=KernelDensity(kernel='epanechnikov', bandwidth=np.sqrt(svar)).fit(sdraws[:,np.newaxis])
        log_dens = vals_kde.score_samples(sgrid[:,np.newaxis])
        epan_pdf = np.exp(log_dens)

        tab=pc.TABLECrate()
        tab.name=parnames[i]+"_"+"Gamma_Dist"
        pcu.add_colvals(tab,'Inten',sgrid)
        pcu.add_colvals(tab,'MargPDF',draws_mpdf)
        pc.set_key(tab,'alpha',alpha)
        pc.set_key(tab,'beta',beta)
        pc.set_key(tab,'smean',smean)
        pc.set_key(tab,'svar',svar)
        pc.set_key(tab,'Stepsize',ds)
        cds.add_crate(tab)

        tab=pc.TABLECrate()
        tab.name=parnames[i]+"_"+"Epan_KDE"
        pcu.add_colvals(tab,'Inten',sgrid)
        pcu.add_colvals(tab,'MargPDF',epan_pdf)
        pc.set_key(tab,'alpha',alpha)
        pc.set_key(tab,'beta',beta)
        pc.set_key(tab,'smean',smean)
        pc.set_key(tab,'svar',svar)
        pc.set_key(tab,'Stepsize',ds)
        cds.add_crate(tab)
        
    cds.write(filename,clobber=True)

    return

###################################################################################################

def modalpoint(f,CL):
    """
    Estimate mode and CL level bounds of probability density function given a set of MCMC samples

    Usage:

    mode = modalpoint(f)

    Inputs:

    f    array of MCMC samples
    CL   desired confidence level

    Outputs:

    mode of distribution
    lower CL bound
    upper CL bound
    UL flag - True if array limit reached when searching for lower bound
    LL flag - True if array limit reached when searching for upper bound
    CL_real - Actual CL achieved

    Description:

    Samples are sorted and divided into two subsets, each covering half of the range.
    The subset with the larger number of points is kept, and the process is repeated
    until the larger subset has only two points, or until to number of points does not change.
    The average of the remaining samples is returned as the mode. The test for number of points
    not changing is used to trap the case where a number of rejected draws appear after an
    accepted draw, and the parameter value is the same for all of them. In this case, the draws
    can be winowed down to a single parameter value appearing many times, and the mode will not
    converge.
    
    Once the mode is estimated, the routine starts at the index in the sorted array of samples closest to
    but less than the mode, and adds values alternately above and below the starting point until CL% of the
    total number of samples is reached. If the search encounters 0, an upper limit flag is set and the
    search continues above the mode only. If search encounters upper array bound, a lower limit flag is set,
    the search continues below the mode only.
    """

    import numpy as np
    import copy

    fcopy = copy.copy(f)
    fcopy.sort()

    npts = len(fcopy)
    if npts < 1:
        raise ValueError("Invalid Number of Samples")
    fmin=fcopy[0]
    fmax=fcopy[-1]
    npts_old=-1
    testit=True

    while testit:
        midrange = 0.5*(fmin+fmax)
        lorange = fcopy[fcopy<midrange]
        hirange = fcopy[fcopy>=midrange]
#        print fmin,fmax,len(lorange),len(hirange)
#        raw_input("PRESS ENTER TO CONTINUE")
        if(len(lorange)>=len(hirange)):
            fcopy=lorange
            fmax=midrange
        if(len(lorange)<len(hirange)):
            fcopy=hirange
            fmin=midrange
        npts_old=copy.copy(npts)
        npts=len(fcopy)
        testit=npts>2 and npts!=npts_old

    mode=sum(fcopy)/len(fcopy)
        
    # Now try to find bounds, starting at mode

    # First, get unique draws values
    
    fun=np.unique(f)

    if(mode>fun[-1]):
        msg="Mode %f greater than highest draws value %f" % (mode,fun[-1])
        raise ValueError(msg)
    if(mode<fun[0]):
        msg="Mode %f less than lowest draws value %f" % (mode,fun[0])
        raise ValueError(msg)
        
    # Find index closest to but less than mode

    istart=max(np.where(fun<=mode)[0])


#    print "Mode:\t%f" % mode
#    print "Start looking at value %f" % fun[istart]
    
#    raw_input("PRESS ENTER TO CONTINUE.")
    ULFlag=False
    LLFlag=False
    ilo=istart
    ihi=istart
    ncl=int(CL*len(f))
    ncount=0

    while ncount<ncl:
        ihi=ihi+1
        if ihi>(len(fun)-1):
            ihi=len(fun)-1
            LLFlag=True
        ncount=len(np.where((f>=fun[ilo]) & (f<=fun[ihi]))[0])
#        print "%d values between %f and %f" % (ncount,fun[ilo],fun[ihi])
        if(ncount>=ncl):
            break
        ilo=ilo-1
        if ilo<0:
            ilo=0
            ULFlag=True
        ncount=len(np.where((f>=fun[ilo]) & (f<=fun[ihi]))[0])
#        print "%d values between %f and %f" % (ncount,fun[ilo],fun[ihi])
        
    hibound=fun[ihi]
    lobound=fun[ilo]
    if ULFlag:
        lobound=0.0

    clreal=float(ncount)/len(f)
    return (mode,lobound,hibound,ULFlag,LLFlag,clreal)
    
###################################################################################################

def comp_pdfs(vals):

    # Estimate pdf from a set of values, assumed sampled from a probability density, by
    # epanechnikov kde and Fit to Gamma Distribution.
    # Only works on laptop, due to sklearn installation
    
    import numpy as np
    import xap_funs as xfuns
    import sys
    sys.path.append("/Users/fap/anaconda/lib/python2.7/site-packages/")
    from sklearn.neighbors import KernelDensity
    
    nvals=len(vals)
    vmean=sum(vals)/nvals
    vsig=np.sqrt(sum(vals*vals)/nvals-vmean*vmean)
    vals_grid=np.linspace(max(vmean-5.0*vsig,0.0),vmean+5.0*vsig,100)

    # First, sklearn kde with epan. kernel and bw set to sigma 
    vals_kde=KernelDensity(kernel='epanechnikov', bandwidth=vsig).fit(vals[:,np.newaxis])
    log_dens = vals_kde.score_samples(vals_grid[:,np.newaxis])
    epan_pdf = np.exp(log_dens)

    # Next, Gamma Distribution
    alpha = (vmean/vsig)*(vmean/vsig)
    beta = vmean/(vsig*vsig)
    gps = xfuns.Gamma_Prior(alpha,beta)
    Gam_pdf = gps(vals_grid)

    return vals_grid,epan_pdf,Gam_pdf
