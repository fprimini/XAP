#! /usr/bin/env python

# 06/14/2012  Updated to run under CIAO 4.4. Scipy and Pyfits dependencies removed.
# 06/19/2012  Updated to use akima interpolation provided by Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>
# 09/24/2012  Updated to remove _akima c library in akima interpolation routine.
# 06/13/2014  Updated to allow separate Gamma Distribution alpha/beta parameters via stacks for each source
#             and background
# 06/13/2014  Updated to allow command line parameter input

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
from xap_funs import *
from crates_contrib.utils import *

# Get Inputs from parameter file:

fp=paramopen("xap.par","wL",sys.argv)

evtfile=pget(fp,"infile")
outfile=pget(fp,"outfile")
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
    print "Unable to find PSF files.\nSetting source region ecfs to 1.0\n and background ecf to 0.0\n"

# Read expstack and test for null values

exps = array([])
expstack=pget(fp,"expstack")
try:
    ecr=read_file(expstack)
    exps=get_colvals(ecr,0)
except:
    print "Unable to find exposure maps.\nOutput units will be \"counts/sec\".\n"

#if len(sregs) != len(psfs):
#    print "Error: Source and PSF Stacks Inconsistent"

CL_desired   = pgetd(fp,"CL")
intenstack = pget(fp,"intenstack")
nmesh        = pgeti(fp,"nmesh")
clob         = pgetb(fp,"clobber")
verb         = pgeti(fp,"verbose")

paramclose(fp)

# Get EXPOSURE from event list header

exposure = 1.0
try:
    exposure= get_keyval(read_file(evtfile),'exposure')
except:
    print "Unable to find EXPOSURE keyword in header to %s\nOutput units will be \"counts\"." % evtfile

# Get MJD_OBS from event list header

mjdobs = -1.0
try:
    mjdobs= get_keyval(read_file(evtfile),'MJD_OBS')
except:
    print "Unable to find MJD_OBS keyword in header to %s\nObservation Time will be set to -1.0" % evtfile

if verb>0:
    print "Getting Events from %s\n" % evtfile
    print "Exposure:\t%f" % exposure
    print "Output results to %s\n" % outfile
    print "Source Regions:"
    for i in arange(0,len(sregs)):
        print sregs[i]

    print "\nBackground Regions:"
    print breg
    print "\nPSF Images:"
        
    for i in arange(0,len(psfs)):
        print psfs[i]
    print "\n"

# Now set up F and C arrays:

F,C = get_F_C_exp(evtfile,sregs,breg,psfs,exps,exposure)

number_of_params = len(C)          # This is source plus background
number_of_sources = number_of_params -1        # Number of sources only

if verb>0:
    for i in arange(0,number_of_sources):
        print "Counts in Region %d:\t\t%f" % (i,C[i])

    print "\nCounts in Background Region:\t%f" % C[number_of_sources]

    print "\nF:"
    print F
    print "\n"

# Solve for MLE intensities and errors:

s,sigma_s = get_s_sigma(F,C)

# Set up to build joint posterior hypercube

ds = zeros([number_of_params])
svecs = []                                     # List or arrays of grid points for each s
sgrid = Intensity_Range(nvals=nmesh)                      # Instance of class to compute grid for a given s,sigma

if verb>0:
    sgrid.attributes()                             # Print the attributes

for i in arange(0,number_of_params):
    stemp,ds[i] = sgrid(s[i],sigma_s[i])
    svecs.append(stemp)
    if(verb>0):
        print "Source %d:\tMesh Length: %d" % (i,len(stemp))

smesh = ix_(*svecs)                            # Reshape svecs for broadcasting  

# Set up Prior Distributions

print "\nBuilding Priors....."

# Try to read stack of intensities and variances for Source/Background Gamma PRior Distributions

try:
    
    # If a stack of intens and variances were input, use them to compute means and vairances of aperture counts (theta's), and use those
    # to compute alphas and betas of Gamma Distributions (remember, priors are for apetures counts, not individual sources).
 
    abcr = read_file(intenstack)
    s_mean = get_colvals(abcr,0)
    s_var  = get_colvals(abcr,1)
    theta_mean = zeros(number_of_params)
    theta_var  = zeros(number_of_params)

    print "\nMeans and Variances of counts for each aperture"

    for i in range(0,number_of_params):
        for j in range(0,number_of_params):
            theta_mean[i] = theta_mean[i] + F[i,j]*s_mean[j]
            theta_var[i] = theta_var[i] + F[i,j]*F[i,j]*s_var[j]
        print "%e\t%e" % (theta_mean[i],theta_var[i])

    alphas = theta_mean*theta_mean/theta_var
    betas  = theta_mean/theta_var

    print "Estimates for Prior Distribution Parameters:"
    print "\nalpha    \tbeta"
    print "-----    \t----"
    for i in range(0,number_of_params):
        print "%9.3e\t%9.3e" % (alphas[i],betas[i])


except:

    # Failed, so set to default uninformative priors

    print "Using Non-informative Priors....."
    alphas = ones(len(sregs)+1)
    betas  = zeros(len(sregs)+1)


gps=[]
for i in arange(0,number_of_sources):          # Gamma Priors with source attributes
    gps.append(Gamma_Prior(alphas[i],betas[i]))    # For sources
    
gps.append(Gamma_Prior(alphas[-1],betas[-1]))        # for background

# Now build joint posterior hypermesh

joint_posterior = 1.0

for i in arange(0,number_of_sources):          # For sources
    joint_posterior = joint_posterior*ones(len(smesh[i]))

joint_posterior = joint_posterior*ones(len(smesh[number_of_sources])) # last member of smesh tuple is for background

# At this point, all mesh points in the hypercube are populated. Now need to add the likelihoods

for i in arange(0,number_of_params):
    Theta = 0.0
    for j in arange(0,number_of_params):
        Theta = Theta + F[i,j]*smesh[j]
    joint_posterior = joint_posterior*pmf(C[i],Theta)*gps[i](Theta)

# And now compute marginalized pdf for each source in turn and store in list mpdfs

mpdfs=[]
for i in range(0,number_of_params):
    mpdfs.append(marginalize_posterior(joint_posterior,i,ds[i]))

# And save in Output file:

allcrates=[]                                          # A list of TABLECrates for all sources and bkg
for i in range(0,number_of_sources):
    # Changed 7/24/13 to follow kjg suggestion to fix crates bug in ciao4.5 re unnamed tablecrates
    tab=TABLECrate()
    tab.name="Source_{}".format(i)
    allcrates.append(tab)

tab=TABLECrate()
tab.name="Background"
allcrates.append(tab)

#thdulist=HDUList(PrimaryHDU(arange(100)))

# First the sources, one per extension

print "\t\t\t\tIntensity\tLower Bound\tUpper Bound"
print "\t\t\t\t---------\t-----------\t-----------"

for i in range(number_of_sources):
    srcmode,slo,shi,CLreal,zmode,zCL=pdf_summary(svecs[i],mpdfs[i],CL_desired)
    print "MLE   Estimate for Source %d =\t%e\t%e\t%e" % (i,s[i],s[i]-sigma_s[i],s[i]+sigma_s[i])
    print "Bayes Estimate for Source %d =\t%e\t%e\t%e" % (i,srcmode,slo,shi)

    add_colvals(allcrates[i],'Inten',svecs[i])
    add_colvals(allcrates[i],'MargPDF',mpdfs[i])
    set_key(allcrates[i],'Source',sregs[i])
    set_key(allcrates[i],'Stepsize',ds[i])
    set_key(allcrates[i],'MLE',s[i])
    set_key(allcrates[i],'sigmaMLE',sigma_s[i])
    set_key(allcrates[i],'Mode',srcmode)
    set_key(allcrates[i],'S_lolim',slo)
    set_key(allcrates[i],'S_hilim',shi)
    set_key(allcrates[i],'CL',CLreal)
    set_key(allcrates[i],'ZeroMode',zmode)
    set_key(allcrates[i],'ZeroCL',zCL)
    set_key(allcrates[i],'EXPOSURE',exposure)
    set_key(allcrates[i],'MJD_OBS',mjdobs)

# The last extension contains the background pdf
    
srcmode,slo,shi,CLreal,zmode,zCL=pdf_summary(svecs[-1],mpdfs[-1],CL_desired)
print "MLE   Estimate for Background =\t%e\t%e\t%e" % (s[-1],s[-1]-sigma_s[-1],s[-1]+sigma_s[-1])
print "Bayes Estimate for Background =\t%e\t%e\t%e" % (srcmode,slo,shi)

add_colvals(allcrates[-1],'Inten',svecs[-1])
add_colvals(allcrates[-1],'MargPDF',mpdfs[-1])
set_key(allcrates[-1],'Source',sregs[-1])
set_key(allcrates[-1],'Stepsize',ds[-1])
set_key(allcrates[-1],'MLE',s[-1])
set_key(allcrates[-1],'sigmaMLE',sigma_s[-1])
set_key(allcrates[-1],'Mode',srcmode)
set_key(allcrates[-1],'S_lolim',slo)
set_key(allcrates[-1],'S_hilim',shi)
set_key(allcrates[-1],'CL',CLreal)
set_key(allcrates[-1],'ZeroMode',zmode)
set_key(allcrates[-1],'ZeroCL',zCL)
set_key(allcrates[-1],'EXPOSURE',exposure)
set_key(allcrates[-1],'MJD_OBS',mjdobs)

# and write to file specified in outfile parameter

ds = CrateDataset()
for i in range(0,number_of_params):
    ds.add_crate(allcrates[i])

ds.write(outfile,clobber=clob)

# Compute means and variances of individual source (or background) intensities

print "\nEstimates of Means and Variances of Individual Sources"
print "\ns_mean    \tVar(s)"
print "------    \t------"

s_mean = zeros(number_of_params)
s_var  = zeros(number_of_params)

for i in range(0,number_of_params):
    s_mean[i],s_var[i] = posterior_stats(svecs[i],mpdfs[i])
    print "%10.4e\t%10.4e" % (s_mean[i],s_var[i])



