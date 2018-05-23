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
from xap_mcmc_funs import *
from crates_contrib.utils import *

# Get Inputs from parameter file:

fp=paramopen("xap_mcmc_only.par","wL",sys.argv)

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

# Finally, run mcmc:

print "Running MCMC..."

parnames,stats,accept,intensities=run_mcmc(F,C,s,sigma_s,ndraws,nburn,scale)

write_draws(drawsfile,parnames,stats,accept,intensities)

# Write equivalent of mpdfs files, but using draws to estimate mpdfs

write_draws_mpdfs(drawsmpdffile,parnames,intensities,ndrmesh)
