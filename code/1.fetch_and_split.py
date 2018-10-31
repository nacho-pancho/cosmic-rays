#!/bin/bash

import fitsio
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
import sys

DATADIR = '../data/'
RESDIR = '../results/'
EXT = '.fits'

#
# download some HST images I have stored on my web site
# there are "infinte" of these at http://archive.stsci.edu/
# or
#
os.system('wget -c http://iie.fing.edu.uy/~nacho/data/cosmic_rays.7z')
#
# I have them 7zipped (much better than zip)
#
os.system('7zr x cosmic_rays.7z')
#
# create output dir
#
os.system('mkdir -p ../' + RESDIR + 'cielo')
os.system('mkdir -p ../' + RESDIR + 'dark')
#
# optionally, accept a list of images to process
# by default we process all
#
if len(sys.argv) > 1:
    flist = sys.argv[1]
else:
    flist = 'all.list'

#
# images come in pairs; they are actually two different 2048x2048 images side by side
# with some margin in between.
# Here we separate them as two, and ad the suffix .1 and .2
#
k = 0
with open(DATADIR+flist) as filelist:
        for fname  in filelist:
                fbase = fname[:-len(EXT)]
                I = fitsio.read(DATADIR+fname)
                #hdr = fitsio.read_header(DATADIR+fname)
                m,n = I.shape
                I1 = I[:2048, 24:(24+2048)]
                I2 = I[:2048, n-(2048+24):(n-24)]
                fitsio.write(DATADIR+fbase+'1.fits',I1, extname='fits',compress='RICE')
                fitsio.write(DATADIR+fbase+'2.fits',I2, extname='fits',compress='RICE')
                k = k + 1

