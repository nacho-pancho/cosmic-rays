import fitsio
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
import pnmgz
import sys

DATADIR = '../data/'
RESDIR = '../results/'
EXT='.fits'
cmd = 'mkdir -p ' + RESDIR + 'cielo'
print cmd
os.system(cmd)
cmd = 'mkdir -p ' + RESDIR + 'dark'
print cmd
os.system(cmd)
k = 0
plt.close('all')

if len(sys.argv) > 1:
    lista = sys.argv[1]
else:
    lista = "cielo.txt"

with open(DATADIR+lista) as filelist:
    for fname  in filelist:
        img = fitsio.read(DATADIR+fname).astype(np.double)
        fbase = fname[:-len(EXT)-1]
        fbase2 = fbase[fbase.rfind('/')+1:]
        #
        # separamos en dos imagenes
        #
        hdr = fitsio.read_header(DATADIR+fname)
        print 'IMAGEN',fbase,
        print 'EXPTIME=',hdr['EXPTIME'],
        print 'CCDGAIN=',hdr['CCDGAIN']
        k = k + 1
