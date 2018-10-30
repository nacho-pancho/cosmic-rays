import fitsio
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
import sys

DATADIR = '../data/'
RESDIR = '../results/'
EXT = '.fits'

os.system('mkdir -p ../' + RESDIR + 'cielo')
os.system('mkdir -p ../' + RESDIR + 'dark')
k = 0
if len(sys.argv) > 1:
    flist = sys.argv[1]
else:
    flist = 'all.list'

with open(DATADIR+flist) as filelist:
        for fname  in filelist:
                fbase = fname[:-len(EXT)]
                I = fitsio.read(DATADIR+fname)
                #
                # separamos en dos imagenes
                #
                hdr = fitsio.read_header(DATADIR+fname)
                m,n = I.shape
                I1 = I[:2048, 24:(24+2048)]
                I2 = I[:2048, n-(2048+24):(n-24)]
                fitsio.write(DATADIR+fbase+'1.fits',I1, extname='fits',compress='RICE')
                fitsio.write(DATADIR+fbase+'2.fits',I2, extname='fits',compress='RICE')
                k = k + 1
                #
                # falta: obtener lista de coordenadas normalizadas para pasarle al algoritmo de IPOL
                #
	#plt.show()
