#
# -*- coding: UTF-8 -*-
#
# En esta etapa se rellenan las zonas marcadas como rayos cósmicos con muestras al azar
# similares a la distribución empírica observada en el fondo de la imagen.
#
# Esto genera una imagen intermedia supuestamente libre de CRs para luego superponerle
# una imagen de CRs "seguros".
#
import fitsio
import pnm
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology as morph
import tifffile as tif
import sys
import pnmgz
import crimg

DATADIR = '../data/'
RESDIR = '../results/'
CMAP = plt.get_cmap('PuRd')
EXT='.fits'
NHOOD = morph.disk(2) # para operaciones morfologicas

k = 0
plt.close('all')

if len(sys.argv) > 1:
    lista = sys.argv[1]
else:
    lista = "sky_sep.list"

with open(DATADIR+lista) as filelist:
    k = 0
    for fname  in filelist:
        img = fitsio.read(DATADIR+fname)
        img_filled = np.copy(img)
        fbase = fname[:-len(EXT)-1]
        fbase2 = fbase[fbase.rfind('/')+1:]
        mask = pnmgz.imread(RESDIR + fbase + '-7.mask2.pbm.gz')
        print k,fbase2
        img_bg = img[mask==False]
        Fbg = crimg.discrete_histogram(img_bg)
        Fbg = np.cumsum(Fbg)
        Fbg = Fbg.astype(np.double)*(1.0/Fbg[-1])
        crimg.inpaint(img,mask,Fbg,img_filled)
        fitsio.write(DATADIR + fbase + '-filled.fits', img_filled)
        preview = np.log(img_filled-np.min(img_filled)+1)
        preview = preview*(255.0/np.max(preview.ravel()))
        flap = RESDIR + fbase + '-filled.tiff'
        tif.imsave(flap,preview.astype(np.uint8))
        k = k + 1
