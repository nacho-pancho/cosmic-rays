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
    lista = "cielo_sel.txt"

with open(DATADIR+lista) as filelist:
    for fname  in filelist:
        img = fitsio.read(DATADIR+fname)
        img_filled = np.copy(img)
        fbase = fname[:-len(EXT)-1]
        fbase2 = fbase[fbase.rfind('/')+1:]
        mask = pnmgz.imread(RESDIR + fbase + '-7.mask2.pbm.gz')
#       mask = morph.binary_dilation(mask,NHOOD) # para operaciones morfologicas
        #flap = RESDIR + fbase + '-mask.pbm.gz'
        #pnmgz.imwrite(flap,mask,1)
        #flap = RESDIR + fbase + '-mask3.tiff'
        #tif.imsave(flap, mask.astype(np.uint8)*255)
        img_bg = img[mask==False].ravel()
        img_bg_s = np.sort(img_bg)
        n = len(img_bg_s)
        print fbase
        for i in range(2048):
            for j in range(2048):
                if mask[i,j] == False:
                    continue
                a = np.random.random_sample()
                idx = np.int(a*n/2)
                v = img_bg_s[idx]
                img_filled[i,j] = v
        fitsio.write(DATADIR + fbase + '-filled.fits', img_filled)
        preview = np.log(img_filled-np.min(img_filled)+1)
        preview = preview*(255.0/np.max(preview.ravel()))
        flap = RESDIR + fbase + '-filled.tiff'
        tif.imsave(flap,preview.astype(np.uint8))
        plt.show()
