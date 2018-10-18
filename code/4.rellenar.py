import fitsio
import pnm
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology as morph
from skimage import io
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
    lista = "cielo_sep.txt"

with open(DATADIR+lista) as filelist:
    for fname  in filelist:
        img = fitsio.read(DATADIR+fname).astype(np.double)
        img_filled = np.copy(img)
        fbase = fname[:-len(EXT)-1]
        fbase2 = fbase[fbase.rfind('/')+1:]
        mask = pnmgz.imread(RESDIR + fbase + '-mask.pbm.gz')
#        mask = morph.binary_dilation(mask,NHOOD) # para operaciones morfologicas
        flap = RESDIR + fbase + '-mask.pbm.gz'
        pnmgz.imwrite(flap,mask,1)
        flap = RESDIR + fbase + '-mask.png'
        io.imsave(flap, mask.astype(np.double))
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
        fitsio.write(DATADIR + fbase + '-nacho.fits', img_filled)
        preview = np.log(img_filled-np.min(img_filled)+1)
        preview = preview*(1.0/np.max(preview.ravel()))
        flap = RESDIR + fbase + '-nacho.png'
        io.imsave(flap,preview)
        plt.show()
