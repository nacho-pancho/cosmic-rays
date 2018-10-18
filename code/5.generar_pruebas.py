import fitsio
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
import pnmgz
import sys
import pnm

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
    lista = "cielo_sep.txt"

darklistfile = "dark_sep.txt"

plt.close('all')
CMAP = plt.get_cmap('hot')
with open(DATADIR+lista) as filelist:
    for fname  in filelist:
        img = fitsio.read(DATADIR + fname).astype(np.double)
        fbase = fname[:-6]
        fbase2 = fbase[(fbase.rfind('/')+1):]
        print fbase, fbase2
        print DATADIR + darklistfile
        d = 0
        with open(DATADIR + darklistfile) as darklist:
            for fdark in darklist:
                img = fitsio.read(DATADIR + fbase + "-nacho.fits").astype(np.double)
                dark = fitsio.read(DATADIR + fdark).astype(np.double)
                darkbase = fdark[5:-6]
                print darkbase
                mask = pnmgz.imread(RESDIR + fbase + '-mask.pbm.gz')
                #
                # procesamiento
                #
                img2 = np.copy(img)
                sorted_dark = np.sort(dark.ravel())
                sorted_img  = np.sort(img.ravel())
                p10d = sorted_dark[len(sorted_dark)/10]
                p10i = sorted_img[len(sorted_img)/10]
                img2[mask == 0] = img2[mask == 0] + ( 35.0/900.0)*(dark[mask == 0]-p10d)
                fitsio.write(RESDIR+fbase+"-"+darkbase+"-artif.fits",img2)
                img2 = np.log(img2-np.min(img2)+1)
                img2 = (1.0/np.max(img2))*img2
                io.imsave(RESDIR+fbase+"-"+darkbase+"-artif.png",img2)
                if d == -1:
                    plt.figure()
                    plt.imshow(np.log(img2-np.min(img2)+1),cmap=CMAP)
                d = d + 1
        k = k + 1
plt.show()
