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
    lista = "nacho.txt"

darklistfile = "dark_sep.txt"

plt.close('all')
CMAP = plt.get_cmap('hot')
with open(DATADIR+lista) as filelist:
    for fname  in filelist:
        img = fitsio.read(DATADIR + fname).astype(np.double)
        fbase = fname[6:-6]
        print fbase
        print DATADIR + darklistfile
        d = 0
        with open(DATADIR + darklistfile) as darklist:
            for fdark in darklist:
                img = fitsio.read(DATADIR + fname).astype(np.double)
                dark = fitsio.read(DATADIR + fdark).astype(np.double)
                darkbase = fdark[5:-6]
                print darkbase
                mask = pnm.imread(DATADIR + 'verdad/' + darkbase + '.pbm').astype(np.double)
                #
                # procesamiento
                #
                img2 = np.copy(img)
                offset = np.median(img.ravel())-np.median(dark.ravel())
                print offset
                img2[mask == 0] = img2[mask == 0] + (35.0/900.0)*(dark[mask == 0] )
                img2[mask == 0] = img2[mask == 0] + offset
                fitsio.write(DATADIR+"artif/"+fbase+"-"+darkbase+".fits",img2 )
                if d > 0:
                    plt.figure()
                    plt.imshow(np.log(img2-np.min(img2)+1),cmap=CMAP)
                d = d + 1
        k = k + 1
plt.show()
