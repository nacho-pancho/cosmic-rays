import fitsio
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
import pnmgz
import sys
import crimg

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

plt.figure()

with open(DATADIR+lista) as filelist:
    for fname  in filelist:
        print fname,
        img = fitsio.read(DATADIR+fname).astype(np.uint16)
        hist = crimg.discrete_histogram(img)
        chist = np.cumsum(hist)
        N = img.shape[0]*img.shape[1]
        med = np.flatnonzero(chist)[0]
        #print 'median=',med
        #mask = (img > (med+100))
        #print 'nz=',np.count_nonzero(mask)
        #plt.figure(1)
        #plt.semilogy(hist)
        img = np.maximum(img - med,0)
        #limg = crimg.discrete_log2rootk(img,2)
        limg = np.round(16.0*np.log2(img.astype(np.double))).astype(np.uint8)
        print np.unique(limg),
        hist = crimg.discrete_histogram(limg)
        chist = np.cumsum(hist)
        med = np.flatnonzero(chist > N/2)[0]
        print 'median=',med
        plt.figure(1,figsize=(10,10))
        io.imsave(fname[(fname.rfind('/')+1):-1]+'.log.png',limg)
        mask = (limg > (med+12)) # great threshold!
        io.imsave(fname[(fname.rfind('/')+1):-1]+'.mask.png',mask*255)
        plt.figure(2,figsize=(10,10))
        plt.semilogy(hist,'*-')
        plt.figure(3,figsize=(10,10))
        plt.semilogy(chist,'*-')
        label = crimg.label(mask)
        print np.max(label)
        label = label.astype(np.double)*(1.0/np.max(label))
	cmap = plt.get_cmap('hot')
        io.imsave(fname[(fname.rfind('/')+1):-1]+'.label.png',cmap(label))
        k = k + 1
plt.show()
