import fitsio
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os

DATADIR = '../data/'
RESDIR = '../results/'
EXT='.fits'
os.system('mkdir -p ../' + RESDIR + 'cielo')
os.system('mkdir -p ../' + RESDIR + 'dark')
k = 0
with open(DATADIR+'cielo_sep.txt') as filelist:
    for fname  in filelist:
        img = fitsio.read(DATADIR+fname).astype(np.double)
        fbase = fname[:-len(EXT)]
        fbase2 = fbase[fbase.rfind('/'):]
        #
        # separamos en dos imagenes
        #
        hdr = fitsio.read_header(DATADIR+fname)
        m,n = img.shape
        mask = np.empty((m,n))
        print hdr
        colstats = np.zeros((7,n))
        for j in range(n):
            v = np.sort(img[:,j])
            p00 = v[0]
            p10 = v[10*m/100]
            p50 = v[m/2]
            p90 = v[95*m/100]
            p100 = v[-1]
            rmean = np.mean(v[(10*m/100):(90*m/100)]) # media robusta
            rvar =np.var(v[:(90*m/100)]) # varianza robusta
            u1 = p50 + (p90-p50)*5 # otro umbral
            u2 = rmean + np.sqrt(rvar)*5 # umbral de deteccion
            #print 'j=',j,'p00=',p00,'p10',p10,' p50=',p50,' p90=',p90,' p100=',p100,' u1=',u1, 'u2=',u2
            colstats[:,j] = np.array([p00,p10,p50,p90,p100,rmean,rvar])
            mask[:,j] = (img[:,j] > u2).astype(np.double)
        # desplegue
        plt.figure(2*k+1)
        plt.title(fname)
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.plot(colstats[i,:])
            plt.grid(True)
        plt.savefig(RESDIR+fbase+'stats.png')
        out = np.zeros((m,n,3))
        img = np.log2(img+1)
        img = img - np.min(img.ravel())
        img = img*(1.0/np.max(img))
        out[:,:,0] = mask
        out[:,:,1] = img
        plt.title(fname)
        plt.figure(2*k+2)
        plt.imshow(out)
        io.imsave(RESDIR+fbase+"pseudo.png",out)
        io.imsave(RESDIR+fbase+"det.png",mask)
        k = k + 1
