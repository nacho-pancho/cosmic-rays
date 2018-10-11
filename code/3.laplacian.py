import fitsio
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
import pnmgz

DATADIR = '../data/'
RESDIR = '../results/'
EXT='.fits'
cmd = 'mkdir -p ' + RESDIR + 'cielo'
os.system(cmd)
cmd = 'mkdir -p ' + RESDIR + 'dark'
os.system(cmd)
k = 0
plt.close('all')

def condlap(img,mask):
    m,n = img.shape
    #
    # entrada: padding con repeticion de bordes
    #
    img2 = np.zeros((m+2,n+2)).astype(np.double)
    img2[1:(m+1),1:(n+1)] = img.astype(np.double)
    img2[0,1:(n+1)] = img[0,:]
    img2[m,1:(n+1)] = img[0,:]
    img2[:,0] = img2[:,1]
    img2[:,n] = img2[:,n-11]
    #
    # resultado
    #
    L = np.zeros((m,n)).astype(np.double)
    #
    # solo calculado en la mascara
    #
    for i in range(m):
        for j in range(n):
            if mask[i,j]:
                dif = np.abs(4*img2[i+1,j+1]-img2[i,j+1]-img2[i+1,j]-img2[i+2,j+1]-img2[i+1,j+2])
                sum = np.abs(4*img2[i+1,j+1]+img2[i,j+1]+img2[i+1,j]+img2[i+2,j+1]+img2[i+1,j+2]) 
                L[i,j] = dif / sum 
    #L = 4.0*img2[1:m,1:n] - img2[1:m,0:(n-1)] - img2[1:m,2:(n+1)] - img2[0:(m-1),1:n] - img2[2:(m+1),1:n]
    return L

plt.close('all')
with open(DATADIR+'cielo_sep.txt') as filelist:
    for fname  in filelist:
        fbase  = fname[:-len(EXT)-1]
        fbase2 = fbase[fbase.rfind('/')+1:]
        print k,fbase2
        fdet   = RESDIR+fbase+"-det0.pbm.gz"
        img    = pnmgz.imread(DATADIR+fbase+'-log.pgm.gz')
        stats  = np.load(RESDIR+fbase+"-stats.npz")
        colstats = stats['arr_0']
        mask   = pnmgz.imread(fdet)
        M,N = colstats.shape
        lap    = condlap(img,mask)
        plt.figure(figsize=(30,20))
        llap = np.log2(lap.ravel()+1)
        h,b = np.histogram(llap[np.flatnonzero(llap)],bins=50)
        plt.plot((b[1:]+b[:-1])/2,h)
        plt.grid(True)
        flap   = RESDIR+fbase+"-lap.png"
        plt.savefig(RESDIR+fbase+"-laphist.png")
        #lap    = np.log2(lap+1e-5)
        lap    = (0.99/(np.max(lap)-np.min(lap)))*(lap-np.min(lap))
        print np.max(lap)
        io.imsave(flap,lap)
        flap   = RESDIR+fbase+"-lapmap.png"
        io.imsave(flap,plt.get_cmap('hot')(lap))
        k = k + 1
