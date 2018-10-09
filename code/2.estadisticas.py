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
print cmd
os.system(cmd)
cmd = 'mkdir -p ' + RESDIR + 'dark'
print cmd
os.system(cmd)
k = 0
plt.close('all')
with open(DATADIR+'cielo_sep.txt') as filelist:
    for fname  in filelist:
        img = fitsio.read(DATADIR+fname).astype(np.double)
        fbase = fname[:-len(EXT)-1]
        fbase2 = fbase[fbase.rfind('/')+1:]
        #
        # separamos en dos imagenes
        #
        hdr = fitsio.read_header(DATADIR+fname)
        m,n = img.shape
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
            colstats[:,j] = np.array([p00,p10,p50,p90,p100,rmean,rvar])
        # desplegue
        np.savez(RESDIR+fbase+'-stats.npz',colstats)
        plt.figure(2*k+1,figsize=(30,20))
        plt.title(fname)
        leg = ('p00','p10','p50','p90','p100','rmean','rvar')
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.plot(colstats[i,:])
            plt.grid(True)
            plt.title(leg[i])
        plt.savefig(RESDIR+fbase+'-stats.png')
        plt.close()
        plt.figure(2*k+1,figsize=(30,20))
        colstats = np.sort(colstats,axis=1)
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.plot(colstats[i,:])
            plt.title(leg[i])
            plt.grid(True)
        plt.savefig(RESDIR+fbase+'-ordenadas.png')
        #
        # estadisticas robustas a dos niveles
        # son las estadisticas de las estadisticas de las filas
        # mas o menos deberia dar lo mismo qiue las estadisticas globales
        #
        p00   = colstats[0,0]
        p10   = colstats[1,(10*n/100)]
        p50   = colstats[2,n/2]
        p90   = colstats[3,90*n/100]
        p100  = colstats[4,-1]
        rmean = colstats[5,n/2]
        rvar  = colstats[6,n/2]
        u1 = p50 + (p90-p50)*5 # otro umbral
        u2 = rmean + np.sqrt(rvar)*5 # umbral de deteccion

        print 'IMAGEN',fbase2,'p00=',p00,'p10=',p10,'p50=',p50,'p90=',p90,'p100=',p100,'rmean=',rmean,'rvar=',rvar,'u1=',u1,'u2=',u2
        mask = (img >= u1).astype(np.double)
        out = np.zeros((m,n,3))
        img = img - p00 
        img = img*(0.99/np.max(img))
        #img = np.log2(img.astype(np.double)+1)
        out[:,:,0] = mask
        out[:,:,1] = img
        out[:,:,2] = img
        plt.title(fname)
        plt.figure(2*k+2)
        plt.imshow(out)
        #io.imsave(RESDIR+fbase+"-pseudo.png",out)
        #io.imsave(RESDIR+fbase+"-det.png",mask)
        pnmgz.imwrite(RESDIR+fbase+"-det0.pbm.gz",mask.astype(np.uint8),1)
        k = k + 1
