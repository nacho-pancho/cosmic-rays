#
# -*- coding: UTF-8 -*-
#
# Este script toma las imágenes "rellenadas" (nacho)
# y genera una imagen de prueba superponiendo sobre ellas
# los CRs obtenidos de los darks.
#
# Para hacer esto correctamente hay que ajustar los niveles
# de intensidad de CR de los darks con los de las imagenes "rellenadas",
# que pueden ser bien distintos.
#
# La metodologìa propuesta es tomar la estadística de los CRs ya filtrados
# en la imagen rellenada, compararla con las estadísticas de los CRs de los darks,
# y hacer un mapeo lineal de modo que estas últimas sean semejantes a las primeras
#
# Ahora lo que se hace es mucho más básico y no creo que sea correcto
#
import fitsio
import matplotlib.pyplot as plt
import numpy as np
import os
import pnmgz
import sys
import tifffile as tif

DATADIR = '../data/'
RESDIR = '../results/'
OUTDIR = '../results/artif/'
EXT='.fits'
cmd = 'mkdir -p ' + RESDIR + 'cielo'
print cmd
os.system(cmd)
cmd = 'mkdir -p ' + RESDIR + 'dark'
print cmd
os.system(cmd)
cmd = 'mkdir -p ' + OUTDIR 
print cmd
os.system(cmd)
k = 0
plt.close('all')

if len(sys.argv) > 1:
    lista = sys.argv[1]
else:
    lista = "sky_sep.list"

darklistfile = "dark_sep.list"

plt.close('all')
CMAP = plt.get_cmap('hot')
with open(DATADIR+lista) as filelist:
    for fname  in filelist:
        img = fitsio.read(DATADIR + fname).astype(np.double)
        fbase = fname[:-6]
        fbase2 = fbase[(fbase.rfind('/')+1):]
        #print DATADIR + darklistfile
        d = 1
        with open(DATADIR + darklistfile) as darklist:
            plt.close('all')
            for fdark in darklist:
                darkbase = fdark[:-6]
                darkbase2 = fdark[(darkbase.rfind('/')+1):-6]
                print "sky=",fbase2, "dark=",darkbase2,
                outfile = OUTDIR+fbase2+"+"+darkbase2+"-artif.fits"
                if os.path.exists(outfile):
                    print "ya calculado."
                    continue
                #
                # read filled in image
                #
                sky = fitsio.read(DATADIR + fbase + "-filled.fits")
                #
                # read the empirical distribution of its (erased) CRs
                #
                sky_roi_hist = np.load(RESDIR + fbase + '-9.roi-hist2.npz')['arr_0']
                Fs = np.cumsum(sky_roi_hist).astype(np.double)
                Fs = Fs*(1.0/Fs[-1])
                #
                # read dark frame from which we will superimpose our "ground truth CRs"
                #
                dark = fitsio.read(DATADIR + fdark)
                dark_mask = pnmgz.imread(RESDIR + darkbase + '-7.mask2.pbm.gz').astype(np.bool)
                dark_roi_hist = np.load(RESDIR + darkbase + '-9.roi-hist2.npz')['arr_0']
                Fd = np.cumsum(dark_roi_hist).astype(np.double)
                Fd = Fd*(1.0/Fd[-1])
                #if len(Fs) < len(Fd):
                #    Fs = np.concatenate(Fs,np.ones(len(Fd)-len(Fs)))
                plt.figure(d*2)
                plt.loglog(Fs)
                plt.loglog(Fd)
                plt.grid(True)
                plt.legend(('sky','dark'))
                plt.savefig(OUTDIR + fbase2+"+"+darkbase2+"-hist.svg")
                #
                # processing: must match distributions (PENDING)
                #                
                sky2 = np.copy(sky)
                
                M,N = dark.shape
                for i in range(M):
                    for j in range(N):
                        if dark_mask[i,j]:
                            d = dark[i,j]
                            q = Fd[d] # q = F[d]                            
                            sky2[i,j] = np.flatnonzero(Fs >= q)[0] # s = F^-1[Fd[q]]
                fitsio.write(outfile,sky2)
                
                sky2 = np.log(sky2-np.min(sky2)+1)
                sky2 = (255.0/np.max(sky2))*sky2
                tif.imsave(OUTDIR + fbase2+"+"+darkbase2+"-artif-log.tiff",sky2.astype(np.uint8))
                if d == -1:
                    plt.figure()
                    plt.imshow(np.log(sky2-np.min(sky2)+1),cmap=CMAP)
                d = d + 1
                print "pronto."
        k = k + 1
plt.show()
