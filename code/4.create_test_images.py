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
import crimg

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
DEBUG = True

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
                print "cargando imagen"
                #
                # read filled in image
                #
                sky_filled = fitsio.read(DATADIR + fbase + "-filled.fits")
                #
                # read the empirical distribution of its (erased) CRs
                #
                print "cargando histogramas"
                sky_roi_hist = np.load(RESDIR + fbase + '-9.roi-hist2.npz')['arr_0']
                Fs = np.cumsum(sky_roi_hist).astype(np.double)
                Fs = Fs*(1.0/Fs[-1])
                #
                # read dark frame from which we will superimpose our "ground truth CRs"
                #
                dark = fitsio.read(DATADIR + fdark)
                dark_roi_hist = np.load(RESDIR + darkbase + '-9.roi-hist2.npz')['arr_0']
                Fd = np.cumsum(dark_roi_hist).astype(np.double)
                Fd = Fd*(1.0/Fd[-1])
                plt.figure(d*2)
                plt.loglog(Fs)
                plt.loglog(Fd)
                plt.grid(True)
                plt.legend(('sky','dark'))
                plt.savefig(OUTDIR + fbase2+"+"+darkbase2+"-hist.svg")
                #
                # processing: must match distributions (PENDING)
                #                
                print "cargando mascara"
                dark_mask = pnmgz.imread(RESDIR + darkbase + '-7.mask2.pbm.gz').astype(np.bool)
                sky_test = np.copy(sky_filled)

                print "creando imagen sintetica"
                crimg.paste_cr(dark,dark_mask,Fd,Fs,sky_test)
                
                # M,N = dark.shape
                # for i in range(M):
                #      for j in range(N):
                #          if dark_mask[i,j]:
                #              d = dark[i,j]
                #              q = Fd[d] # q = F[d]                            
                #              aux = np.flatnonzero(Fs >= q)
                #              if len(aux):
                #                  if aux[0] > sky_fiiled[i,j]:
                #                     sky_test[i,j] = aux[0] # s = F^-1[Fd[q]]
                #              else:
                #                  sky[i,j] = len(Fs)-1
                print "guardando resultado."
                fitsio.write(outfile,sky_test)
               
                sky_test = np.log(sky_test-np.min(sky_test)+1)
                sky_test = (255.0/np.max(sky_test))*sky_test
                tif.imsave(OUTDIR + fbase2+"+"+darkbase2+"-artif-log.tiff",sky_test.astype(np.uint8))
                print "pronto."
        k = k + 1
