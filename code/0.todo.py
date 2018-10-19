# -*- coding: UTF-8 -*-
import fitsio
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.morphology import closing,disk
from skimage.morphology import remove_small_objects
from skimage.measure import label
from skimage.color import label2rgb
import os
import pnmgz
import sys

#################################################################################################

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
                #L[i,j] = dif / sum
                L[i,j] = dif
    #L = 4.0*img2[1:m,1:n] - img2[1:m,0:(n-1)] - img2[1:m,2:(n+1)] - img2[0:(m-1),1:n] - img2[2:(m+1),1:n]
    return L

#################################################################################################

DATADIR = '../data/'
RESDIR = '../results/'
CMAP = plt.get_cmap('PuRd')
EXT='.fits'
NHOOD = disk(1) # para operaciones morfologicas


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

with open(DATADIR+lista) as filelist:
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
        p00   = colstats[0,n/2]
        p10   = colstats[1,n/2]
        p50   = colstats[2,n/2]
        p90   = colstats[3,n/2]
        p100  = colstats[4,n/2]
        rmean = colstats[5,n/2]
        rvar  = colstats[6,n/2]
        u1 = p50 + (p90-p50)*5 # otro umbral
        u2 = rmean + np.sqrt(rvar)*3 # umbral de deteccion
        u3 = p90
        #
        # primera máscara es definida en base a un umbral simple sobre el brillo
        # la idea es que contenga a todos los CRs, aunque incluya muchos falsos positivos
        # esto va a ser refinado luego
        #
        mask = (img >= u2)
        print 'IMAGEN',fbase2,'p00=',p00,'p10=',p10,'p50=',p50,'p90=',p90,'p100=',p100,'rmean=',rmean,'rvar=',rvar,'u1=',u1,'u2=',u2,
        print 'NDET=',np.sum(mask)
        sorted_roi0 = np.sort(img[mask])
        np.savez(RESDIR+fbase+'-roi-stats.npz',sorted_roi0)
        out = np.zeros((m,n,3))
        #img = img - p00
        #img = img*(0.99/np.max(img))
        #img = np.log2(img.astype(np.double)+1)
        out[:,:,0] = mask
        out[:,:,1] = img
        out[:,:,2] = img
        plt.title(fname)
        plt.figure(2*k+2)
        plt.imshow(out)
        #io.imsave(RESDIR+fbase+"-pseudo.png",out)
        #io.imsave(RESDIR+fbase+"-det.png",mask)
        pnmgz.imwrite(RESDIR+fbase+"-mask0.pbm.gz",mask.astype(np.uint8),1)
        io.imsave(RESDIR+fbase+"-mask0.png",mask)
        #
        # para la etapa posterior, trabajamos con el logaritmo de la imagen
        #
        img     = np.log2(img)
        plt.close('all')
        plt.figure( k, figsize=(16,12) )
        h,b = np.histogram( img.ravel()[np.flatnonzero(img)],bins=20)
        plt.semilogy( b[1:],h,'*-' )
        plt.savefig(RESDIR + fbase + "-log-hist.png")
        nimg = img - np.min(img)
        nimg     = nimg*( 1.0 / np.max(nimg) )
        fpseudo = RESDIR + fbase + "-log.png"
        io.imsave( fpseudo, CMAP(nimg) )
        fpgmgz  = DATADIR + fbase + "-log.pgm.gz"
        pnmgz.imwrite( fpgmgz, (255.0*nimg).astype(np.uint8), 255 )
        #
        # la segunda etapa de detecciòn
        # està basada en la variación de cáda zona maracada
        # en la primera etapa. La hipótesis es que los CRs
        # tienen una variación muy fuerte en toda la zona,
        # mientras que estrellas u otros objetos celeestes
        # se ven más difuminados
        #
        # la medida de variación es el valor absoluto del Laplaciano
        #
        lap    = condlap(img,mask)
        plt.figure(figsize=(16,12))
        lapnz = lap.ravel()[np.flatnonzero(lap)]
        plt.hist(lapnz,20)
        plt.savefig(RESDIR+fbase+"-laphist.png")
        slap = np.sort(lapnz)
        N = len(slap)
        p90 = slap[N*90/100]
        lap = np.minimum(p90,lap)
        lap    = ( 1.0/p90 )*lap
        flap   = DATADIR+fbase+"-lap.pgm.gz"
        pnmgz.imwrite(flap,(255.0*lap).astype(np.uint8),255)
        flap   = RESDIR+fbase+"-lap.png"
        io.imsave(flap,lap)
        flap   = RESDIR+fbase+"-lapmap.png"
        io.imsave(flap,CMAP(lap))
        flap   = RESDIR+fbase+"-lapmap-mask.png"
        lapmap = CMAP(lap)
        mask = closing(mask,NHOOD)
        mask  = remove_small_objects(mask,1)
        lapmap[mask==False,0] = 0
        io.imsave(flap,lapmap)

        label_image = label(mask)
        #label_image_overlay = label2rgb(label_image,image=lapmap)
        #flap   = RESDIR+fbase+"-lapmap-labeling.png"
        #io.imsave(flap,label_image_overlay)
        #
        # con el labeling realizado, marcamos como CR todas las zonas que tienen un valor medio de Laplaciano alto
        # dentro de la zona
        #
        max_label = np.max(label_image.ravel())
        ndet = 0
        ntot = max_label
        for l in range(max_label):
            ml = np.median(lap[label_image == l])
            if ml < 0.1:
                ndet = ndet + 1
                lapmap[label_image == l,0] = 0
        flap   = RESDIR+fbase+"-lapmap-filtered.png"
        io.imsave(flap,lapmap)
        final_mask = (lapmap[:,:,0] > 0)
        flap   = RESDIR+fbase+"-mask.pbm.gz"
        pnmgz.imwrite(flap,final_mask,1)
        flap   = RESDIR+fbase+"-mask.png"
        io.imsave(flap,final_mask)
        k = k + 1
