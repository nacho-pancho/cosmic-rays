import fitsio
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
import pnmgz

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
        p00   = colstats[0,n/2]
        p10   = colstats[1,n/2]
        p50   = colstats[2,n/2]
        p90   = colstats[3,n/2]
        p100  = colstats[4,n/2]
        rmean = colstats[5,n/2]
        rvar  = colstats[6,n/2]
        u1 = p50 + (p90-p50)*5 # otro umbral
        u2 = rmean + np.sqrt(rvar)*5 # umbral de deteccion

        print 'IMAGEN',fbase2,'p00=',p00,'p10=',p10,'p50=',p50,'p90=',p90,'p100=',p100,'rmean=',rmean,'rvar=',rvar,'u1=',u1,'u2=',u2
        mask = (img >= u2).astype(np.double)
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
        pnmgz.imwrite(RESDIR+fbase+"-det0.pbm.gz",mask.astype(np.uint8),1)
        io.imsave(RESDIR+fbase+"-det0.png",mask)
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
        # segunda etapa: laplaciano
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
        #lap    = ( 1.0/np.max(lap) )*lap
        lap    = ( 1.0/p90 )*lap
        flap   = DATADIR+fbase+"-lap.pgm.gz"
        pnmgz.imwrite(flap,(255.0*lap).astype(np.uint8),255)
        flap   = RESDIR+fbase+"-lap.png"
        io.imsave(flap,lap)
        flap   = RESDIR+fbase+"-lapmap.png"
        io.imsave(flap,CMAP(lap))
        flap   = RESDIR+fbase+"-lapmap-mask.png"
        lapmap = CMAP(lap)
        lapmap[mask==False,0] = 0
        io.imsave(flap,lapmap)
        #
        # la idea es marcar como rayos cósmicos aquellas zonas muy brillantes y que además tienen
        # mucho contraste.
        # las zonas brillantes incluyen a las de contraste alto; en general, lass zonas brillantes en los CR
        # son casi todos pixeles de alto contraste local.
        # en cambio, objetos celestes legítimos tienen gradientes más bajos en relación al área que cubren
        # entonces la detección se plantea como una razón "cantidad de pixeles de alto contraste / cant brillantes"
        # en cada zona
        # las zonas son primero tratadas morfológicamente (clausura) y luego etiquetadas
        # todas estas operaciones son elementales y muy rápidas de realizar en C por ejemplo (no tanto en Python)
        #
        # FALTA
        #
        k = k + 1
