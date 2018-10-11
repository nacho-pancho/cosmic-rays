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
        img     = fitsio.read(DATADIR+fname).astype(np.double)
        print len(np.unique(img.ravel())),np.unique(img.ravel())
        fbase   = fname[:-len(EXT)-1]
        fbase2  = fbase[fbase.rfind('/')+1:]
        #
	# mapeamos la imagen a escala logaritmica entre 0 y 255
	# a los efectos de deteccion talvez alcance y sobre
 	# y hasta facilite la cosa.
	#
        m,n = img.shape
	print   k,fbase2
        data = np.load(RESDIR+fbase+'-stats.npz')
        colstats = data["arr_0"]
        p00   = colstats[0,0]
        p10   = colstats[1,n/2]
        p50   = colstats[2,n/2]
        p90   = colstats[3,n/2]
        p100  = colstats[4,n/2]
        rmean = colstats[5,n/2]
        rvar  = colstats[6,n/2]
        img = np.minimum(np.maximum(p50,img),p100)
        
	print "LOG:",
	img     = np.log2(img)
        print np.unique(img)
	plt.close('all')
	plt.figure( k, figsize=(16,12) )
	h,b = np.histogram( img.ravel()[np.flatnonzero(img)],bins=50)
        print b,h
	plt.semilogy( b[1:],h,'*-' )
	plt.savefig(RESDIR + fbase + "-log-hist.png")
        nimg = img - np.min(img)
        nimg     = nimg*( 1.0 / np.max(nimg) )
        fpseudo = RESDIR + fbase + "-log.png"
	io.imsave( fpseudo, plt.get_cmap('hot')(nimg) )
 	fpgmgz  = DATADIR + fbase + "-log.pgm.gz"
        pnmgz.imwrite( fpgmgz, (255.0*nimg).astype(np.uint8), 255 )

        # print "LOGLOG:",
	# img     = np.log2(img)
	# plt.close('all')
	# plt.figure( k, figsize=(16,12) )
	# h,b = np.histogram( img.ravel()[np.flatnonzero(img)],bins=50)
	# plt.semilogy( b[1:],h,'*-' )
        # print b,h
	# plt.savefig(RESDIR + fbase + "-loglog-hist.png")
        # nimg     = img - np.min(img)
        # nimg     = nimg*( 1.0 / np.max(nimg) )
        # fpseudo = RESDIR + fbase + "-loglog.png"
	# io.imsave( fpseudo, plt.get_cmap('hot')(nimg) )
 	# fpgmgz  = DATADIR + fbase + "-loglog.pgm.gz"
        # pnmgz.imwrite( fpgmgz, (255.0*nimg).astype(np.uint8), 255 )
        # k       = k + 1
