import fitsio
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
import pnmgz
import sys
import crimg

CMAP = plt.get_cmap('nipy_spectral')
DATADIR = '../data/'
RESDIR = '../results/'
EXT='.fits'
k = 0
plt.close('all')

if len(sys.argv) > 1:
    lista = sys.argv[1]
else:
    lista = "cielo_sep.txt"

with open(DATADIR+lista) as filelist:
    for fname  in filelist:
        fname = fname[:-1] 
        fpath = fname[:(fname.rfind('/')+1)]
        print fpath
        if not os.path.exists(RESDIR + fpath):
            os.system("mkdir -p " + RESDIR + fpath)
        fbase = fname[(fname.rfind('/')+1):fname.rfind('.')]
        fprefix = RESDIR + fpath + fbase
        print fname,fpath,fbase,fprefix,
        img = fitsio.read(DATADIR+fname).astype(np.uint16)
        hist = crimg.discrete_histogram(img)
        chist = np.cumsum(hist)
        N = img.shape[0]*img.shape[1]
        base = np.flatnonzero(chist)[0]
        #print 'median=',med
        #mask = (img > (med+100))
        #print 'nz=',np.count_nonzero(mask)
        #plt.figure(1)
        #plt.semilogy(hist)
        img = np.maximum(img - base,0)
        #limg = crimg.discrete_log2rootk(img,2)
        limg = np.round(16.0*np.log2(img.astype(np.double))).astype(np.uint8)
        #print np.unique(limg),
        hist = crimg.discrete_histogram(limg)
        chist = np.cumsum(hist)
        med = np.flatnonzero(chist > N/2)[0]
        print 'median=',med
        io.imsave(fprefix +'-log.png',CMAP(limg))
        mask = (limg > (med+12)) # great threshold!
        mask = crimg.binary_closure(mask) # closure
        mask = crimg.binary_closure(mask) # closure
        pnmgz.imwrite(fprefix + "-mask1.pbm.gz",mask,1)
        mask_lap = crimg.mask_laplacian(limg, mask);
        mask_lap_img = mask_lap.astype(np.double)*(1.0/np.max(mask_lap))
        io.imsave(fprefix +'.mask-laplacian.png',CMAP(mask_lap_img))
        #plt.figure(2,figsize=(10,10))
        #plt.semilogy(hist,'*-')
        #plt.figure(3,figsize=(10,10))
        #plt.semilogy(chist,'*-')
        roi_label = crimg.roi_label(mask)
        print np.max(roi_label)
        label_img = roi_label.astype(np.double)*(1.0/np.max(roi_label))
        io.imsave(fprefix + '-label.png',CMAP(label_img))
        roi_stats = crimg.roi_stats(roi_label,mask_lap)
        #
        # filter out roi's based on stats
        #
        # each roi_stats row consists of
        #  0    1   2   3   4   5   6    7   8    9
        # size p00 p10 p25 p50 p75 p90 p100 mean std
        # row 0 contains these values for the union of all ROI's; these
        # may be useful for classifying 1-pixel ROIs
        #
        # roi_mask below is a vector with L+1 elements where a 1 in the l-th position
        # indicates that ROI l is to be kept, and 0 indicates that it should be filtered out
        #
        # below we give a sample filtering criterion: we keep those areas where p90-p50 < p50-p10
        # that is, the median is significantly closer to the maximum (p90 is more robust) 
        # than to the minimum
        #
        roi_mask = np.zeros(roi_stats.shape[0])
        print roi_stats[0,:]
        print "# unfiltered ROIs",np.sum(roi_stats[:,0] > 0)
        roi_mask = (roi_stats[:,5] >= 0.1*roi_stats[0,7]) 
        print "# filtered ROIs",np.sum(roi_mask)
        roi_label_filtered     = crimg.roi_filter(roi_label,roi_mask)
        roi_label_filtered_img = roi_label_filtered.astype(np.double)*(1.0/np.max(roi_label_filtered))
        io.imsave(fprefix + '-filtered.png',CMAP(roi_label_filtered_img))
        pnmgz.imwrite(fprefix + '-mask2.pbm.gz',(roi_label_filtered_img > 0),1)
        np.savez(fprefix + '-stats.npz',roi_stats)
        k = k + 1
#plt.show()
