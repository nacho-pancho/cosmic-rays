import fitsio
import matplotlib.pyplot as plt
import numpy as np
#from skimage import io
import tifffile as tif
import os
import pnmgz
import sys

#
# recompile crimg module
# 
os.system('python setup.py build')
#
# import compiled module
#
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
    lista = "all_sep.list"

with open(DATADIR+lista) as filelist:
    for fname  in filelist:
        plt.close('all')
        #
        # boilerplate, file input/output preparation
        #
        fname = fname[:-1] 
        fpath = fname[:(fname.rfind('/')+1)]
        if not os.path.exists(RESDIR + fpath):
            os.system("mkdir -p " + RESDIR + fpath)
        fbase = fname[(fname.rfind('/')+1):fname.rfind('.')]
        fprefix = RESDIR + fpath + fbase
        #print fname,fpath,fbase,fprefix,
        print fname,
        #
        # load original, unfiltered image
        #
        img = fitsio.read(DATADIR+fname).astype(np.uint16)
        #
        # see if there is any non-uniform intensity
        #
        #a = np.median(img[ 100:, 100:])
        #b = np.median(img[ 100:,-100:])
        #c = np.median(img[-100:, 100:])
        #d = np.median(img[-100:,-100:])
        #print a,b,c,d
        #continue
        #
        # transform so that its percentile 10 is mapped to 0
        # and its maximum to approximately 255, in logarithmic scale
        #
        hist = crimg.discrete_histogram(img)
        chist = np.cumsum(hist)
        N = img.shape[0]*img.shape[1]
        base = np.flatnonzero(chist > (N/10))[0]
        #
        # the base of the logarithm is 2^(1/16)
        #
        limg = np.round(16.0*np.log2(np.maximum(img-base,1).astype(np.double))).astype(np.uint8)
        #
        # compute the discrete histogram of the log-transformed image
        #
        hist = crimg.discrete_histogram(limg)
        chist = np.cumsum(hist)
        hist = hist.astype(np.double)*(1.0/chist[-1])
        chist = chist.astype(np.double)*(1.0/chist[-1])
        #
        # show histograms
        #
        plt.figure(1,figsize=(10,15))
        plt.semilogy(hist,'*-')
        plt.semilogy(chist,'*-')
        plt.savefig(fprefix + '-1.hist.tiff')
        #
        #
        #
        med = np.flatnonzero(chist > 0.5)[0]
        thres = 50 # empirically observed 
        print 'median=',med,
        print 'thres=',thres
        tif.imsave(fprefix +'-2.log.tiff',limg)
        #
        # first  binary classification mask, crude
        # the marked regions are referred to as ROI (Region Of Interest)
        mask = (limg > thres) #
        #
        # close holes using morphological operations 
        #
        prev_mask = np.copy(mask)
        mask = crimg.binary_closure_8(mask) 
        dif = np.sum(np.abs(prev_mask-mask))
        #while dif:
        #    np.copyto(prev_mask,mask)
        #    mask = crimg.binary_closure_8(mask) 
        #    dif = np.sum(np.abs(prev_mask-mask))
        #    #print 'dif=',dif
        pnmgz.imwrite(fprefix + "-3.mask1.pbm.gz",mask,1)
        #tif.imsave(fprefix + '-3.mask1.tiff',mask.astype(np.uint8)*255)
        tif.imsave(fprefix + '-3.mask1.tiff',mask.astype(np.uint8)*255)
        #
        # compute Laplacian on ROIs
        #
        mask_lap = crimg.mask_laplacian(limg, mask);
        #
        # create a nice image for visualization purposes
        #
        NM = np.sum(mask)
        lhist = crimg.discrete_histogram(mask_lap[mask])
        clhist = np.cumsum(lhist)
        l95 = np.flatnonzero(clhist >= 95*NM/100)[0]
        mask_lap_img = np.minimum(mask_lap,l95)
        mask_lap_img = mask_lap_img.astype(np.double)*(255.0/np.max(mask_lap_img))
        tif.imsave(fprefix +'-4.roi_lap.tiff',mask_lap_img.astype(np.uint8))
        plt.figure(4,figsize=(10,15))
        lhist = lhist*(1.0/clhist[-1])
        clhist = clhist*(1.0/clhist[-1])
        plt.loglog(lhist,'*-')
        plt.loglog(clhist,'*-')
        plt.grid(True)
        plt.legend(('P','F'))
        plt.savefig(fprefix + '-5.roi-lap-hist1.svg')
        #
        # assign a unique label to each ROI
        #
        roi_label = crimg.roi_label(mask)
        #
        # save it (for debugging purposes)
        #
        #
        # Compute various statistics for each ROI.
        # For each ROI, the statistics are:
        #  0    1   2   3   4   5   6    7   8    9
        # size p00 p10 p25 p50 p75 p90 p100 mean std
        #
        # row 0 contains the aforementioned values for the union of all ROI's
        #
        hist = crimg.discrete_histogram(img[mask > 0])
        np.savez(fprefix + '-5.roi-hist1.npz',hist)        
        chist = np.cumsum(hist)
        hist = hist.astype(np.double)*(1.0/chist[-1])
        chist = chist.astype(np.double)*(1.0/chist[-1])
        plt.figure(2,figsize=(10,15))
        plt.loglog(hist[2450:],'*-')
        plt.loglog(chist[2450:],'*-')
        plt.grid(True)
        plt.legend(('P','F'))
        plt.savefig(fprefix + '-5.roi-hist1.svg')

        roi_stats = crimg.roi_stats(roi_label,mask_lap)
        np.savez(fprefix + '-6.roi-stats1.npz',roi_stats)
        #print roi_stats[0,:]
        #
        # filter out roi's based on stats
        #
        # first we create a vector of length L+1 called roi_mask
        # a 1 in the l-th position of roi_mask indicates that ROI l is to be kept;
        # a  0 indicates that it should be filtered out
        #
        roi_mask = np.empty(roi_stats.shape[0])
        # many criteria are possible. Below we give a sample filtering criterion:
        #
        print "# unfiltered ROIs in mask 1:",np.sum(roi_stats[:,0] > 0)
        p50 = roi_stats[0,4]
        p75 = roi_stats[0,5]
        p90 = roi_stats[0,6]
        p100 = roi_stats[0,7]        
        thres2 = 70
        #
        # large regions are considered CRs if their 75th percentile is above 70 in log scale
        #
        roi_mask = roi_stats[:,5] > thres2
        print "# after filtering large ROIs:",np.sum(roi_mask)
        #
        # small regions (<= 4 pixels) are considered CRs if their 25th percentile is above 140
        #
        singletons = roi_stats[:,0] <= 4 
        roi_mask[singletons]  = (roi_stats[singletons,3] > 140) 
        print "# after filtering small ROIs:",np.sum(roi_mask)
        #
        # filter out ROIs using the defined roi_mask vector
        #
        roi_label     = crimg.roi_filter(roi_label,roi_mask)
        # refine binary mask
        mask = roi_label > 0
        pnmgz.imwrite(fprefix + '-7.mask2.pbm.gz',mask,1)
        tif.imsave(fprefix + '-3.mask2.tiff',mask.astype(np.uint8)*255)
        #
        # save filtered labeled ROI pseudo-image
        #
        #roi_img = roi_label.astype(np.double)*(1.0/np.max(roi_label))
        #
        # compute and save stats of filtered ROIs in logarithmic scale
        #
        roi_stats = crimg.roi_stats(roi_label,mask_lap)
        np.savez(fprefix + '-8.roi-stats2.npz',roi_stats)        
        k = k + 1
        #
        # finally, compute and save histograms of filtered ROIs and
        # their complement in original image scale
        # 
        # this is for inpainting ROIs in input images
        # in and properly superimposing CRs on clean images
        #
        hist = crimg.discrete_histogram(img[mask > 0])
        np.savez(fprefix + '-9.roi-hist2.npz',hist)        
        chist = np.cumsum(hist)
        hist = hist.astype(np.double)*(1.0/chist[-1])
        chist = chist.astype(np.double)*(1.0/chist[-1])
        plt.figure(3,figsize=(10,15))
        plt.loglog(hist[2450:],'*-')
        plt.loglog(chist[2450:],'*-')
        plt.grid(True)
        plt.legend(('P','F'))
        plt.savefig(fprefix + '-9-roi-hist2.svg')

        hist = crimg.discrete_histogram(img[mask == 0])
        np.savez(fprefix + '-9.non-roi-hist2.npz',chist)        
        
#plt.show()
