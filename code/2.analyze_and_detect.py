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
        M,N = img.shape
        #
        # transform image to logarithmic scale
        # so that all values below the median are mapped to 0
        # and all values above the median are mapped from 1 to 255
        #
        hist = crimg.discrete_histogram(img)
        chist = np.cumsum(hist)
        L = img.shape[0]*img.shape[1]
        base = np.flatnonzero(chist > (L/2))[0]
        
        img = img.astype(np.double)
        img0 = img-base
        img1 = np.maximum(img0.astype(np.double),1)
        img2 = np.log2(img1)
        limg = (img2*(255.0/np.max(img2))).astype(np.uint8)
        tif.imsave(fprefix +'-2.log.tiff',limg)
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
        plt.figure(1,figsize=(10,15),dpi=300)
        plt.semilogy(hist,'*-')
        plt.semilogy(chist,'*-')
        plt.savefig(fprefix + '-1.hist.tiff')
        #
        # first binary classification mask: 
        #  log-scale value above absolute value 50
        #  50 is a magic number set by hand.
        #
        # the marked regions are referred to as ROI (Region Of Interest)
        #
        thres = 50 
        print 'thres=',thres
        mask = (limg > thres)
        #
        # close holes using morphological closure with neighborhood 8
        #
        tif.imsave(fprefix + '-3.mask0.tiff',mask.astype(np.uint8)*255)
        mask = crimg.binary_closure_8(mask)
        pnmgz.imwrite(fprefix + "-3.mask1.pbm.gz",mask,1)
        tif.imsave(fprefix + '-3.mask1.tiff',mask.astype(np.uint8)*255)
        #
        # compute Laplacian on ROIs
        #
        lap = crimg.laplacian(limg)
        lap = lap.astype(np.uint16)
        mask_lap = lap
        mask_lap[mask == False] = 0
        #lap = limg.astype(np.int32)
        #                        center           south        north          east            west
        #lap[1:-1,1:-1] = 8*lap[1:-1,1:-1] - lap[2:,1:-1] - lap[:-2,1:-1] - lap[1:-1,2:] - lap[1:-1,:-2]
        #                                         se           ne               nw           sw
        #lap[1:-1,1:-1] =   lap[1:-1,1:-1] - lap[2:,2:] - lap[:-2, 2:] - lap[:-2, :-2] - lap[2:,:-2]
        #lap = - lap / 16
        #
        # PARENTHESIS: create a nice image for visualization purposes
        #
        if False:
            NM = np.sum(mask)
            lhist = crimg.discrete_histogram(mask_lap[mask])
            clhist = np.cumsum(lhist)
            l95 = np.flatnonzero(clhist >= 95*NM/100)[0]
            print 'l95:',l95
            mask_lap_img = np.minimum(mask_lap,l95)
            mask_lap_img = mask_lap_img.astype(np.double)*(255.0/l95)
            tif.imsave(fprefix +'-4.roi_lap.tiff',mask_lap_img.astype(np.uint8))
            mask_lap_img = []        
            #
            # compute histogram of log-scale laplacian
            #
            lhist = crimg.discrete_histogram(lap)
            clhist = np.cumsum(lhist)
            lap_img = lap.astype(np.double)*(255.0/np.max(lap))
            tif.imsave(fprefix +'-4.lap.tiff',lap_img.astype(np.uint8))
            lap_img =  []
            plt.figure(4,figsize=(10,15))
            lhist = lhist*(1.0/clhist[-1])
            clhist = clhist*(1.0/clhist[-1])
            plt.loglog(lhist,'*-')
            plt.loglog(clhist,'*-')
            plt.grid(True)
            plt.legend(('P','F'))
            plt.savefig(fprefix + '-5.roi-lap-hist1.tiff')
        #
        # END PARENTHESIS
        #
        #
        # assign a unique label to each ROI
        #
        roi_label = crimg.roi_label(mask)
        labels = np.unique(roi_label)
        #
        # Compute various statistics for each ROI.
        # For each ROI, the statistics are:
        #  0    1   2   3   4   5   6    7   8    9
        # size p00 p10 p25 p50 p75 p90 p100 mean std
        #
        # row 0 contains the aforementioned values for the union of all ROI's
        #
        if False:
            hist = crimg.discrete_histogram(img[mask > 0].astype(np.uint8))
            np.savez(fprefix + '-5.roi-hist1.npz',hist)
            chist = np.cumsum(hist)
            hist = hist.astype(np.double)*(1.0/chist[-1])
            chist = chist.astype(np.double)*(1.0/chist[-1])
            plt.figure(2,figsize=(10,15))
            plt.loglog(hist[2450:],'*-')
            plt.loglog(chist[2450:],'*-')
            plt.grid(True)
            plt.legend(('P','F'))
            plt.savefig(fprefix + '-5.roi-hist1.tiff')

        #
        #
        #
        roi_stats = crimg.roi_stats(roi_label,mask_lap)
        np.savez(fprefix + '-6.roi-stats1.npz',roi_stats)
        laphist = crimg.discrete_histogram(mask_lap[mask_lap > 0])
        claphist = np.cumsum(laphist)
        laphist = laphist.astype(np.double)*(1.0/claphist[-1])
        plt.figure(3)

        loglik = crimg.roi_loglik(mask_lap.astype(np.uint16),roi_label,laphist)
        loklik = loglik / roi_stats[:,0]
        plt.plot(loglik)
        plt.plot(roi_stats[:,0])
        #plt.show()
        maxll = np.max(loglik)
        print "max log lik=",maxll

        loglik_img = np.zeros((M,N))
        labels = np.flatnonzero(loglik)
        print labels
        print loglik[labels]
        for i in labels:
            loglik_img[roi_label == i] = 255.0*loglik[i]/maxll
        tif.imsave(fprefix + '-5.loglik.tiff',loglik_img.astype(np.uint8))
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
        #
        # large regions are considered CRs if their average is above twice the global average
        #
        roi_mask = roi_stats[:,8] > roi_stats[0,8]
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
        tif.imsave(fprefix + '-7.mask2.tiff',mask.astype(np.uint8)*255)
        #
        # show the result in pseudocolor
        #
        lap = []
        print M,N
        result = np.empty((M,N,3)).astype(np.uint8)
        result[:,:,0] = mask*limg
        result[:,:,1] = mask*limg
        result[:,:,2] = limg
        tif.imsave(fprefix + '-7.result.tiff',result,photometric='rgb')

        #
        # save filtered labeled ROI pseudo-image
        #
        #roi_img = roi_label.astype(np.double)*(1.0/np.max(roi_label))
        #
        # compute and save stats of filtered ROIs in logarithmic scale
        #
        roi_stats = crimg.roi_stats(roi_label,mask_lap)
        np.savez(fprefix + '-8.roi-stats2.npz',roi_stats)
        mask_lap =  []
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
        plt.savefig(fprefix + '-9-roi-hist2.tiff')

        hist = crimg.discrete_histogram(img[mask == 0])
        np.savez(fprefix + '-9.non-roi-hist2.npz',chist)
        k = k + 1

#plt.show()
