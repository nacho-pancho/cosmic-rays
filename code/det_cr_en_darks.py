import fitsio
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
with open('darks.txt') as filelist:
        for fname  in filelist:
                img = fitsio.read(fname).astype(np.double)
                hdr = fitsio.read_header(fname)
                m,n = img.shape
                mask = np.empty((m,n))
                print hdr
                for j in range(n):
                        v = np.sort(img[:,j])
                        t0 = v[m/2]
                        t1 = v[95*m/100]
                        u = t0 + (t1-t0)*5
                        vi = np.var(v[:(90*m/100)])
                        u = t0 + np.sqrt(vi)*10
                        print 'j=',j,' t0=',t0,' t1=',t1,' u=',u,'var=',vi,' max=',v[-1]
                        mask[:,j] = (img[:,j] > u).astype(np.double)
                out = np.zeros((m,n,3))
                img = np.log2(img+1)
                img = img - np.min(img.ravel())
                img = img*(1.0/np.max(img))
                out[:,:,0] = mask
                out[:,:,1] = img
                io.imsave(fname[:-4]+"_pseudo.png",out)
                io.imsave(fname[:-4]+"_det.png",mask)
                #
                # falta: obtener lista de coordenadas normalizadas para pasarle al algoritmo de IPOL
                #
