import fitsio
import('../code/pnm.py')
cd ..
pwd
cd code
import pnm
cd ..
cd data/
ls
ls -ltr
I = fitsio.read
I = fitsio.read('cielo/linda3.1.fits')
I.shape
M = pnm.imread('mask/linda3.1-mascara.png')
M = pnm.imread('mask/linda3.1-mascara.pbm')
M.shape
import numpy as np
np.sum(M)
M
Im = I * M
lIm = np.log2(Im)
lIm = np.log2(Im+1)
plt.imshow(lIm)
import matplotlib.pyplot as plt
plt.imshow(lIm)
plt.show()
Im[M==0] = np.min(I)
lIm = np.log2(Im+1)
plt.imshow(lIm)
plt.show(lIm)
Im = I
Im[M==0] = np.min(I)
np.min(Im)
np.max(Im)
plt.imshow(Im)
plt.show()
plt.imshow(I)
plt.show()
Im = Im - np.min(Im)
plt.imshow(Im)
plt.show()
plt.imshow(np.log2(Im+1))
plt.show()
Ibg = I[M==1].ravel()
Ibgs = np.sort(Ibg)
plt.plot(Ibgs)
plt.show()
n = len(Ibgs)
plt.plot(Ibgs[1:n/2])
plt.show()
np.rand?
np.random
rand
rando,
random
np.random?
for i in range(2048):
    for j in range(2048):
        a = np.random.random_sample()
        idx = a*n/2
        v = Ibgs[idx]
        if M[i,j] == 0:
            Im[i,j] = v
for i in range(2048):
    for j in range(2048):
        a = np.random.random_sample()
        idx = np.int(a*n/2)
        v = Ibgs[idx]
        if M[i,j] == 0:
            Im[i,j] = v
plt.show(np.log2(Im))
Im = I
for i in range(2048):
    for j in range(2048):
        if M[i,j] == 1:
            continue
        a = np.random.random_sample()
        idx = np.int(a*n/2)
        v = Ibgs[idx]
for i in range(2048):
    for j in range(2048):
        if M[i,j] == 1:
            continue
        a = np.random.random_sample()
        idx = np.int(a*n/2)
        v = Ibgs[idx]
        Im[i,j] = v
plt.show(np.log2(Im))
plt.imshow(np.log2(Im))
plt.show()
plt.imshow(np.log2(Im-np.min(Im)+1))
plt.show()
fitsio.write?
fitsio.write('linda3.1-nacho.fits',Im)
np.min(Im)
%history

