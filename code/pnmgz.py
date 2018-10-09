import gzip
import pnm

def imread(fname):
	gzfile = gzip.GzipFile(fname)
	pnmfile = pnm.NetpbmFile(gzfile)
	return pnmfile.asarray()

def imwrite(fname,I,mv):
	gfo = gzip.GzipFile(filename=fname, mode='w')
	pfo = pnm.NetpbmFile(I, maxval=mv)
	pfo.write(gfo)
	gfo.close()

