#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <math.h>
#include <omp.h>

///
/// Contains all the relevant parameters  about the patch decomposition procedure.
///
#define CCLIP(x,a,b) ( (x) > (a) ? ( (x) < (b) ? (x) : (b) ) : (a) )
#define UCLIP(x,a) ( (x) < (a) ? (x) : (a)-1 )
#define REFLECT(x,a,b) ( (x) > (a) ? ( (x) < (b) ? (x) : ((b)-(x)) ) : ((a)-(x)) )

/// Python adaptors
static PyObject *binary_closure       (PyObject* self, PyObject* args);
static PyObject *discrete_histogram   (PyObject* self, PyObject* args);
static PyObject *discrete_log2rootk   (PyObject *self, PyObject *args);
static PyObject *mask_laplacian       (PyObject *self, PyObject *args); 
static PyObject *mask_refine          (PyObject *self, PyObject *args); 
static PyObject *roi_label            (PyObject *self, PyObject *args); 
static PyObject *roi_stats            (PyObject *self, PyObject *args); 


/*****************************************************************************
 * Python/NumPy -- C boilerplate
 *****************************************************************************/
//
//--------------------------------------------------------
// function declarations
//--------------------------------------------------------
//
static PyMethodDef methods[] = {
  { "binary_closure", binary_closure, METH_VARARGS, "Morphological closure for binary images."},
  { "discrete_histogram", discrete_histogram, METH_VARARGS, "Fast histogram for images."},
  { "discrete_log2rootk", discrete_log2rootk, METH_VARARGS, "Fast k-th root-of-2 logarithm for discrete signalsi (k integer)."},
  { "mask_laplacian", mask_laplacian, METH_VARARGS, "Compute Laplacian within each ROI ."},
  { "mask_refine", mask_refine, METH_VARARGS, "Assign CR score to each ROI"},
  { "roi_label", roi_label, METH_VARARGS, "Image labeling"},
  { "roi_stats", roi_stats, METH_VARARGS, "Statistics of the different ROIs as defined by the labeling image"},
  { NULL, NULL, 0, NULL } /* Sentinel */
};
//
//--------------------------------------------------------
// module initialization
//--------------------------------------------------------
//
PyMODINIT_FUNC initcrimg(void) {
  (void) Py_InitModule("crimg", methods);
  import_array();
}


//
//--------------------------------------------------------
// binary 4 or 8 neighbor closure
//--------------------------------------------------------
//
static PyObject *binary_closure(PyObject *self, PyObject *args) {
  PyArrayObject *py_M, *py_C;
  // Parse arguments: image, neighborhood: 4 or 8
  if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &py_M)) {
    return NULL;
  }
  const PyArray_Descr* desc = PyArray_DESCR(py_M);
  const char typecode = desc->type_num;
  if ((typecode != NPY_BOOL) && ((typecode != NPY_UINT8))) {
    PyErr_Warn(PyExc_Warning,"Data type must be numpy.uint8 or numpy.bool.");
    return NULL;
  }
  const npy_intp M = PyArray_DIM(py_M,0);
  const npy_intp N = PyArray_DIM(py_M,1);
  
  // data type ok, proceed
  // create returned mask
  py_C = (PyArrayObject*) PyArray_SimpleNew(2,PyArray_DIMS(py_M),typecode);
  npy_uint8* rM = PyArray_DATA(py_M);
  npy_intp   sM = PyArray_STRIDE(py_M,0);
  npy_uint8* rC = PyArray_DATA(py_C);
  npy_intp   sC = PyArray_STRIDE(py_C,0);
  //
  // 1.top row
  //
  // 1.1 top-left corner
  // ·#
  // #
  rC[0] = rM[0] || (rM[1] && rM[sM]); // east and south ON
  // 1.2 middle columns
  for (npy_intp j = 1; j < (N-1); ++j) {
    // #·#  or  #·  or ·#
    //  #        #     #
    rC[j] = rM[j] || (rM[sM+j] && (rM[j-1] || rM[j+1])); 
  }
  // 1.3 top-right corner
  // #·
  //  #
  rC[N-1] = rM[N-1] || (rM[N-2] && rM[sM+N-1]); // west and south ON
  //
  // advance row
  //
  rC += sM; rM += sM;
  //
  // 2. middle
  //
  for (npy_intp i = 1; i < (M-1); ++i, rC+=sC, rM+=sM) {
    // 2.1 left  coluumn
    // ·# or #   or #
    // #     ·#     ·#
    //              #
    rC[0] = rM[0] || (rM[1] && (rM[sM] || rM[-sM])); 
    // 2.2 middle columns
    //  #        #     #
    // #·#  or  #·  or ·# or #·#
    //  #        #     #      #
    for (npy_intp j = 1; j < (N-1); ++j) {
      // n + s + e +w >= 3
      if (rM[j]) {
	rC[j] = 1;
      } else {
	npy_uint8 n = rM[j-sC] > 0 ? 1 : 0;
	npy_uint8 s = rM[j+sC] > 0 ? 1 : 0;
	npy_uint8 e = rM[j+1] > 0 ? 1 : 0;
	npy_uint8 w = rM[j-1] > 0 ? 1 : 0;
	rC[j] = (( n + s + e +w ) >= 3);
      }
    }
    // 2.3 right column
    //      #   #
    // #·  #·  #·
    //  #       #
    rC[N-1] = rM[N-1] || ( rM[N-2] && (rM[N-1+sM] || rM[N-1-sM]) ); 
  }
  // last row
  //
  // 1.1 bottom-left corner
  // #
  // ·#
  rC[0] = rM[0] || (rM[1] && rM[-sM]); // east and south ON
  // 1.2 middle columns
  for (npy_intp j = 1; j < (N-1); ++j) {
    //  #        #     #
    // #·#  or  #·  or ·#
    rC[j] = rM[j] || ( rM[j-sC] && (rM[j-1] || rM[j+1]) ); 
  }
  // 1.3 bottom-right corner
  //  #    
  // #·
  rC[N-1] = rM[N-1] || (rM[N-2] && rM[N-1-sC]); // west and south ON
  //
  // finish!
  //
  return PyArray_Return(py_C);
}
//
//--------------------------------------------------------
// discrete_log2rootk
//--------------------------------------------------------
//
static void _discrete_log2rootk_8 (PyArrayObject* py_I, PyArrayObject* py_H, npy_uint k);
static void _discrete_log2rootk_16(PyArrayObject* py_I, PyArrayObject* py_H, npy_uint k);
static void _discrete_log2rootk_32(PyArrayObject* py_I, PyArrayObject* py_H, npy_uint k);
static void _discrete_log2rootk_64(PyArrayObject* py_I, PyArrayObject* py_H, npy_uint k);

static PyObject *discrete_log2rootk(PyObject *self, PyObject *args) {
  PyArrayObject *py_I, *py_H;
  npy_uint k;
  // Parse arguments: image
  if(!PyArg_ParseTuple(args, "O!I",
                       &PyArray_Type,
                       &py_I,
		       &k
		       )) {
    return NULL;
  }
  const PyArray_Descr* desc = PyArray_DESCR(py_I);
  if (desc->kind != 'u') {
    PyErr_Warn(PyExc_Warning,"Data type must be unsigned integer.");
    return NULL;
  }
  const char typecode = desc->type_num;

  py_H = (PyArrayObject*) PyArray_SimpleNew(PyArray_NDIM(py_I),PyArray_DIMS(py_I),NPY_UINT8);
  //
  // H[i,j] = [ log_2^(1/k) (x) ] = [ k log_2 x ]
  //
  switch (typecode) {
  case NPY_UINT8: case NPY_BOOL:
    _discrete_log2rootk_8(py_I, py_H, k);
    break;
  case NPY_UINT16:
    _discrete_log2rootk_16(py_I, py_H, k);
    break;
  case NPY_UINT32:
    _discrete_log2rootk_32(py_I, py_H, k);
    break;
  case NPY_UINT64:
    _discrete_log2rootk_64(py_I, py_H, k);
    break;
  default:
    PyErr_Warn(PyExc_Warning, "Only unsigned integers allowed.");
    return NULL;
  }
  return PyArray_Return(py_H);
}

static void _discrete_log2rootk_8(PyArrayObject* py_I, PyArrayObject* py_H, npy_uint k) {
  npy_uint8*      rH = PyArray_DATA(py_H);
  const npy_intp  sH = PyArray_STRIDE(py_H,0);
  npy_uint8*      rI = PyArray_DATA(py_I);
  const npy_intp  sI = PyArray_STRIDE(py_I,0);
  const npy_intp  M = PyArray_DIM(py_I,0);
  const npy_intp  N = PyArray_DIM(py_I,1);
  
  for (npy_intp i = 0; i < M; i++, rH += sH, rI += sI) {
    for (npy_intp j = 0; j < N; j++) {
      npy_uint8 lx = 0;
      npy_int64 x = rI[j];
      npy_uint kk = k;
      while( kk--) { x *= x; }
      while (x >>= 1)
	lx++;
      rH[j] = (npy_uint8) lx; // overflow must be controlled by user
    }
  }
}

static void _discrete_log2rootk_16(PyArrayObject* py_I, PyArrayObject* py_H, npy_uint k) {
  npy_uint8*      rH = PyArray_DATA(py_H);
  const npy_intp  sH = PyArray_STRIDE(py_H,0);
  npy_uint16*     rI = PyArray_DATA(py_I);
  const npy_intp  sI = PyArray_STRIDE(py_I,0)/2;
  const npy_intp  M  = PyArray_DIM(py_H,0);
  const npy_intp  N  = PyArray_DIM(py_H,1);
  for (npy_intp i = 0; i < M; i++, rH += sH, rI += sI) {
    for (npy_intp j = 0; j < N; j++) {
      npy_uint8 lx = 0;
      npy_uint64 x = rI[j];
      npy_uint kk = k;
      while( kk--) { x *= x; }
      while (x >>= 1) lx++;
      rH[j] = lx;
    }
  }
}
static void _discrete_log2rootk_32(PyArrayObject* py_I, PyArrayObject* py_H, npy_uint k) {
  npy_uint8*      rH = PyArray_DATA(py_H);
  const npy_intp  sH = PyArray_STRIDE(py_H,0);
  npy_uint32*     rI = PyArray_DATA(py_I);
  const npy_intp  sI = PyArray_STRIDE(py_I,0)/4;
  const npy_intp  M  = PyArray_DIM(py_I,0);
  const npy_intp  N  = PyArray_DIM(py_I,1);
  
  for (npy_intp i = 0; i < M; i++, rH += sH, rI += sI) {
    for (npy_intp j = 0; j < N; j++) {
      npy_uint8 lx = 0;
      npy_uint64 x = rI[j];
      npy_uint kk = k;
      while( kk--) { x *= x; }
      while (x >>= 1) lx++;
      rH[j] = lx;
    }
  }
}
static void _discrete_log2rootk_64(PyArrayObject* py_I, PyArrayObject* py_H, npy_uint k) {
  npy_uint8*      rH = PyArray_DATA(py_H);
  const npy_uint8 sH = PyArray_STRIDE(py_H,0);
  npy_uint64*     rI = PyArray_DATA(py_I);
  const npy_uint8 sI = PyArray_STRIDE(py_I,0);
  const npy_intp  M  = PyArray_DIM(py_I,0);
  const npy_intp  N  = PyArray_DIM(py_I,1);
  
  for (npy_intp i = 0; i < M; i++, rH += sH, rI += sI) {
    for (npy_intp j = 0; j < N; j++) {
      npy_uint8 lx = 0;
      npy_uint64 x = rI[j];
      npy_uint kk = k;
      while( kk--) { x *= x; }
      while (x >>= 1) lx++;
      rH[j] = lx;
    }
  }
}

//
//--------------------------------------------------------
// discrete_histogram
//--------------------------------------------------------
//
static PyObject *discrete_histogram(PyObject *self, PyObject *args) {
  PyArrayObject *py_I, *py_H;
  // Parse arguments: image
  if(!PyArg_ParseTuple(args, "O!",
                       &PyArray_Type,
                       &py_I
		       )) {
    return NULL;
  }
  const PyArray_Descr* desc = PyArray_DESCR(py_I);
  if (desc->kind != 'u') {
    PyErr_Warn(PyExc_Warning,"Data type must be 8 or 16 bit unsigned integer.");
    return NULL;
  }
  const char typecode = desc->type_num;
  // histograms for 8 or 16 bit integers; larger integers are squashed down
  npy_intp dim;
  switch (typecode) {
  case NPY_UINT8: case NPY_BOOL:
    dim = 1<<8;
    break;
  case NPY_UINT16:
    dim = 1<< 16;
      break;
  default:
      PyErr_Warn(PyExc_Warning,"Only 8 or 16 bit integers allowed.");
      return NULL;
  }
  py_H = (PyArrayObject*) PyArray_SimpleNew(1,&dim,NPY_INT64);
  //
  // fill in the histogram
  //
  PyArray_FILLWBYTE(py_H,0);
  PyArrayIterObject *iter = (PyArrayIterObject *)PyArray_IterNew((PyObject*)py_I);
  if (iter == NULL) {
    PyErr_Warn(PyExc_Warning,"Failed creating iterator??.");
    return NULL;
  }
  if (typecode == NPY_UINT16) { 
    while (PyArray_ITER_NOTDONE(iter)) {
      const npy_intp x = *((npy_uint16*)PyArray_ITER_DATA(iter));
      (*(npy_int64*)PyArray_GETPTR1(py_H, x))++;
      PyArray_ITER_NEXT(iter);
    }
  } else {
    while (PyArray_ITER_NOTDONE(iter)) {
      const npy_intp x = *((npy_uint8*)PyArray_ITER_DATA(iter));
      (*(npy_int64*)PyArray_GETPTR1(py_H, x))++;
      PyArray_ITER_NEXT(iter);
    }
  }
  return PyArray_Return(py_H);
}

//
//--------------------------------------------------------
// mask_laplacian
//--------------------------------------------------------
//
// Receives an image, a ROI mask and replaces it in-place with
// the absolute value of the Laplacian
//
//
static void _mask_laplacian_8(PyArrayObject* py_I, PyArrayObject* py_M, PyArrayObject* py_L);
static void _mask_laplacian_16(PyArrayObject* py_I, PyArrayObject* py_M, PyArrayObject* py_L);
static void _mask_laplacian_32(PyArrayObject* py_I, PyArrayObject* py_M, PyArrayObject* py_L);
static void _mask_laplacian_64(PyArrayObject* py_I, PyArrayObject* py_M, PyArrayObject* py_L);

static PyObject *mask_laplacian(PyObject *self, PyObject *args) {
  PyArrayObject *py_I, *py_M, *py_L;
  if(!PyArg_ParseTuple(args, "O!O!",
		       &PyArray_Type, &py_I,
		       &PyArray_Type, &py_M)) {
    return NULL;
  }

  //
  // type checking
  //
  // mask:
  PyArray_Descr* desc = PyArray_DESCR(py_M);
  desc = PyArray_DESCR(py_M);
  char typecode = desc->type_num;
  if ((typecode != NPY_BOOL) && ((typecode != NPY_UINT8))) {
    PyErr_Warn(PyExc_Warning,"Mask must be numpy.uint8 or numpy.bool.");
    return NULL;
  }
  // image:
  desc = PyArray_DESCR(py_I);
  typecode = desc->type_num;

  py_L = (PyArrayObject*) PyArray_SimpleNew(PyArray_NDIM(py_I),PyArray_DIMS(py_I),NPY_UINT16);

  switch (typecode) {
  case NPY_UINT8: case NPY_BOOL:
    _mask_laplacian_8(py_I, py_M, py_L);
    break;
  case NPY_UINT16:
    _mask_laplacian_16(py_I, py_M, py_L);
    break;
  case NPY_UINT32:
    _mask_laplacian_32(py_I, py_M, py_L);
    break;
  case NPY_UINT64:
    _mask_laplacian_64(py_I, py_M, py_L);
    break;
  default:
    PyErr_Warn(PyExc_Warning, "Only unsigned integers allowed.");
    return NULL;
  }
  return PyArray_Return(py_L);  
}

static void _mask_laplacian_8(PyArrayObject* py_I, PyArrayObject* py_M, PyArrayObject* py_L) {
  const npy_intp M = PyArray_DIM(py_I,0);
  const npy_intp N = PyArray_DIM(py_I,1);
  npy_uint8*     rM = PyArray_DATA(py_M);
  npy_intp       sM = PyArray_STRIDE(py_M,0);
  npy_uint8*     rI = PyArray_DATA(py_I);
  npy_intp       sI = PyArray_STRIDE(py_I,0);
  npy_uint16*    rL = PyArray_DATA(py_L);
  npy_intp       sL = PyArray_STRIDE(py_L,0)/2;
  for (npy_intp i = 0; i < M; i++) {
    npy_uint8* rIs, *rIn;
    if (i > 0) {
      rIn = rI-sI;
    } else {
      rIn = rI+sI; // reflected
    }
    if (i < (M-1)) {
      rIs = rI + sI; 
    } else {
      rIs = rI - sI; // reflected
    }
    for (npy_intp j = 0; j < N; j++) {
      if (!rM[j]) {
	rL[j] = 0;
      } else {
	const npy_intp jw = (j > 0) ? j-1 : 1;
	const npy_intp je = (j < (N-1)) ? j+1 : N-2;
	const npy_int64 n = rIn[j];
	const npy_int64 s = rIs[j];
	const npy_int64 w = rI[jw];
	const npy_int64 e = rI[je];
	const npy_int64 x = rI[j];	
	npy_int64 Lij = (x<<2) - s - n - e - w;
	if (Lij < 0) Lij = -Lij;
	rL[j] = (npy_uint16) Lij;
      }
    }
    rM += sM; rI += sI; rL += sL;    
  }
}

static void _mask_laplacian_16(PyArrayObject* py_I, PyArrayObject* py_M, PyArrayObject* py_L) {
}

static void _mask_laplacian_32(PyArrayObject* py_I, PyArrayObject* py_M, PyArrayObject* py_L) {
}

static void _mask_laplacian_64(PyArrayObject* py_I, PyArrayObject* py_M, PyArrayObject* py_L) {
}


//
//--------------------------------------------------------
// mask_refine
//--------------------------------------------------------
//
static void  _mask_refine_(PyArrayObject*, PyArrayObject*, PyArrayObject*);

static PyObject *mask_refine(PyObject *self, PyObject *args) {
  PyArrayObject *py_Lab, *py_Lap, *py_Mask;
  // Parse arguments: input label image, input laplacian of masks, output refined mask
  if(!PyArg_ParseTuple(args, "O!O!O!",
                       &PyArray_Type, &py_Lab,
                       &PyArray_Type, &py_Lap)) {
    return NULL;
 }
  py_Mask = (PyArrayObject*) PyArray_SimpleNew(2,PyArray_DIMS(py_Lab),NPY_UINT8); 
  PyArray_FILLWBYTE(py_Mask,0);
  _mask_refine_(py_Lab,py_Lap,py_Mask);
  return PyArray_Return(py_Mask);  
}

static void  _mask_refine_(PyArrayObject* py_Lab,
			   PyArrayObject* py_Lap,
			   PyArrayObject* py_Mask) {
}

//
//--------------------------------------------------------
// labeling
//--------------------------------------------------------
//
static void _replace_label_(PyArrayObject* pL, npy_intp a, npy_intp b, npy_intp lasti, npy_intp lastj);

//static void _compact_label_(PyArrayObject* pL);
static void _roi_label_(PyArrayObject* pM, PyArrayObject* pL);

static PyObject *roi_label(PyObject *self, PyObject *args) {
  PyArrayObject *py_M, *py_L;
  // Parse arguments: input mask
  if(!PyArg_ParseTuple(args, "O!",
                       &PyArray_Type, &py_M)) {
    return NULL;
 }
  // PENDING: check that mask is bool!
  py_L = (PyArrayObject*) PyArray_SimpleNew(2,PyArray_DIMS(py_M),NPY_UINT32); 
  PyArray_FILLWBYTE(py_L,0);
  _roi_label_(py_M,py_L);
  return PyArray_Return(py_L);  
}

/*------------------------------------------------------------------------*/

void _roi_label_(PyArrayObject* pM, PyArrayObject* pL) {
  const npy_intp hsize = PyArray_DIM(pM,1);
  const npy_intp vsize = PyArray_DIM(pM,0);
  const npy_intp mask_vstride = PyArray_STRIDE(pM,0);
  const npy_intp label_vstride = PyArray_STRIDE(pL,0)/4;
  
  const npy_uint8 *mask_row = PyArray_DATA(pM);
  npy_uint32 *label_data = PyArray_DATA(pL);
  npy_uint32 *label_row = label_data;
  npy_uint32 *label_prev_row = label_row - label_vstride;
  
  npy_uint32 L = 1; // current label number
  register int i,j;
  for (i = 0; i < vsize; ++i, label_row += label_vstride, label_prev_row += label_vstride, mask_row += mask_vstride) {
    for (j = 0; j < hsize; ++j) {
      if (! mask_row[j]) {
	continue;
      }
      
      npy_uint32 Ln = (i > 0) ? label_prev_row[j] : 0; // mask value to the north
      npy_uint32 Lw = (j > 0) ? label_row[j-1] : 0; // mask value to the west
      if (Ln == 0) {
	if (Lw == 0) { // no ROI to either north or west
	  label_row[j] = L++; // create new (tentative) label at this point
	} else {  // ROI to the west
	  label_row[j] = Lw;
	}	
      } else { // ROI to the north
	if (Lw == 0) { // no ROI to the west; propagate northern one
	  label_row[j] = Ln;
	}  else { 
	  label_row[j] = Lw; 
	  if (Ln != Lw) { // different labels to N and W! Resolve 
	    _replace_label_(pL,Lw,Ln,i,j); 
	  }
	}
      } 
    } /* j: dentro de cada fila */
  } /* i: para cada fila */
}

//--------------------------------------------------------

void _replace_label_(PyArrayObject* pL, npy_intp a, npy_intp b, npy_intp lasti, npy_intp lastj) {
  const npy_intp hsize = PyArray_DIM(pL,1);
  const npy_intp label_vstride = PyArray_STRIDE(pL,0)/4;
  npy_uint32 *label_data = PyArray_DATA(pL);
  npy_uint32 *label_row = label_data+ label_vstride*lasti;
  //  printf("replace %lu by %lu starting at %lu %lu\n",a,b,lasti,lastj);
  npy_intp i,j;
  for (i = lasti; i; i--, label_row -= label_vstride ) {
    char any_replacement_in_this_row = 0;
    for (j = hsize-1 ; j ; j-- ) {
      if (label_row[j] == a) {
	label_row[j] = b;
	any_replacement_in_this_row = 1;
      }
    }
    if (!any_replacement_in_this_row)
      break;
  } 
}



//--------------------------------------------------------
// roi_stats
//
static void _roi_stats_8 (PyArrayObject* py_Lab, PyArrayObject* py_Lap, PyArrayObject* py_S);
static void _roi_stats_16(PyArrayObject* py_Lab, PyArrayObject* py_Lap, PyArrayObject* py_S);
static void _roi_stats_32(PyArrayObject* py_Lab, PyArrayObject* py_Lap, PyArrayObject* py_S);
static void _roi_stats_64(PyArrayObject* py_Lab, PyArrayObject* py_Lap, PyArrayObject* py_S);

static PyObject *roi_stats(PyObject *self, PyObject *args) {
  PyArrayObject *py_Lap, *py_Lab, *py_S;
  if(!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &py_Lab,  &PyArray_Type, &py_Lap)) {
    return NULL;
  }
  //
  // type checking
  //
  // label image:
  PyArray_Descr* desc = PyArray_DESCR(py_Lab);
  desc = PyArray_DESCR(py_Lab);
  char typecode = desc->type_num;
  if (typecode != NPY_UINT16) {
    PyErr_Warn(PyExc_Warning,"Label must be numpy.uint16.");
    return NULL;
  }
  // Laplacian image:
  desc = PyArray_DESCR(py_Lap);
  typecode = desc->type_num;
  //
  // returned array has size NL+1 x 6
  // where the columns are:
  // 0) mean
  // 1) p00 = min
  // 2) p10
  // 3) p25
  // 4) p50 = med
  // 5) p75
  // 6) p90
  // 7) p100 = max
  //
  // row 0 corresponds to global statistics:
  // 0) global mean
  // 1) global  min
  // 2) median of the p10 of all ROIs
  // 3) median of the p25s
  // 4-7) etc.
  // rows 1 to N+1 correspond to the N ROI's
  //
  //
  // first we have to find N
  //
  npy_intp N = 0;
  PyArrayIterObject *iter = (PyArrayIterObject *)PyArray_IterNew((PyObject*)py_Lab);
  while (PyArray_ITER_NOTDONE(iter)) {
    const npy_intp l = *((npy_uint16*)PyArray_ITER_DATA(iter));
    if (l > N)
      N = l;
    PyArray_ITER_NEXT(iter);
  }
  
  npy_intp dims[2];
  dims[0] = N+1;
  dims[1] = 8;
  py_S = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_UINT16);

  switch (typecode) {
  case NPY_UINT8: case NPY_BOOL:
    _roi_stats_8(py_Lab, py_Lap, py_S);
    break;
  case NPY_UINT16:
    _roi_stats_16(py_Lab, py_Lap, py_S);
    break;
  case NPY_UINT32:
    _roi_stats_32(py_Lab, py_Lap, py_S);
    break;
  case NPY_UINT64:
    _roi_stats_64(py_Lab, py_Lap, py_S);
    break;
  default:
    PyErr_Warn(PyExc_Warning, "Only unsigned integers allowed.");
    return NULL;
  }
  
  return PyArray_Return(py_S);  
}

static void _roi_stats_8(PyArrayObject* py_Lab, PyArrayObject* py_Lap, PyArrayObject* py_S) {
  const npy_intp M = PyArray_DIM(py_Lab,0);
  const npy_intp N = PyArray_DIM(py_Lab,1);
  npy_uint8 *    rLap = PyArray_DATA(py_Lap);
  npy_intp       sLap = PyArray_STRIDE(py_Lap,0);
  npy_uint32*    rLab = PyArray_DATA(py_Lab);
  npy_intp       sLab = PyArray_STRIDE(py_Lab,0);
  npy_uint16*    rS = PyArray_DATA(py_S);
  npy_intp       sS = PyArray_STRIDE(py_S,0)/2;
  for (npy_intp i = 0; i < M; i++) {
    for (npy_intp j = 0; j < N; j++) {
    }
    rLap += sLap; rLab += sLab;
  }
}

static void _roi_stats_16(PyArrayObject* py_I, PyArrayObject* py_M, PyArrayObject* py_L) {
}

static void _roi_stats_32(PyArrayObject* py_I, PyArrayObject* py_M, PyArrayObject* py_L) {
}

static void _roi_stats_64(PyArrayObject* py_I, PyArrayObject* py_M, PyArrayObject* py_L) {
}

//--------------------------------------------------------
//
// if region merging occurs, the final labels are not consecutive
// here we re-label the image so that labels are consecutive
//
void _compact_label_(PyArrayObject* pL) {
}

