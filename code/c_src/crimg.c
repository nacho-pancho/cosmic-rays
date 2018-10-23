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

/// Python adaptors
static PyObject *discrete_histogram   (PyObject* self, PyObject* args);
static PyObject *discrete_log2rootk   (PyObject *self, PyObject *args);
static PyObject *mask_laplacian       (PyObject *self, PyObject *args); 
static PyObject *mask_refine          (PyObject *self, PyObject *args); 
static PyObject *label                (PyObject *self, PyObject *args); 


/*****************************************************************************
 * Python/NumPy -- C boilerplate
 *****************************************************************************/
//
//--------------------------------------------------------
// function declarations
//--------------------------------------------------------
//
static PyMethodDef methods[] = {
  { "discrete_histogram", discrete_histogram, METH_VARARGS, "Fast histogram for images."},
  { "discrete_log2rootk", discrete_log2rootk, METH_VARARGS, "Fast k-th root-of-2 logarithm for discrete signalsi (k integer)."},
  { "mask_laplacian", mask_laplacian, METH_VARARGS, "Compute Laplacian within each ROI ."},
  { "mask_refine", mask_refine, METH_VARARGS, "Assign CR score to each ROI"},
  { "label", label, METH_VARARGS, "Image labeling"},
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
static void _mask_laplacian_(PyArrayObject* py_I, PyArrayObject* py_M, PyArrayObject* py_L);

static PyObject *mask_laplacian(PyObject *self, PyObject *args) {
  PyArrayObject *py_I, *py_M, *py_L;
  if(!PyArg_ParseTuple(args, "O!O!",
		       &PyArray_Type, &py_I,
		       &PyArray_Type, &py_M)) {
    return NULL;
  }
  py_L = (PyArrayObject*) PyArray_SimpleNew(2,PyArray_DIMS(py_I),NPY_UINT16);
  _mask_laplacian_(py_I,py_M,py_L);
  return PyArray_Return(py_L);  
}

static void _mask_laplacian_(PyArrayObject* py_I, PyArrayObject* py_M, PyArrayObject* py_L) {
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
static void _label_(PyArrayObject* pM, PyArrayObject* pL);

static PyObject *label(PyObject *self, PyObject *args) {
  PyArrayObject *py_M, *py_L;
  // Parse arguments: input mask
  if(!PyArg_ParseTuple(args, "O!",
                       &PyArray_Type, &py_M)) {
    return NULL;
 }
  // PENDING: check that mask is bool!
  py_L = (PyArrayObject*) PyArray_SimpleNew(2,PyArray_DIMS(py_M),NPY_UINT32); 
  PyArray_FILLWBYTE(py_L,0);
  _label_(py_M,py_L);
  return PyArray_Return(py_L);  
}

/*------------------------------------------------------------------------*/

void _label_(PyArrayObject* pM, PyArrayObject* pL) {
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
	    _replace_label_(pL,Ln,Lw,i,j); 
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
//
// if region merging occurs, the final labels are not consecutive
// here we re-label the image so that labels are consecutive
//
//void _compact_label_(PyArrayObject* pL) {
//}

