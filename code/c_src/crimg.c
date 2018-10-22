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
static PyObject *imghist              (PyObject* self, PyObject* args);
static PyObject *roilap               (PyObject *self, PyObject *args); 
static PyObject *roiscore             (PyObject *self, PyObject *args); 
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
  { "imghist", imghist, METH_VARARGS, "Fast histogram for images."},
  { "roilap", roilap, METH_VARARGS, "Compute Laplacian within each ROI ."},
  { "roiscore", roiscore, METH_VARARGS, "Assign CR score to each ROI"},
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
// imghist
//--------------------------------------------------------
//
static PyObject *imghist(PyObject *self, PyObject *args) {
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
// roilap
//--------------------------------------------------------
//
static PyObject *roilap(PyObject *self, PyObject *args) {
  PyArrayObject *py_I, *py_P;
  npy_int64 M, N, w, s;
  // Parse arguments. 
  if(!PyArg_ParseTuple(args, "O!O!",
		       &PyArray_Type, &py_I, &w, &s)) {
    return NULL;
  }
#if 0
  py_P = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE); 
  //
  // copy padded image
  //
  for (npy_int64 i = 0; i < M2; i++) {
    for (npy_int64 j = 0; j < N2; j++) {
      *((npy_double*)PyArray_GETPTR2(py_P,i,j)) = *(npy_double*)PyArray_GETPTR2(py_I, UCLIP(i,M), UCLIP(j,N) );
    }
  }
  return PyArray_Return(py_P);  
#endif
}

//
//--------------------------------------------------------
// roiscore
//--------------------------------------------------------
//
static PyObject *roiscore(PyObject *self, PyObject *args) {
  PyArrayObject *py_P, *py_I, *py_R;
  npy_int64 M, N, w, s;
  // Parse arguments: input image, input ROI labeling, output score for each ROI
  if(!PyArg_ParseTuple(args, "O!O!O!",
                       &PyArray_Type, &py_P,
                       &PyArray_Type, &py_R)) {
    return NULL;
 }
#if 0
  py_I = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE); 
  PyArray_FILLWBYTE(py_I,0);
  _stitch_(py_P,&map,py_I,py_R);
  return PyArray_Return(py_I);  
#endif
}



//
//--------------------------------------------------------
// labeling
//--------------------------------------------------------
//
static void _replace_label_(PyArrayObject* pI, npy_intp a, npy_intp b, npy_intp maxp);
static void _replace_label_(PyArrayObject* pI, npy_intp a, npy_intp b, npy_intp maxp);
static void _label_(const PyArrayObject* pM, PyArrayObject* pL);

static PyObject *label(PyObject *self, PyObject *args) {
  PyArrayObject *py_P, *py_I, *py_R;
  npy_int64 M, N, w, s;
  // Parse arguments: input mask, output labeling
  if(!PyArg_ParseTuple(args, "O!O!",
                       &PyArray_Type, &py_P,
                       &PyArray_Type, &py_R)) {
    return NULL;
 }
#if 0
  py_I = (PyArrayObject*) PyArray_SimpleNew(2,dims,NPY_DOUBLE); 
  PyArray_FILLWBYTE(py_I,0);
  _stitch_(py_P,&map,py_I,py_R);
  return PyArray_Return(py_I);  
#endif
}

//--------------------------------------------------------

void _replace_label_(PyArrayObject* pI, npy_intp a, npy_intp b, npy_intp maxp) {
#if 0 
// arreglar para PyArray
      	const int N = pI->ancho;
  const int M = pI->alto;  
  npy_intp *pi = pI->pixels;
  register int i,j,k,ult_reemplazo = maxp;
  if ((maxp >= M*N) || (maxp == 0))
    maxp = M*N-1;
  i = maxp / N;
  j = maxp % N;
  k = maxp;
  while ( i >= 0 ) {
    while ( j >= 0 ) {
      if (pi[k] == a) {
	pi[k] = b;
	ult_reemplazo = k;
      }
      j--;
      k--;
    }
    j = N-1;
    i--;
    if ((ult_reemplazo - k) > pI->ancho) {
      break;
    }
  }
#endif
}

/*------------------------------------------------------------------------*/

void _label_(const PyArrayObject* pM, PyArrayObject* pL) {
#if 0
  const int ancho = pG->ancho;
  const int alto  = pG->alto;
  register int i,j,k;
  const npy_intp *pg = pG->pixels;
  npy_intp *pe = pE->pixels;
  npy_intp *pen = pE->pixels - ancho; /* fila anterior */
  int L = 0; /* etiqueta */
  for (k = 0, i = 0; i < alto; ++i) {
    /* printf("%6d/%6d L=%6d\n",i,alto,L); */
    for (j = 0; j < ancho; ++j, ++k) {
      if (pg[k] > u) { 
	pe[k] = 0;
	continue;
      }
      int n = (i > 0) ? pen[k] : 0;
      int w = (j > 0) ? pe[k-1] : 0;
      if (n == 0) {
	if (w == 0) { /* ambos borde */
	  pe[k] = ++L; /* nueva etiqueta */
	} else { /* oeste no era borde */
	  pe[k] = w;
	}	
      } else { /* n no es borde */
	if (w == 0) {
	  pe[k] = n;
	}  else { /* etiquetas distintas */
	  pe[k] = w;
	  if (n != w) {
	    reemplazar_etiqueta2(pE,n,w,k); 
	  }
	}
      } 
    } /* j: dentro de cada fila */
  } /* i: para cada fila */
  pE->valor_maximo = L;
#endif
}



