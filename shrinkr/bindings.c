#include <Python.h>
#include "c_shrinkr.h"
#include "methodobject.h"
#include "pyerrors.h"

// PyArray_* functions:
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayscalars.h>
#include <numpy/ufuncobject.h>


static PyObject* py_lw_analytical(PyObject* self, PyObject* args) {
  if (PyTuple_Size(args) != 4)
    return PyErr_Format(PyExc_RuntimeError, "expected 4 arguments");

  // Unpack args
  PyObject* lams_obj = PyTuple_GetItem(args, 0);
  if (!lams_obj) return NULL;
  PyObject* n_obj = PyTuple_GetItem(args, 1);
  if (!n_obj) return NULL;
  PyObject* p_obj = PyTuple_GetItem(args, 2);
  if (!p_obj) return NULL;
  PyObject* eps_obj = PyTuple_GetItem(args, 3);
  if (!eps_obj) return NULL;

  // Checks for lams
  if (!PyArray_Check(lams_obj)) {
    PyErr_SetString(PyExc_RuntimeError, "expected a numpy array");
    return NULL;
  }
  const PyArrayObject* lams_pyarr = (const PyArrayObject*) lams_obj;
  if (PyArray_TYPE(lams_pyarr) != NPY_DOUBLE) {
    PyErr_SetString(PyExc_RuntimeError, "expected a numpy double-typed array");
    return NULL;
  }
  if (!PyArray_IS_C_CONTIGUOUS(lams_pyarr)) {
    PyErr_Format(PyExc_RuntimeError, "expected a contiguous array"); 
    return NULL;
  }
  const double* const lams = PyArray_DATA(lams_pyarr);

  // Checks for n
  if (!PyNumber_Check(n_obj)) {
    PyErr_SetString(PyExc_TypeError, "n must be positive integer");
    return NULL;
  }
  size_t n = PyNumber_AsSsize_t(n_obj, NULL);
  if (n == -1 && PyErr_Occurred())  return NULL;

  // Checks for p
  if (!PyNumber_Check(p_obj)) { 
    PyErr_SetString(PyExc_TypeError, "p must be positive integer");
    return NULL;
  }
  size_t p = PyNumber_AsSsize_t(p_obj, NULL);
  if (p == -1 && PyErr_Occurred())  return NULL;

  // Checks for eps
  if (!PyFloat_Check(eps_obj)) {
    PyErr_SetString(PyExc_TypeError, "eps must be a positive float");
    return NULL;
  }
  double eps = PyFloat_AsDouble(eps_obj);
  if (PyErr_Occurred()) return NULL;
  if (eps <= 0) {
      PyErr_SetString(PyExc_ValueError, "eps must be a positive float");
      return NULL;
  }

  // Create output object
  npy_intp dims[1] = {p};
  PyObject * const lams_star_obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  if (!lams_star_obj) return NULL;
  PyArrayObject* lams_star_pyarr = (PyArrayObject*) lams_star_obj;
  double * lams_star = PyArray_DATA(lams_star_pyarr);

  // Execute C code
  C_LWAnalytical(lams, lams_star, n, p, eps);

  // Return object
  return lams_star_obj;
}

static PyObject* py_oas(PyObject* self, PyObject* args) {
  if (PyTuple_Size(args) != 3)
    return PyErr_Format(PyExc_RuntimeError, "expected 3 arguments");

  // Unpack args
  PyObject* sample_cov_obj = PyTuple_GetItem(args, 0);
  if (!sample_cov_obj) return NULL;
  PyObject* n_obj = PyTuple_GetItem(args, 1);
  if (!n_obj) return NULL;
  PyObject* p_obj = PyTuple_GetItem(args, 2);
  if (!p_obj) return NULL;

  // Checks for sample_cov
  if (!PyArray_Check(sample_cov_obj)) {
    PyErr_SetString(PyExc_RuntimeError, "expected a numpy matrix");
    return NULL;
  }
  const PyArrayObject* sample_cov_pyarr = (const PyArrayObject*) sample_cov_obj;
  if (PyArray_TYPE(sample_cov_pyarr) != NPY_DOUBLE) {
    PyErr_SetString(PyExc_RuntimeError, "expected a numpy double-typed matrix");
    return NULL;
  }
  const double* const sample_cov = PyArray_DATA(sample_cov_pyarr);

  // Checks for n
  if (!PyNumber_Check(n_obj)) {
    PyErr_SetString(PyExc_TypeError, "n must be positive integer");
    return NULL;
  }
  size_t n = PyNumber_AsSsize_t(n_obj, NULL);
  if (n == -1 && PyErr_Occurred())  return NULL;

  // Checks for p
  if (!PyNumber_Check(p_obj)) { 
    PyErr_SetString(PyExc_TypeError, "p must be positive integer");
    return NULL;
  }
  size_t p = PyNumber_AsSsize_t(p_obj, NULL);
  if (p == -1 && PyErr_Occurred())  return NULL;

  // Create output object
  npy_intp dims[2] = {p, p};
  // TODO: Make sure that sample_cov_obj and sample_cov_star_obj
  // have the same structure in terms of the memory layout. 
  // Or enforce the C contiguous on sample_cov
  PyObject * const sample_cov_star_obj = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  if (!sample_cov_star_obj) return NULL;
  PyArrayObject* sample_cov_star_pyarr = (PyArrayObject*) sample_cov_star_obj;
  double * sample_cov_star = PyArray_DATA(sample_cov_star_pyarr);

  // Execute C code
  C_OAS(sample_cov, sample_cov_star, n, p);

  // Return object
  return sample_cov_star_obj;
}

static PyMethodDef Methods[] = {
    {"py_lw_analytical", py_lw_analytical, METH_VARARGS, "Performs LW Analytical Shrinkage"},
    {"py_oas", py_oas, METH_VARARGS, "Performs (OAS) Oracle Approximating Shrinkage"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    NULL,
    -1,
    Methods
};

PyMODINIT_FUNC PyInit__native(void) {
    import_array();

    return PyModule_Create(&module);
}
