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

#define PARSE_POSITIVE_SIZE_T(py_value, c_value, name)       \
    do {                                                     \
        if ((py_value) <= 0) {                               \
            PyErr_Format(PyExc_TypeError,                    \
                         "%s must be a positive integer",    \
                         (name));                            \
            return NULL;                                     \
        }                                                    \
        (c_value) = (size_t)(py_value);                      \
    } while (0)


typedef int bool;
#define TRUE 1
#define FALSE 0

// Helper functions
static double * get_numpy_data_safe(PyObject* obj, const char * name, const bool c_cont, const bool f_cont) {
  if (!PyArray_Check(obj)) {
    PyErr_Format(PyExc_RuntimeError, "%s to be a expected a numpy array", name);
    return NULL;
  }
  const PyArrayObject* pyarr = (const PyArrayObject*) obj;
  if (PyArray_TYPE(pyarr) != NPY_DOUBLE) {
    PyErr_Format(PyExc_RuntimeError, "%s expected to be a numpy double-typed array", name);
    return NULL;
  }
  if (c_cont && !PyArray_IS_C_CONTIGUOUS(pyarr)) {
    PyErr_Format(PyExc_RuntimeError, "%s expected to be a C contiguous array", name); 
    return NULL;
  }
  if (f_cont && !PyArray_IS_F_CONTIGUOUS(pyarr)) {
    PyErr_Format(PyExc_RuntimeError, "%s expected to be a F contiguous array", name); 
    return NULL;
  }
  double* lams = PyArray_DATA(pyarr);
  return lams;
}

// Bindings

static PyObject* py_lw_analytical(PyObject* self, PyObject* args) {
  PyObject *lams_obj;
  Py_ssize_t py_n, py_p;
  size_t n, p;
  double eps;

  // Extract args
  if (!PyArg_ParseTuple(
    args, "Onnd",
    &lams_obj, &py_n, &py_p, &eps
  )) return NULL;

  // Checks for lams
  const double* const lams = get_numpy_data_safe(lams_obj, "lams", TRUE, FALSE);
  if (!lams) return NULL;

  // Checks for shape
  PARSE_POSITIVE_SIZE_T(py_n, n, "n");
  PARSE_POSITIVE_SIZE_T(py_p, p, "p");

  // Checks for eps
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
  PyObject *sample_cov_obj;
  Py_ssize_t py_n, py_p;
  size_t n, p;

  // Extract args
  if (!PyArg_ParseTuple(
    args, "Onn",
    &sample_cov_obj, &py_n, &py_p
  )) return NULL;

  // Checks for sample_cov
  const double* const sample_cov = get_numpy_data_safe(sample_cov_obj, "sample_cov", TRUE, FALSE);
  if (!sample_cov) return NULL;

  // Checks for shape
  PARSE_POSITIVE_SIZE_T(py_n, n, "n");
  PARSE_POSITIVE_SIZE_T(py_p, p, "p");

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


static PyObject* py_lw_linear(PyObject* self, PyObject* args) {
  PyObject *data_obj;
  Py_ssize_t py_n, py_p;
  size_t n, p;

  // Extract args
  if (!PyArg_ParseTuple(
    args, "Onn",
    &data_obj, &py_n, &py_p
  )) return NULL;

  // Checks for data
  const double* const data = get_numpy_data_safe(data_obj, "data", TRUE, FALSE);
  if (!data) return NULL;

  // Checks for positive numbers
  PARSE_POSITIVE_SIZE_T(py_n, n, "n");
  PARSE_POSITIVE_SIZE_T(py_p, p, "p");

  // Create output object
  npy_intp dims[2] = {p, p};
  PyObject * const sample_cov_star_obj = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  if (!sample_cov_star_obj) return NULL;
  PyArrayObject* sample_cov_star_pyarr = (PyArrayObject*) sample_cov_star_obj;
  double * sample_cov_star = PyArray_DATA(sample_cov_star_pyarr);

  // Execute C code
  C_LWLinear(data, sample_cov_star, n, p);

  // Return object
  return sample_cov_star_obj;
}


static PyMethodDef Methods[] = {
    {"py_lw_analytical", py_lw_analytical, METH_VARARGS, "Performs LW Analytical Shrinkage"},
    {"py_lw_linear", py_lw_linear, METH_VARARGS, "Performs LW Linear Shrinkage"},
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
