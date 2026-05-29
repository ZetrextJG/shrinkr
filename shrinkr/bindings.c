#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "c_shrinkr.h"

// PyArray_* functions:
#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <numpy/arrayscalars.h>
#include <numpy/ufuncobject.h>

static PyObject* py_add(PyObject* self, PyObject* args) {
    int a, b;

    if (!PyArg_ParseTuple(args, "ii", &a, &b))
        return NULL;

    return PyLong_FromLong(add(a, b));
}

static PyMethodDef Methods[] = {
    {"add", py_add, METH_VARARGS, "Add two integers"},
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
    return PyModule_Create(&module);
}
