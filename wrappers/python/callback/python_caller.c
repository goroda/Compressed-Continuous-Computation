#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <stdio.h>
#include <assert.h>

#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"

struct Obj
{
    void * f;
    void * params;
};

void assign_pointer(struct Obj * obj, PyObject * f, PyObject * params)   
{
    obj->f = (void*)f;
    obj->params = (void*)params; 
}

struct Obj * obj_alloc(void)
{
    struct Obj * obj = malloc(sizeof(struct Obj));
    assert (obj != NULL);
    return obj;
};

void obj_free(struct Obj ** obj){
    free(*obj);
    *obj = NULL;
}

void destruct_obj(PyObject * objin)
{
    void* objPtr = PyCapsule_GetPointer(objin,NULL);
    struct Obj* obj = (struct Obj*)objPtr;
    obj_free(&obj);
}

static PyObject* alloc_cobj( PyObject* self, PyObject* args ) {
    /* import_array(); */
    struct Obj * obj = obj_alloc();
    PyObject* pyObj = PyCapsule_New((void*)obj,NULL,&destruct_obj);
    return pyObj;
}

static PyObject* assign( PyObject* self, PyObject* args ) {
    PyObject* pyObj;
    PyObject* pyF;
    PyObject* pyParams;

    if(!PyArg_ParseTuple(args,"OOO",&pyObj, &pyF,&pyParams,NULL)) return NULL;

    void* objPtr = PyCapsule_GetPointer(pyObj,NULL);
    struct Obj* obj = (struct Obj*)objPtr;
    assign_pointer(obj,pyF,pyParams);
    return Py_None;
}


double eval( double a, double b, PyObject* pyFunction, PyObject* pyParams ) {
    PyObject* pyA = PyFloat_FromDouble(a);
    PyObject* pyB = PyFloat_FromDouble(b);

    PyObject* pyResult = PyObject_CallFunctionObjArgs(pyFunction,pyA,pyB,pyParams,NULL);

    double result = PyFloat_AsDouble(pyResult);

    Py_XDECREF(pyA);
    Py_XDECREF(pyB);
    Py_XDECREF(pyResult);

    return result;
}

int eval_arr(size_t N, size_t dim, const double * x, double * out, PyObject* pyFunction, PyObject* pyParams ) {

    long int dims[2];
    dims[0] = N;
    dims[1] = dim;
        
    PyObject* pyX = PyArray_SimpleNewFromData(2,dims,NPY_DOUBLE,(void*)x);

    PyObject* pyResult = PyObject_CallFunctionObjArgs(pyFunction,pyX,pyParams,NULL);

    PyArrayObject * arr = (PyArrayObject*)PyArray_FROM_OTF(pyResult,NPY_DOUBLE,NPY_ARRAY_C_CONTIGUOUS);
    
    for (size_t ii = 0; ii < N; ii++){
        PyObject* val = PyArray_GETPTR1(arr,ii);
        out[ii] = PyFloat_AsDouble(val);
        Py_XDECREF(val);
    }

    Py_XDECREF(pyX);
    Py_XDECREF(pyResult);
    Py_XDECREF(arr);

    return 0;

}

double call_obj(double a, double b, struct Obj * obj)
{
    return eval(a,b,obj->f,obj->params);	
}

static PyObject* pycall_obj(PyObject* self, PyObject* args) {
    PyObject* pyA;
    PyObject* pyB;
    PyObject* pyObj;
    if(!PyArg_ParseTuple(args,"OOO",&pyA, &pyB,&pyObj,NULL)) return NULL;

    double a = PyFloat_AsDouble(pyA);
    double b = PyFloat_AsDouble(pyB);
    void* objPtr = PyCapsule_GetPointer(pyObj,NULL);
    double result = call_obj(a,b,objPtr);
    PyObject *pyResult = PyFloat_FromDouble(result);

    Py_XDECREF(pyA);
    Py_XDECREF(pyB);
    Py_XDECREF(pyObj);
   
    return pyResult;

}


static PyMethodDef pycback_methods[] = {
    {"assign",(PyCFunction)assign,METH_VARARGS,""},
    {"alloc_cobj",(PyCFunction)alloc_cobj,METH_VARARGS,""},
    {"call_obj",(PyCFunction)pycall_obj,METH_VARARGS,""},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef pycback_module = {
    PyModuleDef_HEAD_INIT,
    "pycback",
    "",
    -1,
    pycback_methods
};
#endif



#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC
PyInit_pycback(void) {
    PyObject* mod = PyModule_Create(&pycback_module);
    return mod;
}

#else
void initpycback(void) {
    Py_InitModule3("pycback",pycback_methods,"");
}
#endif

