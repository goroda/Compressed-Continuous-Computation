#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <stdio.h>
#include <assert.h>

#include "numpy/arrayobject.h"
#include "numpy/ndarraytypes.h"

struct Obj
{
    size_t dim;
    void * f;
    void * params;
};

void assign_pointer(struct Obj * obj, size_t dim, PyObject * f, PyObject * params)   
{
    obj->dim = dim;
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
    PyObject * pyDim;
    PyObject* pyF;
    PyObject* pyParams;

    if(!PyArg_ParseTuple(args,"OOOO",&pyObj,&pyDim,&pyF,&pyParams,NULL)) return NULL;

    void* objPtr = PyCapsule_GetPointer(pyObj,NULL);
    struct Obj* obj = (struct Obj*)objPtr;

    size_t d = PyLong_AsSsize_t(pyDim);
    
    assign_pointer(obj,d,pyF,pyParams);
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


double call_obj(double a, double b, struct Obj * obj)
{
    return eval(a,b,obj->f,obj->params);	
}

/* static PyObject* pycall_obj(PyObject* self, PyObject* args) { */
/*     PyObject* pyA; */
/*     PyObject* pyB; */
/*     PyObject* pyObj; */
/*     if(!PyArg_ParseTuple(args,"OOO",&pyA, &pyB,&pyObj,NULL)) return NULL; */

/*     double a = PyFloat_AsDouble(pyA); */
/*     double b = PyFloat_AsDouble(pyB); */
/*     void* objPtr = PyCapsule_GetPointer(pyObj,NULL); */
/*     double result = call_obj(a,b,objPtr); */
/*     PyObject *pyResult = PyFloat_FromDouble(result); */

/*     Py_XDECREF(pyA); */
/*     Py_XDECREF(pyB); */
/*     Py_XDECREF(pyObj); */
   
/*     return pyResult; */
/* } */

/* static int eval_arr(size_t N, size_t dim, const double * x, double * out, PyObject* pyFunction, PyObject* pyParams ) { */

/*     long int dims[2]; */
/*     dims[0] = N; */
/*     dims[1] = dim; */

/*     /\* printf("creating pyX\n"); *\/ */
/*     PyObject* pyX = PyArray_SimpleNewFromData(2,dims,NPY_DOUBLE,(void*)x); */

/*     /\* printf("Got PyX\n"); *\/ */
/*     PyObject* pyResult = PyObject_CallFunctionObjArgs(pyFunction,pyX,pyParams,NULL); */

/*     /\* printf("Got Result\n"); *\/ */
/*     PyArrayObject * arr = (PyArrayObject*)PyArray_FROM_OTF(pyResult,NPY_DOUBLE,NPY_ARRAY_C_CONTIGUOUS); */

/*     /\* printf("Converted to array\n"); *\/ */
/*     size_t ndims = PyArray_NDIM(arr); */
/*     if (ndims > 1){ */
/*         fprintf(stderr, "Wrapped python function must return a flattened (1d) array\n"); */
/*         exit(1); */
/*     } */
/*     npy_intp * dimss  = PyArray_DIMS(arr); */

/*     if ((size_t)dimss[0] != N){ */
/*         fprintf(stderr, "Wrapped function must return an array with the same number of rows as input\n"); */
/*         exit(1); */
/*     } */
    
/*     /\* printf("num dim = %zu\n",ndims ); *\/ */
/*     /\* printf("nelem = %ld\n", dimss[0]); *\/ */

/*     double * vals = (double*)PyArray_DATA(arr); */
/*     for (size_t ii = 0; ii < N; ii++){ */
/*         out[ii] = vals[ii]; */
/*     } */

/*     Py_XDECREF(pyX); */
/*     Py_XDECREF(pyResult); */
/*     Py_XDECREF(arr); */

/*     return 0; */

/* } */

/* static PyObject* call_obj_test_arr(PyObject* self, PyObject* args) { */

/*     PyObject* pyObj; */
/*     if(!PyArg_ParseTuple(args,"O",&pyObj,NULL)) return NULL; */
/*     struct Obj * obj = PyCapsule_GetPointer(pyObj,NULL); */

/*     double a[5] = {0.0,2.0,3.0,4.0,5.0}; */
/*     double y[5]; */
/*     /\* printf("Calling eval_arr\n"); *\/ */
/*     eval_arr(5,1,a,y,obj->f,obj->params); */
/*     /\* printf("done Calling eval_arr\n"); *\/ */
/*     /\* for (size_t ii = 0; ii < 5; ii++){ *\/ */
/*     /\*     printf("y[%zu] = %G\n",ii,y[ii]); *\/ */
/*     /\* } *\/ */

/*     Py_XDECREF(pyObj); */

/*     PyObject *pyResult = Py_BuildValue("i",0); */
/*     return pyResult; */
/* } */


/* int c3py_wrapped_eval(size_t N, const double * x, double * out, void * args) */
/* { */
/*     struct Obj * obj = args; */
/*     int res = eval_arr(N,obj->dim,x,out,obj->f,obj->params); */
/*     return res; */
/* } */


/* void destruct_fwrap(PyObject * fwrapin) */
/* { */
/*     void* fwrapPtr = PyCapsule_GetPointer(fwrapin,NULL); */
/*     struct Fwrap* fwrap = (struct Fwrap*)fwrapPtr; */
/*     fwrap_destroy(fwrap); fwrap = NULL; */
/* } */

/* static PyObject* pyfunc_to_fwrap( PyObject* self, PyObject* args ) { */

/*     PyObject* pyObj; */
/*     if(!PyArg_ParseTuple(args,"O",&pyObj,NULL)) return NULL; */
/*     struct Obj * obj = PyCapsule_GetPointer(pyObj,NULL); */

/*     struct Fwrap * fw = fwrap_create(obj->dim,"general-vec"); */
/*     fwrap_set_fvec(fw,c3py_wrapped_eval,obj); */
/*     PyObject* pyFW = PyCapsule_New((void*)fw,NULL,&destruct_fwrap); */
    
/*     return pyFW; */
/* } */


static PyMethodDef pycback_methods[] = {
    {"assign",(PyCFunction)assign,METH_VARARGS,""},
    {"alloc_cobj",(PyCFunction)alloc_cobj,METH_VARARGS,""},
    /* {"call_obj",(PyCFunction)pycall_obj,METH_VARARGS,""}, */
    /* {"call_obj_test_arr",(PyCFunction)call_obj_test_arr,METH_VARARGS,""}, */
    /* {"pyfunc_to_fwrap",(PyCFunction)pyfunc_to_fwrap,METH_VARARGS,""}, */
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
    import_array();
    return mod;
}

#else
void initpycback(void) {
    Py_InitModule3("pycback",pycback_methods,"");
    import_array();
}
#endif

