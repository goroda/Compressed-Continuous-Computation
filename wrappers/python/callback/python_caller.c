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
    obj->f = NULL;
    obj->params = NULL;
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
    printf("Allocating _cobj\n");
    struct Obj * obj = obj_alloc();
    PyObject* pyObj = PyCapsule_New((void*)obj,"fcall",&destruct_obj);
    return pyObj;
}

static PyObject* assign( PyObject* self, PyObject* args ) {
    PyObject* pyObj;
    PyObject * pyDim;
    PyObject* pyF;
    PyObject* pyParams;

    if(!PyArg_ParseTuple(args,"OOOO",&pyObj,&pyDim,&pyF,&pyParams,NULL)) return NULL;

    if (!PyCallable_Check(pyF)){
        PyErr_SetString(PyExc_TypeError, "Third parameter must be callable");
        return NULL;
    }
    /* Py_XINCREF(pyF); */

    printf("valid? == %d\n", PyCapsule_IsValid(pyObj, "fcall"));
    void* objPtr = PyCapsule_GetPointer(pyObj,"fcall");
    struct Obj* obj = (struct Obj*)objPtr;

    size_t d;
    /* printf("%d\n",PyLong_Check(pyDim)); */
    if (PyLong_Check(pyDim)){
        d = PyLong_AsSsize_t(pyDim);
        assign_pointer(obj,d,pyF,pyParams);
    }
    else{
        PyErr_SetString(PyExc_TypeError, "Second parameter must be an int");
        return NULL;
    }

    return Py_None;
}

int eval_py_obj(size_t N, const double * x, double * out, void * obj_void)
{
    struct Obj * obj = obj_void;
    size_t dim = obj->dim;
    
    npy_intp dims[2];
    dims[0] = N;
    dims[1] = dim;

    PyObject * pyX = PyArray_SimpleNewFromData(2,dims,NPY_DOUBLE,(double*)x);
    PyObject * arglist = PyTuple_Pack(2,pyX,obj->params);


    PyObject * pyResult = PyObject_CallObject(obj->f,arglist);
    PyArrayObject * arr = (PyArrayObject*)PyArray_ContiguousFromAny(pyResult,NPY_DOUBLE,1,1);
    
    size_t ndims = (size_t)PyArray_NDIM(arr);
    if (ndims > 1){
        PyErr_SetString(PyExc_TypeError, "Wrapped python function must return a flattened (1d) array\n");
        return 1;
    }
    npy_intp * dimss  = (npy_intp*)PyArray_DIMS(arr);

    if ((size_t)dimss[0] != N){
        PyErr_SetString(PyExc_TypeError, "Wrapped python function must return an array with the same number of rows as input\n");
        return 1;        
    }
    
    double * vals = (double*)PyArray_DATA(arr);
    for (size_t ii = 0; ii < N; ii++){
        out[ii] = vals[ii];
    }

    Py_XDECREF(arr);
    Py_XDECREF(arglist);
    Py_XDECREF(pyResult);
    return 0;
}

static PyObject * eval_test_5d(PyObject * self, PyObject * args)
{
    PyObject* pyObj;

    if(!PyArg_ParseTuple(args,"O",&pyObj, NULL)) return NULL;
    
    size_t N = 5;
    void* objPtr = PyCapsule_GetPointer(pyObj,NULL);
    struct Obj * obj = objPtr;

    double pt[10]= {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
    double out[5];

    int res = eval_py_obj(N,pt,out,obj);
    if (res == 1){
        return NULL;
    }
    return Py_None;
}

typedef enum {ND=0, INTND, VEC, MOVEC, ARRVEC, NUMFT} Ftype;

struct Fwrap
{
    size_t d;

    double (*f)(const double *,void *);
    int (*fvec)(size_t,const double*,double*,void*);

    double (*intf)(const size_t *, void *);

    void * fargs;

    Ftype ftype;

    // special arguments if type == MOVEC (multioutput vec)
    int (*mofvec)(size_t, size_t, const double *, double *, void *);
    size_t evalfunc; // which function to eval from mofvec

    // special arguments if type == ARRVEC (array of functions)
    size_t nfuncs;
    int (**arrfuncs)(size_t,const double*,double*,void*);
    void ** farrargs;


    // fiber stuff
    int fiber_approx;
    size_t fiber_dim;
    size_t nfibers;
    /* double ** fiber_vals; */
    void ** fiber_vals;
    size_t onfiber;


    //Interface
    int interface;
};

static PyObject * fwrap_set_pyfunc(PyObject * self, PyObject * args)
/* void fwrap_set_pyfunc(struct Fwrap * fwrap, struct Object * args)     */
{

    PyObject* pyObj;    
    PyObject* pyFwrap;
    if(!PyArg_ParseTuple(args,"OO",&pyFwrap, &pyObj, NULL)) return NULL;

    struct Obj * obj = PyCapsule_GetPointer(pyObj, "fcall");
    printf("Got obj\n");
    printf("obj == NULL? = %d\n", obj==NULL);
    printf("obj->dim = %zu\n", obj->dim);

    printf("valid? == %d\n", PyCapsule_IsValid(pyFwrap, NULL));
    struct Fwrap * fwrap = PyCapsule_GetPointer(pyFwrap, NULL);
    printf("fwrap==NULL = %d\n", fwrap==NULL);
    /* fwrap->fvec = c3py_wrapped_eval; */
    /* fwrap->fargs = obj; */

    return Py_None;
}

static struct PyMethodDef pycback_methods[] = {
    {"assign",assign,METH_VARARGS,""},
    {"alloc_cobj",alloc_cobj,METH_VARARGS,""},
    {"fwrap_set_pyfunc", fwrap_set_pyfunc, METH_VARARGS,""},
    {"eval_test_5d",eval_test_5d,METH_VARARGS,""},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef pycback_module = {
    PyModuleDef_HEAD_INIT,
    "pycback",
    "",
    -1,
    pycback_methods
};


PyMODINIT_FUNC
PyInit_pycback(void) {
    PyObject* mod = PyModule_Create(&pycback_module);
    import_array();
    return mod;
}


