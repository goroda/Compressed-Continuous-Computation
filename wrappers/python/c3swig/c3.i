// c3.i -Swig interface

%module c3

%{
    #define SWIG_FILE_WITH_INIT

    #include <Python.h>
/*     #include "lib_interface/approximate.h" */
/*     #include "lib_array/array.h" */
    
/* #include "lib_stringmanip/stringmanip.h" */
/* #include "lib_superlearn/regress.h" */
/* #include "lib_superlearn/learning_options.h" */
/* #include "lib_superlearn/objective_functions.h" */
/* #include "lib_superlearn/parameterization.h" */
/* #include "lib_superlearn/superlearn.h" */
/* #include "lib_interface/c3_interface.h" */
/* #include "lib_clinalg/diffusion.h" */
/* #include "lib_clinalg/dmrg.h" */
/* #include "lib_clinalg/dmrgprod.h" */
/* #include "lib_clinalg/ft.h" */
/* #include "lib_clinalg/indmanage.h" */
/* #include "lib_clinalg/lib_clinalg.h" */
/* #include "lib_clinalg/qmarray.h" */
/* #include "lib_clinalg/quasimatrix.h" */
/* #include "lib_funcs/fapprox.h" */
/* #include "lib_funcs/functions.h" */
/* #include "lib_funcs/fwrap.h" */
/* #include "lib_funcs/hpoly.h" */
/* #include "lib_funcs/legtens.h" */
/* #include "lib_funcs/lib_funcs.h" */
/* #include "lib_funcs/linelm.h" */
/* #include "lib_funcs/kernels.h" */
/* #include "lib_funcs/monitoring.h" */
/* #include "lib_funcs/piecewisepoly.h" */
/* #include "lib_funcs/pivoting.h" */
/* #include "lib_funcs/polynomials.h" */
/* #include "lib_funcs/space.h" */
/* #include "lib_linalg/lib_linalg.h" */
/* #include "lib_linalg/linalg.h" */
/* #include "lib_linalg/matrix_util.h" */
/* #include "lib_optimization/lib_optimization.h" */
/* #include "lib_optimization/optimization.h" */
/* #include "lib_probability/lib_probability.h" */
/* #include "lib_probability/probability.h" */
/* #include "lib_quadrature/quadrature.h" */
         
    #include "approximate.h"
    #include "array.h"
    #include "stringmanip.h"
    #include "regress.h"
    #include "learning_options.h"
    #include "objective_functions.h"
    #include "parameterization.h"
    #include "superlearn.h"
    #include "regress.h"
    #include "c3_interface.h"
    #include "diffusion.h"
    #include "dmrg.h"
    #include "dmrgprod.h"
    #include "ft.h"
    #include "indmanage.h"
    #include "lib_clinalg.h"
    #include "qmarray.h"
    #include "quasimatrix.h"
    /* #include "fft.h" */
    #include "fapprox.h"
    #include "functions.h"
    #include "fwrap.h"
    #include "hpoly.h"
    #include "legtens.h"
    #include "lib_funcs.h"
    #include "linelm.h"
    #include "kernels.h"
    #include "monitoring.h"
    #include "piecewisepoly.h"
    #include "pivoting.h"
    #include "polynomials.h"
    /* #include "fourier.h"     */
    #include "space.h"
    #include "lib_linalg.h"
    #include "linalg.h"
    #include "matrix_util.h"
    #include "lib_optimization.h"
    #include "probability.h"
    #include "lib_probability.h"
    #include "optimization.h"
    #include "quadrature.h"

    int c3py_wrapped_eval(size_t N, const double * x, double * out, void * args);
    void fwrap_set_pyfunc(struct Fwrap *, PyObject *);

    typedef long unsigned int size_t;
%}


typedef long unsigned int size_t;
int c3py_wrapped_eval(size_t N, const double * x, double * out, void * args);
void fwrap_set_pyfunc(struct Fwrap *, PyObject *);


%include "numpy.i"

%init %{
    import_array();    
%}

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len2, const double * ydata)
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len1, const double * xdata)
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len1, const double * evalnd_pt)
};

%apply (int DIM1, double* INPLACE_ARRAY1) {
    (size_t len2, double * evalnd_out)
};

%apply (int DIM1, size_t* IN_ARRAY1) {
    (size_t leni, const size_t * xi) 
};


// Python Memory Management
%newobject function_train_alloc;
struct FunctionTrain * function_train_alloc(size_t);
%delobject function_train_free;
void function_train_free(struct FunctionTrain *);


// Modified Python Interface
%rename (ft_regress_run) my_ft_regress_run;
%exception my_ft_regress_run{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    struct FunctionTrain * my_ft_regress_run(struct FTRegress * ftr ,struct c3Opt * opt ,size_t len1, const double* xdata, size_t len2, const double * ydata){
        if (len1 != len2*ft_regress_get_dim(ftr)){
            PyErr_Format(PyExc_ValueError,
                         "Arrays of lengths (%zu,%zu) given",
                         len1, len2);
            return NULL;
        }
        return ft_regress_run(ftr,opt,len2,xdata,ydata);
    }
%}
%ignore ft_regress_run;


%rename (cross_validate_init) my_cross_validate_init;
%exception my_cross_validate_init{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    struct CrossValidate * my_cross_validate_init(size_t dim, size_t len1, const double* xdata, size_t len2, const double * ydata,size_t kfold, int verbose){
        if (len1 != len2*dim){
            PyErr_Format(PyExc_ValueError,
                         "Arrays of lengths (%zu,%zu) given",
                         len1, len2);
            return NULL;
        }
        return cross_validate_init(len2,dim,xdata,ydata,kfold,verbose);
    }
%}
%ignore cross_validate_init;

%rename (lin_elem_exp_aopts_alloc) my_lin_elem_exp_aopts_alloc;
%exception my_lin_elem_exp_aopts_alloc{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    struct LinElemExpAopts * my_lin_elem_exp_aopts_alloc(size_t len1, const double* evalnd_pt){

      struct LinElemExpAopts * lexp = lin_elem_exp_aopts_alloc(len1,(double*)evalnd_pt);
      lin_elem_exp_aopts_set_nodes_copy(lexp,len1,evalnd_pt);
      return lexp;
    }
%}
%ignore lin_elem_exp_aopts_alloc;

%rename (function_train_eval) my_function_train_eval;
%exception my_function_train_eval{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    double my_function_train_eval(struct FunctionTrain * ft ,size_t len1, const double* evalnd_pt){
        if (len1 != function_train_get_dim(ft)){
            PyErr_Format(PyExc_ValueError,
                         "Evaluation point has incorrect dimensions (%zu) instead of %zu",
                         len1,function_train_get_dim(ft));
            return 0.0;
        }
        return function_train_eval(ft,evalnd_pt);
    }
%}
%ignore function_train_eval;

%rename (function_train_gradient_eval) my_function_train_gradient_eval;
%exception my_function_train_gradient_eval{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    void my_function_train_gradient_eval(struct FunctionTrain * ft ,size_t len1, const double* evalnd_pt,
                                         size_t len2, double * evalnd_out){
        if (len1 != function_train_get_dim(ft)){
            PyErr_Format(PyExc_ValueError,
                         "Evaluation point has incorrect dimensions (%zu) instead of %zu",
                         len1,function_train_get_dim(ft));
        }
        if (len1 != len2){
            PyErr_Format(PyExc_ValueError,
                         "Input and outputs must be the same size: currently they are (%zu,%zu)",
                         len1,len2);
        }
        
        function_train_gradient_eval(ft,evalnd_pt,evalnd_out);

    }
%}
%ignore function_train_gradient_eval;


%typemap(in) size_t * ranks {
    if (!PyList_Check($input)) {
        PyErr_SetString(PyExc_ValueError,"Expecting a list");
        return NULL;
    }
    int size = PyList_Size($input);
    $1 = (size_t *) malloc(size * sizeof(size_t));
    for (int i = 0; i < size; i++){
        PyObject *s = PyList_GetItem($input,i);
        if (!PyInt_Check(s)) {
            free($1);
            PyErr_SetString(PyExc_ValueError, "List items must be integers");
            return NULL;
        }
        $1[i] = PyInt_AsLong(s);
    }
 }

%typemap(freearg) (size_t * ranks){
    if ($1) free($1);
}

%typemap(in) void * paramlist {
    if (!PyList_Check($input)) {
        PyErr_SetString(PyExc_ValueError,"Expecting a list");
        return NULL;
    }
    int size = PyList_Size($input);

    PyObject *s = PyList_GetItem($input,0);
    int is_size_t = 0;
    if (PyInt_Check(s)){
        is_size_t = 1;
    }
    else{
        if (!PyFloat_Check(s)){
            PyErr_SetString(PyExc_ValueError, "List items must be integers or floats");
            return NULL;
        }
    }

    if (is_size_t == 1){
        $1 = malloc(size * sizeof(size_t));
    }
    else{
        $1 = malloc(size * sizeof(double));
    }
    for (int i = 0; i < size; i++){
        PyObject *s = PyList_GetItem($input,i);
        if (is_size_t == 1){
            if (!PyInt_Check(s)) {
                free($1);
                PyErr_SetString(PyExc_ValueError, "List items must the same type");
                return NULL;
            }
            ((size_t *)$1)[i] = PyInt_AsLong(s);
        }
        else{
            if (!PyFloat_Check(s)) {
                free($1);
                PyErr_SetString(PyExc_ValueError, "List items must be the same type");
                return NULL;
            }
            ((double *)$1)[i] = PyFloat_AsDouble(s);
        }

    }
 }

%typemap(freearg) (void * paramlist){
    if ($1) free($1);
}

%typemap(in) double * onedx {
    if (!PyList_Check($input)) {
        PyErr_SetString(PyExc_ValueError,"Expecting a list");
        return NULL;
    }
    int size = PyList_Size($input);
    $1 = (double *) malloc(size * sizeof(double));
    for (int i = 0; i < size; i++){
        PyObject *s = PyList_GetItem($input,i);
        if (!PyFloat_Check(s)) {
            free($1);
            PyErr_SetString(PyExc_ValueError, "List items must be floating point values");
            return NULL;
        }
        $1[i] = PyFloat_AsDouble(s);
    }

    /* printf("size = %d\n",size); */
    /* printf("list_elem = "); */
    /* for (int ii = 0; ii < size; ii++){ */
    /*   printf("$1[%d] = %G\n",ii,$1[ii]); */
    /* } */
 }

%typemap(freearg) (double * onedx){
    if ($1) free($1);
}


/* /\* %include "../../include/c3.h" *\/ */

/* %include "../../src/lib_interface/approximate.h" */

/* %include "../../src/lib_array/array.h" */

/* %include "../../src/lib_stringmanip/stringmanip.h" */
/* %include "../../src/lib_superlearn/regress.h" */
/* %include "../../src/lib_superlearn/learning_options.h" */
/* %include "../../src/lib_superlearn/objective_functions.h" */
/* %include "../../src/lib_superlearn/parameterization.h" */
/* %include "../../src/lib_superlearn/superlearn.h" */
/* %include "../../src/lib_interface/c3_interface.h" */
/* %include "../../src/lib_clinalg/diffusion.h" */
/* %include "../../src/lib_clinalg/dmrg.h" */
/* %include "../../src/lib_clinalg/dmrgprod.h" */
/* %include "../../src/lib_clinalg/ft.h" */
/* %include "../../src/lib_clinalg/indmanage.h" */
/* %include "../../src/lib_clinalg/lib_clinalg.h" */
/* %include "../../src/lib_clinalg/qmarray.h" */
/* %include "../../src/lib_clinalg/quasimatrix.h" */
/* /\* %include "../../src/lib_fft/fft.h" *\/ */
/* %include "../../src/lib_funcs/fapprox.h" */
/* %include "../../src/lib_funcs/functions.h" */
/* %include "../../src/lib_funcs/fwrap.h" */
/* %include "../../src/lib_funcs/hpoly.h" */
/* %include "../../src/lib_funcs/legtens.h" */
/* %include "../../src/lib_funcs/lib_funcs.h" */
/* %include "../../src/lib_funcs/linelm.h" */
/* %include "../../src/lib_funcs/kernels.h" */
/* %include "../../src/lib_funcs/monitoring.h" */
/* %include "../../src/lib_funcs/piecewisepoly.h" */
/* %include "../../src/lib_funcs/pivoting.h" */
/* %include "../../src/lib_funcs/polynomials.h" */
/* %include "../../src/lib_funcs/space.h" */
/* %include "../../src/lib_linalg/lib_linalg.h" */
/* %include "../../src/lib_linalg/linalg.h" */
/* %include "../../src/lib_linalg/matrix_util.h" */
/* %include "../../src/lib_optimization/lib_optimization.h" */
/* %include "../../src/lib_optimization/optimization.h" */
/* %include "../../src/lib_probability/lib_probability.h" */
/* %include "../../src/lib_probability/probability.h" */
/* %include "../../src/lib_quadrature/quadrature.h" */



/* %include "../../include/c3.h" */

%include "lib_interface/approximate.h"

%include "lib_array/array.h"

%include "lib_stringmanip/stringmanip.h"
%include "lib_superlearn/regress.h"
%include "lib_superlearn/learning_options.h"
%include "lib_superlearn/objective_functions.h"
%include "lib_superlearn/parameterization.h"
%include "lib_superlearn/superlearn.h"
%include "lib_interface/c3_interface.h"
%include "lib_clinalg/diffusion.h"
%include "lib_clinalg/dmrg.h"
%include "lib_clinalg/dmrgprod.h"
%include "lib_clinalg/ft.h"
%include "lib_clinalg/indmanage.h"
%include "lib_clinalg/lib_clinalg.h"
%include "lib_clinalg/qmarray.h"
%include "lib_clinalg/quasimatrix.h"
/* %include "lib_fft/fft.h" */
%include "lib_funcs/fapprox.h"
%include "lib_funcs/functions.h"
%include "lib_funcs/fwrap.h"
%include "lib_funcs/hpoly.h"
%include "lib_funcs/legtens.h"
%include "lib_funcs/lib_funcs.h"
%include "lib_funcs/linelm.h"
%include "lib_funcs/kernels.h"
%include "lib_funcs/monitoring.h"
%include "lib_funcs/piecewisepoly.h"
%include "lib_funcs/pivoting.h"
%include "lib_funcs/polynomials.h"
%include "lib_funcs/space.h"
%include "lib_linalg/lib_linalg.h"
%include "lib_linalg/linalg.h"
%include "lib_linalg/matrix_util.h"
%include "lib_optimization/lib_optimization.h"
%include "lib_optimization/optimization.h"
%include "lib_probability/lib_probability.h"
%include "lib_probability/probability.h"
%include "lib_quadrature/quadrature.h"



