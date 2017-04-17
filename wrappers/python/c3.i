// c3.i -Swig interface

%module c3

%{
    #define SWIG_FILE_WITH_INIT

    #include <Python.h>
    /* #include "/home/aagorod/Software/c3/include/c3.h" */
    #include "/home/aagorod/Software/c3/src/lib_interface/approximate.h"
    #include "/home/aagorod/Software/c3/src/lib_array/array.h"

    #include "/home/aagorod/Software/c3/src/lib_stringmanip/stringmanip.h"
    #include "/home/aagorod/Software/c3/src/lib_superlearn/regress.h"
    #include "/home/aagorod/Software/c3/src/lib_interface/c3_interface.h"
    #include "/home/aagorod/Software/c3/src/lib_clinalg/diffusion.h"
    #include "/home/aagorod/Software/c3/src/lib_clinalg/dmrg.h"
    #include "/home/aagorod/Software/c3/src/lib_clinalg/dmrgprod.h"
    #include "/home/aagorod/Software/c3/src/lib_clinalg/ft.h"
    #include "/home/aagorod/Software/c3/src/lib_clinalg/indmanage.h"
    #include "/home/aagorod/Software/c3/src/lib_clinalg/lib_clinalg.h"
    #include "/home/aagorod/Software/c3/src/lib_clinalg/qmarray.h"
    #include "/home/aagorod/Software/c3/src/lib_clinalg/quasimatrix.h"
    /* #include "/home/aagorod/Software/c3/src/lib_fft/fft.h" */
    #include "/home/aagorod/Software/c3/src/lib_funcs/fapprox.h"
    #include "/home/aagorod/Software/c3/src/lib_funcs/functions.h"
    #include "/home/aagorod/Software/c3/src/lib_funcs/fwrap.h"
    #include "/home/aagorod/Software/c3/src/lib_funcs/hpoly.h"
    #include "/home/aagorod/Software/c3/src/lib_funcs/legtens.h"
    #include "/home/aagorod/Software/c3/src/lib_funcs/lib_funcs.h"
    #include "/home/aagorod/Software/c3/src/lib_funcs/linelm.h"
    #include "/home/aagorod/Software/c3/src/lib_funcs/kernels.h"
    #include "/home/aagorod/Software/c3/src/lib_funcs/monitoring.h"
    #include "/home/aagorod/Software/c3/src/lib_funcs/piecewisepoly.h"
    /* #include "/home/aagorod/Software/c3/src/lib_funcs/pivoting.h" */
    #include "/home/aagorod/Software/c3/src/lib_funcs/polynomials.h"
    #include "/home/aagorod/Software/c3/src/lib_funcs/space.h"
    #include "/home/aagorod/Software/c3/src/lib_linalg/lib_linalg.h"
    #include "/home/aagorod/Software/c3/src/lib_linalg/linalg.h"
    #include "/home/aagorod/Software/c3/src/lib_linalg/matrix_util.h"
    #include "/home/aagorod/Software/c3/src/lib_optimization/lib_optimization.h"
    #include "/home/aagorod/Software/c3/src/lib_optimization/optimization.h"
    #include "/home/aagorod/Software/c3/src/lib_quadrature/quadrature.h"
    /* #include "/home/aagorod/Software/c3/src/lib_tensdecomp/candecomp.h" */
    /* #include "/home/aagorod/Software/c3/src/lib_tensdecomp/cross.h" */
    /* #include "/home/aagorod/Software/c3/src/lib_tensdecomp/tensortrain.h" */
    /* #include "/home/aagorod/Software/c3/src/lib_tensdecomp/tt_integrate.h" */
    /* #include "/home/aagorod/Software/c3/src/lib_tensdecomp/tt_multilinalg.h" */
    /* #include "/home/aagorod/Software/c3/src/lib_tensor/tensor.h" */


    typedef long unsigned int size_t;
%}

typedef long unsigned int size_t;

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


%rename (ft_regress_run) my_ft_regress_run;
%exception my_ft_regress_run{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    struct FunctionTrain * my_ft_regress_run(struct FTRegress * ftr ,struct c3Opt * opt ,size_t len1, const double* xdata, size_t len2, const double * ydata){
        if (len1 != len2*ft_regress_get_dim(ftr)){
            PyErr_Format(PyExc_ValueError,
                         "Arrays of lengths (%d,%d) given",
                         len1, len2);
            return NULL;
        }
        return ft_regress_run(ftr,opt,len2,xdata,ydata);
    }
%}
%ignore ft_regress_run;


%rename (function_train_eval) my_function_train_eval;
%exception my_function_train_eval{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    double my_function_train_eval(struct FunctionTrain * ft ,size_t len1, const double* evalnd_pt){
        if (len1 != function_train_get_dim(ft)){
            PyErr_Format(PyExc_ValueError,
                         "Evaluation point has incorrect dimensions (%d) instead of %d",
                         len1,function_train_get_dim(ft));
            return 0.0;
        }
        return function_train_eval(ft,evalnd_pt);
    }
%}
%ignore function_train_eval;


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



%include "/home/aagorod/Software/c3/src/lib_interface/approximate.h"
/* %include "/home/aagorod/Software/c3/include/c3.h" */
%include "/home/aagorod/Software/c3/src/lib_array/array.h"

%include "/home/aagorod/Software/c3/src/lib_stringmanip/stringmanip.h"
%include "/home/aagorod/Software/c3/src/lib_superlearn/regress.h"
%include "/home/aagorod/Software/c3/src/lib_interface/c3_interface.h"
%include "/home/aagorod/Software/c3/src/lib_clinalg/diffusion.h"
%include "/home/aagorod/Software/c3/src/lib_clinalg/dmrg.h"
%include "/home/aagorod/Software/c3/src/lib_clinalg/dmrgprod.h"
%include "/home/aagorod/Software/c3/src/lib_clinalg/ft.h"
%include "/home/aagorod/Software/c3/src/lib_clinalg/indmanage.h"
%include "/home/aagorod/Software/c3/src/lib_clinalg/lib_clinalg.h"
%include "/home/aagorod/Software/c3/src/lib_clinalg/qmarray.h"
%include "/home/aagorod/Software/c3/src/lib_clinalg/quasimatrix.h"
/* %include "/home/aagorod/Software/c3/src/lib_fft/fft.h" */
%include "/home/aagorod/Software/c3/src/lib_funcs/fapprox.h"
%include "/home/aagorod/Software/c3/src/lib_funcs/functions.h"
%include "/home/aagorod/Software/c3/src/lib_funcs/fwrap.h"
%include "/home/aagorod/Software/c3/src/lib_funcs/hpoly.h"
%include "/home/aagorod/Software/c3/src/lib_funcs/legtens.h"
%include "/home/aagorod/Software/c3/src/lib_funcs/lib_funcs.h"
%include "/home/aagorod/Software/c3/src/lib_funcs/linelm.h"
%include "/home/aagorod/Software/c3/src/lib_funcs/kernels.h"
%include "/home/aagorod/Software/c3/src/lib_funcs/monitoring.h"
%include "/home/aagorod/Software/c3/src/lib_funcs/piecewisepoly.h"
/* %include "/home/aagorod/Software/c3/src/lib_funcs/pivoting.h" */
%include "/home/aagorod/Software/c3/src/lib_funcs/polynomials.h"
%include "/home/aagorod/Software/c3/src/lib_funcs/space.h"
%include "/home/aagorod/Software/c3/src/lib_linalg/lib_linalg.h"
%include "/home/aagorod/Software/c3/src/lib_linalg/linalg.h"
%include "/home/aagorod/Software/c3/src/lib_linalg/matrix_util.h"
%include "/home/aagorod/Software/c3/src/lib_optimization/lib_optimization.h"
%include "/home/aagorod/Software/c3/src/lib_optimization/optimization.h"
%include "/home/aagorod/Software/c3/src/lib_quadrature/quadrature.h"
/* %include "/home/aagorod/Software/c3/src/lib_tensdecomp/candecomp.h" */
/* %include "/home/aagorod/Software/c3/src/lib_tensdecomp/cross.h" */
/* %include "/home/aagorod/Software/c3/src/lib_tensdecomp/tensortrain.h" */
/* %include "/home/aagorod/Software/c3/src/lib_tensdecomp/tt_integrate.h" */
/* %include "/home/aagorod/Software/c3/src/lib_tensdecomp/tt_multilinalg.h" */
/* %include "/home/aagorod/Software/c3/src/lib_tensor/tensor.h" */



