// c3.i -Swig interface

%module c3

%{
    #define SWIG_FILE_WITH_INIT

    #include <Python.h>
    #include "approximate.h"
    #include "array.h"
    #include "stringmanip.h"
    #include "regress.h"
    #include "learning_options.h"
    #include "objective_functions.h"
    #include "parameterization.h"
    #include "superlearn.h"
    #include "superlearn_util.h"
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

%apply (int DIM1, double* INPLACE_ARRAY1) {
    (size_t len2, double * evalnd_out)
};

%apply (int DIM1, size_t* IN_ARRAY1) {
    (size_t leni, const size_t * xi) 
};

%apply (int DIM1, size_t* IN_ARRAY1) {
    (size_t len1, const size_t * init_ranks) 
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len1, const double * params)
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len1, const double * x)
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len2, double * grad)
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len, double * params)
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len2, double * evals)
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len1, double * x)
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len3, double * running_eval)
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len4, double * running_lr)
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len5, double * running_rl)
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len2, double * out)
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len3, double * y)
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len4, double * init_sample)
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len5, double * prior_cov)
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len6, double * prior_mean)
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len5, double * prior_alphas)
};

%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len6, double * prior_thetas)
};
%apply (int DIM1, double* IN_ARRAY1) {
    (size_t len7, double * prior_out)
};

/* reference https://stackoverflow.com/questions/3843064/swig-passing-argument-to-python-callback-function */

extern void fwrap_set_pyfunc(struct Fwrap * fwrap, PyObject *PyFunc, PyObject * args);

extern void test_py_call(void);

%{
static PyObject *my_pycallback = NULL;
static PyObject *my_pycallback_args = NULL;
static int PythonCallBack(size_t N, const double * x, double * out, void * argsv)
{
    struct Fwrap * fwrap = argsv;
    size_t dim = fwrap_get_d(fwrap);
    /* PyObject * args = argsv; */
    PyObject *func, *arglist;
    PyObject *result;

    /* printf("dim = %zu\n", dim); */
    /* size_t N = 5; */
    npy_intp dims[2];
    dims[0] = N;
    dims[1] = dim;
    /* double pt[10]= {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}; */
    /* double out[5];    */
    PyObject * pyX = PyArray_SimpleNewFromData(2,dims,NPY_DOUBLE,(double*)x);
   
    func = my_pycallback;     /* This is the function .... */
    arglist = Py_BuildValue("(OO)", pyX, my_pycallback_args);  /* No arguments needed */
    result =  PyEval_CallObject(func, arglist);
    PyArrayObject * arr = (PyArrayObject*)PyArray_ContiguousFromAny(result,NPY_DOUBLE,1,1);
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
    Py_DECREF(arglist);
    Py_XDECREF(result);
    Py_XDECREF(pyX);
    return 0 /*void*/;
}

void fwrap_set_pyfunc(struct Fwrap * fwrap, PyObject *PyFunc, PyObject * params)
{
    Py_XDECREF(my_pycallback);        /* Dispose of previous callback */
    Py_XINCREF(PyFunc);               /* Add a reference to new callback */    
    my_pycallback = PyFunc;           /* Remember new callback */

    Py_XDECREF(my_pycallback_args);   /* Dispose of previous callback */
    Py_XINCREF(params);               /* Add a reference to new callback */    
    my_pycallback_args = params;

    fwrap_set_fvec(fwrap, PythonCallBack, fwrap);
}

void test_py_call(void){

    size_t N = 5;
    size_t dim = 2;
    double pt[10]= {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
    double out[5];
    printf("Testing the callback function\n");
    struct Fwrap * fwrap = fwrap_create(dim, "python");
    PythonCallBack(N, pt, out, fwrap);
    fwrap_destroy(fwrap); fwrap = NULL;
    for (size_t ii = 0; ii < N; ii++){
        printf("out[%zu] = %3.5E\n", ii, out[ii]);
    }
}

%}

/* %typemap(pythin, in) PyObject *PyFunc { */
%typemap(in) PyObject *PyFunc {    
  if (!PyCallable_Check($input)) {
      PyErr_SetString(PyExc_TypeError, "Need a callable object!");
      return NULL;
  }
  $1 = $input;
}


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

%rename (ft_regress_get_ft_param) my_ft_regress_get_ft_param;
%exception my_ft_regress_get_ft_param{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    struct FTparam * my_ft_regress_get_ft_param(struct FTRegress * ftr){
        return ft_regress_get_ft_param(ftr);
    }
%}
%ignore ft_regress_get_ft_param;


%rename (ft_param_get_dim) my_ft_param_get_dim;
%exception my_ft_param_get_dim{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    size_t my_ft_param_get_dim(const struct FTparam * ftp){
        return ft_param_get_dim(ftp);
    }
%}
%ignore ft_param_get_dim;


%rename (ft_param_update_params) my_ft_param_update_params;
%exception my_ft_param_update_params{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    void my_ft_param_update_params(struct FTparam * ftp, size_t len1, const double * params){
        return ft_param_update_params(ftp, params);
    }
%}
%ignore ft_param_update_params;


%rename (ft_param_gradeval) my_ft_param_gradeval;
%exception my_ft_param_gradeval{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    double my_ft_param_gradeval(struct FTparam * ftp, 
                        size_t len1, const double * x,
                        size_t len2, double * grad,
                        double * grad_evals,
                        double * mem, 
                        double * evals){
        return ft_param_gradeval(ftp, x, grad, grad_evals, mem, evals);
    }
%}
%ignore ft_param_gradeval;

%rename (ft_param_gradevals) my_ft_param_gradevals;
%exception my_ft_param_gradevals{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    void my_ft_param_gradevals(struct FTparam * ftp, size_t N,
                        size_t len1, const double * x,
                        size_t len2, double * grad,
                        double * grad_evals,
                        double * mem, 
                        double * evals){
        return ft_param_gradevals(ftp, N, x, grad, grad_evals, mem, evals);
    }
%}
%ignore ft_param_gradevals;

%rename (function_train_evals) my_function_train_evals;
%exception my_function_train_evals{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    void my_function_train_evals(struct FunctionTrain * ft, size_t N, size_t len1, const double * x, size_t len2, double * evals){
        return function_train_evals(ft, N, x, evals);
    }
%}
%ignore function_train_evals;


%rename (sl_mem_manager_check_structure) my_sl_mem_manager_check_structure;
%exception my_sl_mem_manager_check_structure{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    void my_sl_mem_manager_check_structure(struct SLMemManager * mem,
                                    const struct FTparam * ftp,
                                    size_t len1, const double * x){
        return sl_mem_manager_check_structure(mem, ftp, x);
    }
%}
%ignore sl_mem_manager_check_structure;

%rename (ft_param_core_gradevals) my_ft_param_core_gradevals;
%exception my_ft_param_core_gradevals{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    void my_ft_param_core_gradevals(struct FTparam * ftp, size_t core, size_t N, size_t len1, double * x,
                              size_t len2, double * grad, int calc_running, size_t len4, double * running_lr, size_t len5, double * running_rl,
                              size_t len3, double * running_eval)
                              {
        return ft_param_core_gradevals(ftp, core, N, x, grad, calc_running, running_lr, running_rl, running_eval);
    }
%}
%ignore ft_param_core_gradevals;

%rename (ft_param_update_core_params) my_ft_param_update_core_params;
%exception my_ft_param_update_core_params{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    void my_ft_param_update_core_params(struct FTparam * ftp, size_t core, size_t len1, const double * params){
        return ft_param_update_core_params(ftp, core, params);
    }
%}
%ignore ft_param_update_core_params;

%rename (function_train_eval_running_right_left) my_function_train_eval_running_right_left;
%exception my_function_train_eval_running_right_left{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    void my_function_train_eval_running_right_left(struct FunctionTrain * ft, size_t N, size_t len1, const double * x, size_t len2, double * out){
        return function_train_eval_running_right_left(ft, N, x, out);
    }
%}
%ignore function_train_eval_running_right_left;

%rename (function_train_eval_running_left_right) my_function_train_eval_running_left_right;
%exception my_function_train_eval_running_left_right{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    void my_function_train_eval_running_left_right(struct FunctionTrain * ft, size_t N, size_t len1, const double * x, size_t len2, double * out){
        return function_train_eval_running_left_right(ft, N, x, out);
    }
%}
%ignore function_train_eval_running_left_right;

%rename (function_train_eval_core) my_function_train_eval_core;
%exception my_function_train_eval_core{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    void my_function_train_eval_core(struct FunctionTrain * ft, size_t core, size_t N, size_t len1, const double * x, size_t len2, double * out){
        return function_train_eval_core(ft, core, N, x, out);
    }
%}
%ignore function_train_eval_core;

%rename (sample_gibbs_linear) my_sample_gibbs_linear;
%exception my_sample_gibbs_linear{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    void my_sample_gibbs_linear(struct FTparam * ftp, size_t N, size_t len1, double * x, size_t len3, double * y, 
        size_t len4, double * init_sample, size_t len5, double * prior_cov, size_t len6, double * prior_mean, double noise_var,
        size_t Nsamples, size_t len2, double * out){
        return sample_gibbs_linear(ftp, N, x, y, init_sample, prior_cov, prior_mean, noise_var, Nsamples, out);
    }
%}
%ignore sample_gibbs_linear;

%rename (sample_gibbs_linear_noise) my_sample_gibbs_linear_noise;
%exception my_sample_gibbs_linear_noise{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    void my_sample_gibbs_linear_noise(struct FTparam * ftp, size_t N, size_t len1, double * x, size_t len3, double * y, 
        size_t len4, double * init_sample, size_t len5, double * prior_cov, size_t len6, double * prior_mean, int noise_alpha,
        double noise_theta, size_t Nsamples, size_t len2, double * out){
        return sample_gibbs_linear_noise(ftp, N, x, y, init_sample, prior_cov, prior_mean, noise_alpha, noise_theta, Nsamples, out);
    }
%}
%ignore sample_gibbs_linear_noise;

%rename (sample_hier_group_gibbs_linear_noise) my_sample_hier_group_gibbs_linear_noise;
%exception my_sample_hier_group_gibbs_linear_noise{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    void my_sample_hier_group_gibbs_linear_noise(struct FTparam * ftp, size_t N, size_t len1, double * x, size_t len3, double * y, 
        size_t len4, double * init_sample, int prior_alpha, double prior_theta, int noise_alpha,
        double noise_theta, size_t Nsamples, size_t len2, double * out){
        return sample_hier_group_gibbs_linear_noise(ftp, N, x, y, init_sample, prior_alpha, prior_theta, noise_alpha, noise_theta, Nsamples, out);
    }
%}
%ignore sample_hier_group_gibbs_linear_noise;

%rename (sample_hier_ind_gibbs_linear_noise) my_sample_hier_ind_gibbs_linear_noise;
%exception my_sample_hier_ind_gibbs_linear_noise{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    void my_sample_hier_ind_gibbs_linear_noise(struct FTparam * ftp, size_t N, size_t len1, double * x, size_t len3, double * y, 
        size_t len4, double * init_sample, size_t len5, double * prior_alphas, size_t len6, double * prior_thetas, int noise_alpha, 
        double noise_theta, size_t Nsamples, size_t len2, double * out, size_t len7, double* prior_out){
        return sample_hier_ind_gibbs_linear_noise(ftp, N, x, y, init_sample, prior_alphas, prior_thetas, noise_alpha, noise_theta, Nsamples, out, prior_out);
    }
%}
%ignore sample_hier_ind_gibbs_linear_noise;

%rename (ft_param_gradeval_lin) my_ft_param_gradeval_lin;
%exception my_ft_param_gradeval_lin{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    double my_ft_param_gradeval_lin(struct FTparam * ftp, const double * grad_evals,
                             size_t len2, double * grad, double * mem, double * evals){
        return ft_param_gradeval_lin(ftp, grad_evals, grad, mem, evals);
    }
%}
%ignore ft_param_gradeval_lin;

%rename (ft_param_get_params) my_ft_param_get_params;
%exception my_ft_param_get_params{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    size_t my_ft_param_get_params(struct FTparam * ftp, size_t nparams, size_t len, double * params){
        return ft_param_get_params(ftp, nparams, params);
    }
%}
%ignore ft_param_get_params;

%rename (function_train_get_params) my_function_train_get_params;
%exception my_function_train_get_params{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    size_t my_function_train_get_params(const struct FunctionTrain * ft, size_t len, double * params){
        return function_train_get_params(ft, params);
    }
%}
%ignore function_train_get_params;


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

%rename (c3approx_init_cross_het) my_c3approx_init_cross_het;
%exception my_c3approx_init_cross_het{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    void my_c3approx_init_cross_het(struct C3Approx * c3a, size_t len1, const size_t *init_ranks, int verbose, double ** startin) {
		if (len1 != (c3approx_get_dim(c3a)+1)){
            PyErr_Format(PyExc_ValueError,
                         "ranks has incorrect dimensions (%zu) instead of %zu",
                         len1, c3approx_get_dim(c3a)+1);
        }
		/* size_t *use_ranks = calloc(len1, sizeof(size_t)); */
		/* for (int ii = 0; ii < len1; ii++) { */
		/* 	use_ranks[ii] = (int)init_ranks[ii]; */
		/* } */
		c3approx_init_cross_het(c3a, init_ranks, verbose, startin);
		/* c3approx_init_cross_het(c3a, use_ranks, verbose, startin); */
		/* free(use_ranks); use_ranks = NULL; */
    }
%}
%ignore c3approx_init_cross_het;

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

%rename (ft1d_array_eval2) my_ft1d_array_eval2;
%exception my_ft1d_array_eval2{
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    void my_ft1d_array_eval2(const struct FT1DArray * fta ,size_t len1, const double* evalnd_pt,
                             size_t len2, double * evalnd_out){
        if (len1 != function_train_get_dim(fta->ft[0])){
            PyErr_Format(PyExc_ValueError,
                         "Evaluation point has incorrect dimensions (%zu) instead of %zu",
                         len1,function_train_get_dim(fta->ft[0]));
        }
        if (len1*len1 != len2){
            PyErr_Format(PyExc_ValueError,
                         "Output must be size of dimension squared (%zu,%zu)",
                         len1,len1);
        }
        
        ft1d_array_eval2(fta,evalnd_pt,evalnd_out);

    }
%}
%ignore ft1d_array_eval2;


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
%include "lib_superlearn/superlearn_util.h"
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



