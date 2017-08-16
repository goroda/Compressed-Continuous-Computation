# Compressed continuous computation in python
from __future__ import print_function
import c3
import numpy as np
import copy
import pycback as pcb
import atexit
# import contextlib

class FunctionTrain(object):

    def __init__(self,din):

        self.dim = din
        self.ft = None
        
        self.opts = []
        self.ranks = [1]
        for ii in range(din):
            self.opts.append(None)
            self.ranks.append(1)
            
        # atexit.register(self.cleanup)

    def copy(self):
        ft_out = FunctionTrain(self.dim)
        ft_out.ft = c3.function_train_copy(self.ft)
        ft_out.opts = copy.deepcopy(self.ft.opts)
        return ft_out

    def save(self,filename):
        c3.function_train_save(self.ft,filename)

    def load(self,filename):
        ft = c3.function_train_load(filename)
        self.dim = c3.function_train_get_dim(ft)
        self.ft = ft
    
    def set_dim_opts(self,dim,ftype,lb=-1,ub=1,nparam=4,kernel_height_scale=1.0,kernel_width_scale=1.0,kernel_adapt_center=0):

        if self.opts[dim] is not None:
            raise AttributeError('cannot call set_dim_opts because was already called')

        o = dict({})
        o['type'] = ftype
        o['lb'] = lb
        o['ub'] = ub
        o['nparam'] = nparam
        o['kernel_height_scale'] = 1.0
        o['kernel_adapt_center'] = 0.0
        self.opts.insert(dim, o)

    def __convert_opts_to_c3_form__(self):

        c3_ope_opts = []
        for ii in range(self.dim):
            ftype = self.opts[ii]['type']
            lb = self.opts[ii]['lb']
            ub = self.opts[ii]['ub']
            nparam = self.opts[ii]['nparam']
            kernel_height_scale = self.opts[ii]['kernel_height_scale']
            kernel_adapt_center = self.opts[ii]['kernel_adapt_center']
            
            if ftype == "legendre":
                c3_ope_opts.append(c3.ope_opts_alloc(c3.LEGENDRE))
                c3.ope_opts_set_lb(c3_ope_opts[ii],lb)
                c3.ope_opts_set_ub(c3_ope_opts[ii],ub)
                c3.ope_opts_set_nparams(c3_ope_opts[ii],nparam)
                c3.ope_opts_set_tol(c3_ope_opts[ii],1e-10)
            elif ftype == "hermite":
                c3_ope_opts.append(c3.ope_opts_alloc(c3.HERMITE))
                c3.ope_opts_set_nparams(c3_ope_opts[ii],nparam)
                c3.ope_opts_set_tol(c3_ope_opts[ii],1e-10)
            elif ftype == "linelm":
                x = np.linspace(lb,ub,nparam)
                c3_ope_opts.append(c3.lin_elem_exp_aopts_alloc(x))
            elif ftype == "kernel":
                x = list(np.linspace(lb,ub))
                width = nparam**(-0.2) / np.sqrt(12.0) * (ub-lb)  * kernel_width_scale
                c3_ope_opts.append(
                    c3.kernel_approx_opts_gauss(nparam,
                                                x,
                                                kernel_height_scale,
                                                kernel_width_scale))
                c3.kernel_approx_opts_set_center_adapt(c3_ope_opts[ii][1],
                                                       kernel_adapt_center)
            elif ftype == "piecewise":
                nregions=20
                c3_ope_opts.append(c3.pw_poly_opts_alloc(c3.LEGENDRE,lb,ub))

                c3.pw_poly_opts_set_maxorder(c3_ope_opts[ii][1],nparam)
                c3.pw_poly_opts_set_coeffs_check(c3_ope_opts[ii][1],0)
                c3.pw_poly_opts_set_tol(c3_ope_opts[ii][1],1e-6)
                c3.pw_poly_opts_set_minsize(c3_ope_opts[ii][1],(ub-lb)/nregions)
                c3.pw_poly_opts_set_nregions(c3_ope_opts[ii][1],nregions)
            else:
                raise AttributeError('No options can be specified for function type '
                                     + ftype)
        return c3_ope_opts

    def _build_approx_params(self,method=c3.REGRESS):

        c3_ope_opts = self.__convert_opts_to_c3_form__()
        
        c3a = c3.c3approx_create(method, self.dim)
        onedopts = []
        for ii in range(self.dim):
            ope_opts = c3_ope_opts[ii]
            if self.opts[ii]['type'] == "legendre":
                onedopts.append(c3.one_approx_opts_alloc(c3.POLYNOMIAL, ope_opts))
            elif self.opts[ii]['type'] == "hermite":
                onedopts.append(c3.one_approx_opts_alloc(c3.POLYNOMIAL, ope_opts))
            elif self.opts[ii]['type'] == "linelm":
                onedopts.append(c3.one_approx_opts_alloc(c3.LINELM, ope_opts))
            elif self.opts[ii]['type'] == "kernel":
                onedopts.append(c3.one_approx_opts_alloc(c3.KERNEL, ope_opts))
            elif self.opts[ii]['type'] == "piecewise":
                onedopts.append(c3.one_approx_opts_alloc(c3.PIECEWISE, ope_opts))
            else:
                raise AttributeError("Don't know what to do here")

            c3.c3approx_set_approx_opts_dim(c3a, ii, onedopts[ii])
            
        return c3a, onedopts, c3_ope_opts

    def _free_approx_params(self, c3a, onedopts, low_opts):
        for ii in range(self.dim):
            # print("ii: ", ii)
            c3.one_approx_opts_free(onedopts[ii])
            # print("woop: !")
            if self.opts[ii]['type'] == "legendre":
                c3.ope_opts_free(low_opts[ii])
            elif self.opts[ii]['type'] == "hermite":
                c3.ope_opts_free(low_opts[ii])                
            elif self.opts[ii]['type'] == "linelm":
                c3.lin_elem_exp_aopts_free(low_opts[ii])
            elif self.opts[ii][0] == "kernel":
                c3.kernel_approx_opts_free(low_opts[ii])
            elif self.opts[ii][0] == "piecewise":
                c3.pw_poly_opts_free(low_opts[ii])
            else:
                raise AttributeError("Don't know what to do here")

        # print("OK")
        c3.c3approx_destroy(c3a)
        # print("OKkkk")
        
    def _assemble_cross_args(self, verbose, init_rank, maxrank=10, cross_tol=1e-8,
                             round_tol=1e-8, kickrank=5):
                             

        start_fibers = c3.malloc_dd(self.dim)

        c3a, onedopts, low_opts = self._build_approx_params(c3.CROSS)
        
        for ii in range(self.dim):
            c3.dd_row_linspace(start_fibers, ii, self.opts[ii]['lb'],
                               self.opts[ii]['ub'], init_rank)

        c3.c3approx_init_cross(c3a,init_rank,verbose,start_fibers);
        c3.c3approx_set_verbose(c3a,verbose);
        c3.c3approx_set_cross_tol(c3a,cross_tol);
        c3.c3approx_set_cross_maxiter(c3a,5); 
        c3.c3approx_set_round_tol(c3a,round_tol);
        c3.c3approx_set_adapt_maxrank_all(c3a,maxrank);
        c3.c3approx_set_adapt_kickrank(c3a,kickrank);

        c3.free_dd(self.dim,start_fibers)
        return c3a, onedopts, low_opts

    def build_approximation(self,f,fargs,init_rank,verbose,adapt,maxrank=10,cross_tol=1e-8,
                            round_tol=1e-8,kickrank=5):

        fobj = pcb.alloc_cobj()
        pcb.assign(fobj,self.dim,f,fargs)
        fw = c3.fwrap_create(self.dim,"python")
        c3.fwrap_set_pyfunc(fw,fobj)
        c3a, onedopts, low_opts = self._assemble_cross_args(verbose,init_rank,
                                                            maxrank=maxrank,
                                                            cross_tol=cross_tol,
                                                            round_tol=round_tol,
                                                            kickrank=kickrank)
        # print("do cross\n");
        self.ft = c3.c3approx_do_cross(c3a,fw,adapt)

        self._free_approx_params(c3a, onedopts, low_opts)
        c3.fwrap_destroy(fw)
    

    def build_data_model(self, ndata, xdata, ydata, alg="AIO", obj="LS", verbose=0,
                         opt_type="BFGS", opt_gtol=1e-10, opt_relftol=1e-10,
                         opt_absxtol=1e-30, opt_maxiter=2000, opt_sgd_learn_rate=1e-3,
                         adaptrank=0, roundtol=1e-5, maxrank=10, kickrank=2,
                         kristoffel=False, regweight=1e-7, cvnparam=None,
                         cvregweight=None, kfold=5, cvverbose=0):
        """
        Note that this overwrites multiopts, and the final rank might not be the same
        as self.rank
        """
        
        #xdata should be ndata x dim

        
        assert isinstance(xdata, np.ndarray)
        assert ydata.ndim == 1

        optimizer = None
        if opt_type == "BFGS":
            optimizer = c3.c3opt_create(c3.BFGS)
            c3.c3opt_set_absxtol(optimizer,opt_absxtol)
        elif opt_type == "SGD":
            optimizer = c3.c3opt_create(c3.SGD)
            c3.c3opt_set_sgd_nsamples(optimizer,xdata.shape[0])
            c3.c3opt_set_sgd_learn_rate(optimizer,opt_sgd_learn_rate)
            c3.c3opt_set_absxtol(optimizer,opt_absxtol)
        else:
            raise AttributeError('Optimizer:  ' + opt_type + " is unknown")
        
        if verbose > 1:
            c3.c3opt_set_verbose(optimizer,1)

        # Set optimization options
        c3.c3opt_set_gtol(optimizer,opt_gtol)
        c3.c3opt_set_relftol(optimizer,opt_relftol)
        c3.c3opt_set_maxiter(optimizer,opt_maxiter)

        c3a, onedopts, low_opts = self._build_approx_params(c3.REGRESS)
        multiopts = c3.c3approx_get_approx_args(c3a)
        
        reg = c3.ft_regress_alloc(self.dim,multiopts,self.ranks)
        if alg == "AIO" and obj == "LS":
            c3.ft_regress_set_alg_and_obj(reg,c3.AIO,c3.FTLS)
        elif alg == "AIO" and obj == "LS_SPARSECORE":
            c3.ft_regress_set_alg_and_obj(reg,c3.AIO,c3.FTLS_SPARSEL2)
            c3.ft_regress_set_regularization_weight(reg,regweight)
        elif alg == "ALS" and obj == "LS":
            c3.ft_regress_set_alg_and_obj(reg,c3.ALS,c3.FTLS)
        elif alg == "ALS" and obj == "LS_SPARSECORE":
            c3.ft_regress_set_alg_and_obj(reg,c3.ALS,c3.FTLS_SPARSEL2)
            c3.ft_regress_set_regularization_weight(reg,regweight)
        else:
            raise AttributeError('Option combination of algorithm and objective not implemented ' + alg + obj)
        if adaptrank != 0:
            c3.ft_regress_set_adapt(reg,adaptrank)
            c3.ft_regress_set_roundtol(reg,roundtol)
            c3.ft_regress_set_maxrank(reg,maxrank)
            c3.ft_regress_set_kickrank(reg,kickrank)
            
        c3.ft_regress_set_verbose(reg,verbose)

        if kristoffel is True:
            c3.ft_regress_set_kristoffel(reg,1)
                        
        if self.ft is not None:
            c3.function_train_free(self.ft)


        cv = None
        cvgrid = None
        if (cvnparam is not None) and (cvregweight is None):
            cvgrid = c3.cv_opt_grid_init(1)
            c3.cv_opt_grid_add_param(cvgrid,"num_param",len(cvnparam),list(cvnparam))
        elif (cvnparam is None) and (cvregweight is not None):
            cvgrid = c3.cv_opt_grid_init(1)
            c3.cv_opt_grid_add_param(cvgrid,"reg_weight",len(cvregweight),list(cvregweight))
        elif (cvnparam is not None) and (cvregweight is not None):
            cvgrid = c3.cv_opt_grid_init(2)
            c3.cv_opt_grid_add_param(cvgrid,"num_param",len(cvnparam),list(cvnparam))
            c3.cv_opt_grid_add_param(cvgrid,"reg_weight",len(cvregeight),list(cvnparam))


        if cvgrid is not None:
            #print("Cross validation is not working yet!\n")
            c3.cv_opt_grid_set_verbose(cvgrid,cvverbose)
            
            cv = c3.cross_validate_init(self.dim,xdata.flatten(order='C'),ydata,kfold,0)
            c3.cross_validate_grid_opt(cv,cvgrid,reg,optimizer)
            c3.cv_opt_grid_free(cvgrid)
            c3.cross_validate_free(cv)
            

        # print("Run regression")
        self.ft = c3.ft_regress_run(reg,optimizer,xdata.flatten(order='C'),ydata)
        # print("Done!")
        
        c3.ft_regress_free(reg)
        c3.c3opt_free(optimizer)

        # Free built approximation options
        # print("Free params")
        self._free_approx_params(c3a, onedopts, low_opts)
        # print("Done Free params")


    def set_ranks(self,ranks):

        if (len(ranks) != self.dim+1):
            raise AttributeError("Ranks must be a list of size dim+1, with the first and last elements = 1")

        if (isinstance(ranks,list)):
            self.ranks = copy.deepcopy(ranks)
        else:
            self.ranks = list(copy.deepcopy(ranks))
            
        if ranks[0] != 1:
            print ("Warning: rank[0] is not specified to 1, overwriting ")
            self.ranks[0] = 1
            
        if ranks[self.dim] != 1:
            print ("Warning: rank[0] is not specified to 1, overwriting ")
            self.ranks[self.dim] = 1

    def eval(self, pt):
        """ Evaluate a FunctionTrain """
        return c3.function_train_eval(self.ft,pt)

    def grad_eval(self, pt):
        """ Evaluate the gradient of a FunctionTrain """
        
        grad_out = np.zeros((self.dim))
        c3.function_train_gradient_eval(self.ft,pt,grad_out)
        return grad_out
    
    def round(self, eps=1e-14):
        """ Round a FunctionTrain """
        c3a, onedopts, low_opts = self._build_approx_params()
        multiopts = c3.c3approx_get_approx_args(c3a)
        c3.function_train_round(self.ft,eps,multiopts)
        self._free_approx_params(c3a, onedopts, low_opts)
        
    def __add__(self,other,eps=1e-14):
        """ Add two function trains """
        out = FunctionTrain(self.dim)
        out.ft = c3.function_train_sum(self.ft,other.ft)
        out.opts = copy.deepcopy(self.opts)
        out.round(eps)
        return out

    def __sub__(self,other,eps=1e-14):
        """ Subtract two function trains """

        # print("subtracting!")
        temp1 = c3.function_train_copy(other.ft)
        c3.function_train_scale(temp1, -1.0)
        
        out_ft = FunctionTrain(self.dim)
        out_ft.opts = copy.deepcopy(self.opts)
        out_ft.ft = c3.function_train_sum(self.ft, temp1)
        out_ft.round(eps)

        c3.function_train_free(temp1)
        return out_ft
        
    def __mul__(self,other,eps=1e-14):
        out = FunctionTrain(self.dim)
        out.ft = c3.function_train_product(self.ft,other.ft)
        out.opts = copy.deepcopy(self.opts)
        out.round(eps)
        return out

    def integrate(self):
        return c3.function_train_integrate(self.ft)

    def scale(self,a,eps=1e-14):
        """ f <- a*f"""
        c3.function_train_scale(self.ft,a)

    def scale_and_shift(self,scale, shift, eps=1e-14):

        c3a, onedopts, low_opts = self._build_approx_params()
        multiopts = c3.c3approx_get_approx_args(c3a)
        
        ft1 = c3.function_train_copy(self.ft)
        c3.function_train_scale(ft1,scale)
        
        ft2 = c3.function_train_constant(shift, multiopts)

        ft_out = FunctionTrain(self.dim)
        ft_out.opts = copy.deepcopy(ft_out.opts)
        ft_out.ft = c3.function_train_sum(ft1,ft2)

        c3.function_train_free(ft1)
        c3.function_train_free(ft2)
        
        c3.function_train_round(ft_out.ft, eps, multiopts)
        self._free_approx_params(c3a, onedopts, low_opts)
        return ft_out
        
    def norm2(self):
        return c3.function_train_norm2(self.ft)

    def expectation(self):
        return c3.function_train_integrate_weighted(self.ft)

    def variance(self):
        mean_val = self.expectation()
        second_moment = c3.function_train_inner_weighted(self.ft,self.ft)
        return second_moment - mean_val*mean_val

    def __del__(self):
        self.close()
        
    def close(self):
        # print("Running Cleanup ")
        if self.ft is not None:
            c3.function_train_free(self.ft)
            self.ft = None
        

        
