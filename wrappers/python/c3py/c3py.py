""" Compressed Continuous Computation in python """

from __future__ import print_function

import sys
import os

import numpy as np
import _c3 as c3

import copy

import atexit
# import contextlib

def poly_randu(dim, ranks, maxorder):
    """ Random function with specified ranks and legendre polynomials of maximum order"""
    bds = c3.bounding_box_init_std(dim)
    ft = c3.function_train_poly_randu(c3.LEGENDRE, bds, ranks, maxorder)
    c3.bounding_box_free(bds)
    return FunctionTrain(dim, ft=ft)

class FunctionTrain(object):
    """ Function-Train Decompositions """

    ft = None
    def __init__(self, din, ft=None):
        """ Initialize a Function Train of dimension *din* """

        self.dim = din
        self.opts = []
        self.ranks = [1]
        for ii in range(din):
            self.opts.append(None)
            self.ranks.append(1)


        self.ft = ft
        # atexit.register(self.cleanup)

    def copy(self):
        """ Copy a function train """

        ft_out = FunctionTrain(self.dim)
        ft_out.ft = c3.function_train_copy(self.ft)
        ft_out.opts = copy.deepcopy(self.opts)
        return ft_out

    def save(self, filename):
        """ Save a function train to a file with name *filename* """
        c3.function_train_save(self.ft, filename)

    def load(self, filename):
        """ Load a function-train from a file with name *filename* """

        ft = c3.function_train_load(filename)
        self.dim = c3.function_train_get_dim(ft)
        self.ft = ft

    def set_dim_opts(self, dim, ftype, lb=-1, ub=1, nparam=4,
                     kernel_height_scale=1.0, kernel_width_scale=1.0,
                     kernel_adapt_center=0,
                     lin_elem_nodes=None, kernel_nodes=None,
                     maxnum=np.inf, coeff_check=2, tol=1e-10,
                     nregions=5):
        """ Set approximation options per dimension """

        if self.opts[dim] is not None:
            raise AttributeError('cannot call set_dim_opts because was already called')

        o = dict({})
        o['type'] = ftype
        o['lb'] = lb
        o['ub'] = ub
        o['nparam'] = nparam
        o['maxnum'] = maxnum
        o['coeff_check'] = coeff_check
        o['kernel_height_scale'] = kernel_height_scale
        o['kernel_width_scale'] = kernel_width_scale
        o['kernel_adapt_center'] = kernel_adapt_center
        o['tol'] = tol
        o['nregions'] = nregions
        if lin_elem_nodes is not None:
            assert lin_elem_nodes.ndim == 1
            lin_elem_nodes = np.unique(lin_elem_nodes.round(decimals=12))
            o['nparam'] = len(lin_elem_nodes)
            o['lb'] = lin_elem_nodes[0]
            o['ub'] = lin_elem_nodes[-1]
        else:
            lin_elem_nodes = np.linspace(lb, ub, nparam)
        o['lin_elem_nodes'] = lin_elem_nodes
        if kernel_nodes is not None:
            assert kernel_nodes.ndim == 1
            kernel_nodes = np.unique(kernel_nodes.round(decimals=12))
            o['nparam'] = len(kernel_nodes)
            o['lb'] = kernel_nodes[0]
            o['ub'] = kernel_nodes[-1]
        o['kernel_nodes'] = kernel_nodes
        self.opts.insert(dim, o)

    def __convert_opts_to_c3_form__(self):

        c3_ope_opts = []
        for ii in range(self.dim):
            ftype = self.opts[ii]['type']
            lb = self.opts[ii]['lb']
            ub = self.opts[ii]['ub']
            nparam = self.opts[ii]['nparam']
            maxnum = self.opts[ii]['maxnum']
            coeff_check = self.opts[ii]['coeff_check']
            tol = self.opts[ii]['tol']
            kernel_height_scale = self.opts[ii]['kernel_height_scale']
            kernel_width_scale = self.opts[ii]['kernel_width_scale']
            kernel_adapt_center = self.opts[ii]['kernel_adapt_center']
            lin_elem_nodes = self.opts[ii]['lin_elem_nodes']
            kernel_nodes = self.opts[ii]['kernel_nodes']
            nregions = self.opts[ii]['nregions']
            if ftype == "legendre":
                c3_ope_opts.append(c3.ope_opts_alloc(c3.LEGENDRE))
                c3.ope_opts_set_lb(c3_ope_opts[ii], lb)
                c3.ope_opts_set_ub(c3_ope_opts[ii], ub)
                c3.ope_opts_set_nparams(c3_ope_opts[ii], nparam)
                c3.ope_opts_set_coeffs_check(c3_ope_opts[ii], coeff_check)
                c3.ope_opts_set_start(c3_ope_opts[ii], nparam)
                if np.isinf(maxnum) is False:
                    c3.ope_opts_set_maxnum(c3_ope_opts[ii], maxnum)
                c3.ope_opts_set_tol(c3_ope_opts[ii], tol)
            elif ftype == "hermite":
                c3_ope_opts.append(c3.ope_opts_alloc(c3.HERMITE))
                c3.ope_opts_set_nparams(c3_ope_opts[ii], nparam)
                c3.ope_opts_set_coeffs_check(c3_ope_opts[ii], coeff_check)
                c3.ope_opts_set_start(c3_ope_opts[ii], nparam)
                if np.isinf(maxnum) is False:
                    c3.ope_opts_set_maxnum(c3_ope_opts[ii], maxnum)
                c3.ope_opts_set_tol(c3_ope_opts[ii], tol)
            elif ftype == "fourier":
                c3_ope_opts.append(c3.ope_opts_alloc(c3.FOURIER))
                c3.ope_opts_set_lb(c3_ope_opts[ii], lb)
                c3.ope_opts_set_ub(c3_ope_opts[ii], ub)
                c3.ope_opts_set_nparams(c3_ope_opts[ii], nparam)
                c3.ope_opts_set_coeffs_check(c3_ope_opts[ii], coeff_check)
                c3.ope_opts_set_start(c3_ope_opts[ii], nparam)
                if np.isinf(maxnum) is False:
                    c3.ope_opts_set_maxnum(c3_ope_opts[ii], maxnum)
                c3.ope_opts_set_tol(c3_ope_opts[ii], tol)
            elif ftype == "linelm":
                c3_ope_opts.append(c3.lin_elem_exp_aopts_alloc(lin_elem_nodes))
            elif ftype == "kernel":
                if kernel_adapt_center == 0:
                    if kernel_nodes is not None:
                        x = list(kernel_nodes)
                    else:
                        x = list(np.linspace(lb, ub, nparam))
                    nparam = len(x)
                    std = np.std(x)
                    # print("standard deviation = ", std,  (ub-lb)/np.sqrt(12.0))
                    width = nparam**(-0.2) / np.sqrt(12.0) * (ub-lb)  * kernel_width_scale
                    width = nparam**(-0.2) * std * kernel_width_scale
                    c3_ope_opts.append(c3.kernel_approx_opts_gauss(nparam, x,
                                                                   kernel_height_scale,
                                                                   kernel_width_scale))
                else:
                    if kernel_nodes is not None:
                        x = list(kernel_nodes)
                        n2 = len(x)
                        nparam = 2*n2
                    else:
                        # print("here!!")
                        assert nparam % 2 == 0, "number of parameters has to be even for adaptation"
                        n2 = int(nparam/2)
                        x = list(np.linspace(lb, ub, n2))
                        # print("x = ", x)
                    std = np.std(x)
                    # width = (n2)**(-0.2) / np.sqrt(12.0) * (ub-lb)  * kernel_width_scale
                    width = n2**(-0.2) * std * kernel_width_scale
                    c3_ope_opts.append(c3.kernel_approx_opts_gauss(n2, x,
                                                                   kernel_height_scale,
                                                                   kernel_width_scale))
                    c3.kernel_approx_opts_set_center_adapt(c3_ope_opts[-1], kernel_adapt_center)
            elif ftype == "piecewise":
                c3_ope_opts.append(c3.pw_poly_opts_alloc(c3.LEGENDRE, lb, ub))
                c3.pw_poly_opts_set_maxorder(c3_ope_opts[-1], nparam)
                c3.pw_poly_opts_set_coeffs_check(c3_ope_opts[-1], coeff_check)
                c3.pw_poly_opts_set_tol(c3_ope_opts[-1], tol)
                # c3.pw_poly_opts_set_minsize(c3_ope_opts[-1], ((ub-lb)/nregions)**8)
                # c3.pw_poly_opts_set_minsize(c3_ope_opts[-1], ((ub-lb)/nregions)**3)
                c3.pw_poly_opts_set_minsize(c3_ope_opts[-1], ((ub-lb)/nregions)**2)                
                c3.pw_poly_opts_set_minsize(c3_ope_opts[-1], (ub-lb))
                c3.pw_poly_opts_set_nregions(c3_ope_opts[-1], nregions)
            else:
                raise AttributeError('No options can be specified for function type '
                                     + ftype)
        return c3_ope_opts

    def _build_approx_params(self, method=c3.REGRESS):

        c3_ope_opts = self.__convert_opts_to_c3_form__()

        c3a = c3.c3approx_create(method, self.dim)
        onedopts = []
        optnodes = []
        for ii in range(self.dim):
            ope_opts = c3_ope_opts[ii]
            if self.opts[ii]['type'] == "legendre":
                onedopts.append(c3.one_approx_opts_alloc(c3.POLYNOMIAL, ope_opts))
            elif self.opts[ii]['type'] == "hermite":
                onedopts.append(c3.one_approx_opts_alloc(c3.POLYNOMIAL, ope_opts))
            elif self.opts[ii]['type'] == "fourier":
                onedopts.append(c3.one_approx_opts_alloc(c3.POLYNOMIAL, ope_opts))
            elif self.opts[ii]['type'] == "linelm":
                onedopts.append(c3.one_approx_opts_alloc(c3.LINELM, ope_opts))
            elif self.opts[ii]['type'] == "kernel":
                onedopts.append(c3.one_approx_opts_alloc(c3.KERNEL, ope_opts))
            elif self.opts[ii]['type'] == "piecewise":
                onedopts.append(c3.one_approx_opts_alloc(c3.PIECEWISE, ope_opts))
            else:
                raise AttributeError("Don't know what to do here")


            lb = self.opts[ii]['lb']
            ub = self.opts[ii]['ub']
            if self.opts[ii]['type'] == "hermite" or self.opts[ii]['type'] == "fourier":
                nn = 50
                x = c3.linspace(lb, ub, nn)
                cv = c3.c3vector_alloc(nn, x)
                optnodes.append((x,cv))
                c3.c3approx_set_opt_opts_dim(c3a, ii, optnodes[-1][1])
                
            c3.c3approx_set_approx_opts_dim(c3a, ii, onedopts[ii])

        return c3a, onedopts, c3_ope_opts, optnodes

    def _free_approx_params(self, c3a, onedopts, low_opts, optnodes):
        for ii in range(self.dim):
            # print("ii: ", ii)
            c3.one_approx_opts_free(onedopts[ii])
            # print("woop: !")
            if self.opts[ii]['type'] == "legendre":
                c3.ope_opts_free(low_opts[ii])
            elif self.opts[ii]['type'] == "hermite":
                c3.ope_opts_free(low_opts[ii])
            elif self.opts[ii]['type'] == "fourier":
                c3.ope_opts_free(low_opts[ii])
            elif self.opts[ii]['type'] == "linelm":
                c3.lin_elem_exp_aopts_free(low_opts[ii])
            elif self.opts[ii]['type'] == "kernel":
                c3.kernel_approx_opts_free(low_opts[ii])
            elif self.opts[ii]['type'] == "piecewise":
                c3.pw_poly_opts_free(low_opts[ii])
            else:
                raise AttributeError("Don't know what to do here")

        #NOT FREING optnodes[ii][0]
        for ii in range(len(optnodes)):
            # c3.free(optnodes[ii][0])
            c3.c3vector_free(optnodes[ii][1])
            
        # print("OK")
        c3.c3approx_destroy(c3a)
        # print("OKkkk")

    def _assemble_cross_args(self, verbose, init_rank, maxrank=10, cross_tol=1e-8,
                             round_tol=1e-8, kickrank=5, maxiter=5):


        start_fibers = c3.malloc_dd(self.dim)

        c3a, onedopts, low_opts, optnodes = self._build_approx_params(c3.CROSS)


        for ii in range(self.dim):
            c3.dd_row_linspace(start_fibers, ii, self.opts[ii]['lb'],
                               self.opts[ii]['ub'], init_rank);

        
            # c3.dd_row_linspace(start_fibers, ii, 0.8,
            #                    1.2, init_rank)
            
        # NEED TO FREE OPTNODES
        
        c3.c3approx_init_cross(c3a, init_rank, verbose, start_fibers)
        c3.c3approx_set_verbose(c3a, verbose)
        c3.c3approx_set_cross_tol(c3a, cross_tol)
        c3.c3approx_set_cross_maxiter(c3a, maxiter)
        c3.c3approx_set_round_tol(c3a, round_tol)
        c3.c3approx_set_adapt_maxrank_all(c3a, maxrank)
        c3.c3approx_set_adapt_kickrank(c3a, kickrank)

        c3.free_dd(self.dim, start_fibers)
        return c3a, onedopts, low_opts, optnodes

    def build_approximation(self, f, fargs, init_rank, verbose, adapt,
                            maxrank=10, cross_tol=1e-8,
                            round_tol=1e-8, kickrank=5, maxiter=5):
        """ Build an adaptive approximation of *f* """

        fw = c3.fwrap_create(self.dim, "python")
        c3.fwrap_set_pyfunc(fw, f, fargs)
        c3a, onedopts, low_opts, optnodes = self._assemble_cross_args(verbose,
                                                                      init_rank,
                                                            maxrank=maxrank,
                                                            cross_tol=cross_tol,
                                                            round_tol=round_tol,
                                                            kickrank=kickrank,
                                                            maxiter=maxiter)
        # print("do cross\n");
        self.ft = c3.c3approx_do_cross(c3a, fw, adapt)

        self._free_approx_params(c3a, onedopts, low_opts, optnodes)
        c3.fwrap_destroy(fw)

    def build_data_model(self, ndata, xdata, ydata, alg="AIO", obj="LS", verbose=0,
                         opt_type="BFGS", opt_gtol=1e-10, opt_relftol=1e-10,
                         opt_absxtol=1e-30, opt_maxiter=2000, opt_sgd_learn_rate=1e-3,
                         adaptrank=0, roundtol=1e-5, maxrank=10, kickrank=2,
                         kristoffel=False, regweight=1e-7, cvnparam=None,
                         cvregweight=None, kfold=5, cvverbose=0, als_max_sweep=20,
                         cvrank=None, norm_ydata=False, store_opt_info=False):
        """
        Note that this overwrites multiopts, and the final rank might not be the same
        as self.rank

        xdata should be ndata x dim
        """

        assert isinstance(xdata, np.ndarray)
        assert ydata.ndim == 1

        optimizer = None
        if opt_type == "BFGS":
            optimizer = c3.c3opt_create(c3.BFGS)
            c3.c3opt_set_absxtol(optimizer, opt_absxtol)
            c3.c3opt_ls_set_maxiter(optimizer, 300)
            # c3.c3opt_ls_set_alpha(optimizer, 0.1)
            # c3.c3opt_ls_set_beta(optimizer, 0.5)
        elif opt_type == "SGD":
            optimizer = c3.c3opt_create(c3.SGD)
            c3.c3opt_set_sgd_nsamples(optimizer, xdata.shape[0])
            c3.c3opt_set_sgd_learn_rate(optimizer, opt_sgd_learn_rate)
            c3.c3opt_set_absxtol(optimizer, opt_absxtol)
        else:
            raise AttributeError('Optimizer:  ' + opt_type + " is unknown")

        if store_opt_info is True:
            c3.c3opt_set_storage_options(optimizer, 1, 0, 0)
            
        if verbose > 1:
            c3.c3opt_set_verbose(optimizer, 1)

        # Set optimization options
        c3.c3opt_set_gtol(optimizer, opt_gtol)
        c3.c3opt_set_relftol(optimizer, opt_relftol)
        c3.c3opt_set_maxiter(optimizer, opt_maxiter)

        c3a, onedopts, low_opts, opt_opts = self._build_approx_params(c3.REGRESS)
        multiopts = c3.c3approx_get_approx_args(c3a)

        reg = c3.ft_regress_alloc(self.dim, multiopts, self.ranks)
        if alg == "AIO" and obj == "LS":
            c3.ft_regress_set_alg_and_obj(reg, c3.AIO, c3.FTLS)
        elif alg == "AIO" and obj == "LS_SPARSECORE":
            c3.ft_regress_set_alg_and_obj(reg, c3.AIO, c3.FTLS_SPARSEL2)
            c3.ft_regress_set_regularization_weight(reg, regweight)
        elif alg == "ALS" and obj == "LS":
            c3.ft_regress_set_alg_and_obj(reg, c3.ALS, c3.FTLS)
            c3.ft_regress_set_max_als_sweep(reg, als_max_sweep)
        elif alg == "ALS" and obj == "LS_SPARSECORE":
            c3.ft_regress_set_alg_and_obj(reg, c3.ALS, c3.FTLS_SPARSEL2)
            c3.ft_regress_set_regularization_weight(reg, regweight)
            c3.ft_regress_set_max_als_sweep(reg, als_max_sweep)
        else:
            raise AttributeError('Option combination of algorithm and objective not implemented '\
                                 + alg + obj)
        if alg == 'ALS':
            c3.ft_regress_set_als_conv_tol(reg, opt_relftol)
            
        if adaptrank != 0:
            c3.ft_regress_set_adapt(reg, adaptrank)
            c3.ft_regress_set_roundtol(reg, roundtol)
            c3.ft_regress_set_maxrank(reg, maxrank)
            c3.ft_regress_set_kickrank(reg, kickrank)
            c3.ft_regress_set_kfold(reg, kfold)

        c3.ft_regress_set_verbose(reg, verbose)

        if kristoffel is True:
            c3.ft_regress_set_kristoffel(reg, 1)

        if self.ft is not None:
            c3.function_train_free(self.ft)



        cv = None
        cvgrid = None
        if (cvnparam is not None) and (cvregweight is None) and (cvrank is None):
            cvgrid = c3.cv_opt_grid_init(1)
            c3.cv_opt_grid_add_param(cvgrid, "num_param", len(cvnparam), list(cvnparam))
        elif (cvnparam is None) and (cvregweight is not None) and (cvrank is None):
            cvgrid = c3.cv_opt_grid_init(1)
            c3.cv_opt_grid_add_param(cvgrid, "reg_weight", len(cvregweight), list(cvregweight))
        elif (cvnparam is not None) and (cvregweight is not None) and (cvrank is None):
            cvgrid = c3.cv_opt_grid_init(2)
            c3.cv_opt_grid_add_param(cvgrid, "num_param", len(cvnparam), list(cvnparam))
            c3.cv_opt_grid_add_param(cvgrid, "reg_weight", len(cvregeight), list(cvnparam))
        elif (cvnparam is not None) and (cvrank is not None):
            cvgrid = c3.cv_opt_grid_init(2)
            c3.cv_opt_grid_add_param(cvgrid, "rank", len(cvrank), list(cvrank))
            c3.cv_opt_grid_add_param(cvgrid, "num_param", len(cvnparam), list(cvnparam))

        
        

        yuse = ydata
        if norm_ydata is True:
            vmin = np.min(ydata)
            vmax = np.max(ydata)
            vdiff = vmax - vmin
            assert (vmax - vmin) > 1e-14
            yuse = ydata / vdiff - vmin / vdiff

        if cvgrid is not None:
            #print("Cross validation is not working yet!\n")
            c3.cv_opt_grid_set_verbose(cvgrid, cvverbose)

            cv = c3.cross_validate_init(self.dim, xdata.flatten(order='C'), yuse, kfold, cvverbose)
            c3.cross_validate_grid_opt(cv, cvgrid, reg, optimizer)
            c3.cv_opt_grid_free(cvgrid)
            c3.cross_validate_free(cv)


        # print("Run regression")
        self.ft = c3.ft_regress_run(reg, optimizer, xdata.flatten(order='C'), yuse)
        # print("Done!")


        if norm_ydata is True: # need to unnormalize
            ft_use = self.scale_and_shift(vdiff, vmin, c3_pointer=True)
            c3.function_train_free(self.ft)
            self.ft = ft_use

        if store_opt_info is True:
            if alg == "ALS":
                nepoch = c3.ft_regress_get_nepochs(reg)
                results = np.zeros((nepoch))
                for ii in range(nepoch):
                    results[ii] = c3.ft_regress_get_stored_fvals(reg, ii)
            else:
                nepoch = c3.c3opt_get_niters(optimizer)
                results = np.zeros((nepoch))
                for ii in range(nepoch):
                    results[ii] = c3.c3opt_get_stored_function(optimizer, ii)

            # print("nepoch", nepoch)

        c3.ft_regress_free(reg)
        c3.c3opt_free(optimizer)

        # Free built approximation options
        # print("Free params")
        self._free_approx_params(c3a, onedopts, low_opts, opt_opts)
        # print("Done Free params")

        if store_opt_info is True:
            return results


    def set_ranks(self, ranks):

        if len(ranks) != self.dim+1:
            raise AttributeError("Ranks must be a list of size dim+1, \
            with the first and last elements = 1")

        if isinstance(ranks, list):
            self.ranks = copy.deepcopy(ranks)
        else:
            self.ranks = list(copy.deepcopy(ranks))

        if ranks[0] != 1:
            print("Warning: rank[0] is not specified to 1, overwriting ")
            self.ranks[0] = 1

        if ranks[self.dim] != 1:
            print("Warning: rank[0] is not specified to 1, overwriting ")
            self.ranks[self.dim] = 1

    def eval(self, pt):
        """ Evaluate a FunctionTrain """
        assert isinstance(pt, np.ndarray)
        if pt.ndim == 1:
            return c3.function_train_eval(self.ft, pt)
        else:
            assert pt.shape[1] == self.dim
            out = np.zeros((pt.shape[0]))
            for ii, p in enumerate(pt):
                out[ii] = c3.function_train_eval(self.ft, p)
            return out

    def grad_eval(self, pt):
        """ Evaluate the gradient of a FunctionTrain """

        grad_out = np.zeros((self.dim))
        c3.function_train_gradient_eval(self.ft, pt, grad_out)
        return grad_out

    def round(self, eps=1e-14):
        """ Round a FunctionTrain """

        c3a, onedopts, low_opts, optopts = self._build_approx_params()
        multiopts = c3.c3approx_get_approx_args(c3a)
        ftc = c3.function_train_round(self.ft, eps, multiopts)
        c3.function_train_free(self.ft)
        # c3.function_train_free(self.ft)
        self.ft = ftc
        self._free_approx_params(c3a, onedopts, low_opts, optopts)

    def __add__(self, other, eps=0):
        """ Add two function trains """

        out = FunctionTrain(self.dim)
        out.ft = c3.function_train_sum(self.ft, other.ft)
        out.opts = copy.deepcopy(self.opts)
        out.round(eps)
        return out

    def __sub__(self, other, eps=0):
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

    def __mul__(self, other, eps=0):
        out = FunctionTrain(self.dim)
        out.ft = c3.function_train_product(self.ft, other.ft)
        out.opts = copy.deepcopy(self.opts)
        out.round(eps)
        return out

    def integrate(self):
        return c3.function_train_integrate(self.ft)

    def inner(self, other):
        return c3.function_train_inner(self.ft, other.ft)
    
    def scale(self, a, eps=0):
        """ f <- a*f"""
        c3.function_train_scale(self.ft, a)
        return self

    def scale_and_shift(self, scale, shift, eps=0, c3_pointer=False):

        c3a, onedopts, low_opts = self._build_approx_params()
        multiopts = c3.c3approx_get_approx_args(c3a)

        ft1 = c3.function_train_copy(self.ft)
        c3.function_train_scale(ft1, scale)

        ft2 = c3.function_train_constant(shift, multiopts)

        ft_out = FunctionTrain(self.dim)
        ft_out.opts = copy.deepcopy(self.opts)
        ft_out.ft = c3.function_train_sum(ft1, ft2)

        c3.function_train_free(ft1)
        c3.function_train_free(ft2)

        c3.function_train_round(ft_out.ft, eps, multiopts)
        self._free_approx_params(c3a, onedopts, low_opts)

        if c3_pointer is True:
            ft_ret = c3.function_train_copy(ft_out.ft)
            return ft_ret
        else:
            return ft_out

    def get_ranks(self):
        dim = c3.function_train_get_dim(self.ft)
        ranks = [1]*(dim+1)
        for ii in range(dim+1):
            ranks[ii] = c3.function_train_get_rank(self.ft, ii)
        return np.array(ranks)
        
    def norm2(self):
        return c3.function_train_norm2(self.ft)

    def expectation(self):
        return c3.function_train_integrate_weighted(self.ft)
    
    def variance(self):
        mean_val = self.expectation()
        second_moment = c3.function_train_inner_weighted(self.ft, self.ft)
        return second_moment - mean_val*mean_val

    def get_uni_func(self, dim, row, col):
        return GenericFunction(c3.function_train_get_gfuni(self.ft, dim, row, col))

    def laplace(self, eps=0.0):
        ft_out = FunctionTrain(self.dim)

        c3a, onedopts, low_opts, opt_opts = self._build_approx_params(c3.REGRESS)
        multiopts = c3.c3approx_get_approx_args(c3a)


        ft_out.ft = c3.exact_laplace(self.ft, multiopts)
        ft_out.opts = copy.deepcopy(self.opts)

        self._free_approx_params(c3a, onedopts, low_opts, opt_opts)

        if eps > 0.0:
            ft_out.round(eps=eps)

        return ft_out
        
        
    def __del__(self):
        self.close()

    def close(self):
        # print("Running Cleanup ")
        if self.ft is not None:
            c3.function_train_free(self.ft)
            self.ft = None



            
class GenericFunction(object):
    """ Univariate Functions """

    gf = None
    def __init__(self, gf):
        self.gf = c3.generic_function_copy(gf)

    def eval(self, x):
        if type(x) == list:
            return [c3.generic_function_1d_eval(self.gf, xx) for xx in x]
        elif type(x) == np.ndarray:
            assert x.ndim == 1, "Only 1d arrays handled"
            return np.array([c3.generic_function_1d_eval(self.gf, xx) for xx in x])
        else:
            return c3.generic_function_1d_eval(self.gf, x)
        
    def __del__(self):
        self.close()
        
    def close(self):
        if self.gf is not None:
            c3.generic_function_free(self.gf)
            self.gf = None

class SobolIndices(object):
    """ Sobol Sensitivity Indices """

    si = None
    def __init__(self, ft, order=None):
        if order is None:
            order = ft.dim
        self.si = c3.c3_sobol_sensitivity_calculate(ft.ft, order)

    def get_total_sensitivity(self, index):
        return c3.c3_sobol_sensitivity_get_total(self.si, index)

    def get_main_sensitivity(self, index):
        return c3.c3_sobol_sensitivity_get_main(self.si, index)

    def get_variance(self):
        return c3.c3_sobol_sensitivity_get_variance(self.si)
    
    def get_interaction(self, variables):
        # assert len(variables) == 2, "Only effects between two variables is currently supported"
        for ii in range(len(variables)-1):
            assert variables[ii] < variables[ii+1], \
                "Sobol index variables must be ordered with v[i] < V[j] for i < j"
        
        ret = c3.c3_sobol_sensitivity_get_interaction(self.si, variables)
        return ret
        
    def __del__(self):
        self.close()
        
    def close(self):
        if self.si is not None:
            c3.c3_sobol_sensitivity_free(self.si)
            self.si = None
