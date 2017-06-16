# Compressed continuous computation in python
from __future__ import print_function
import c3
import numpy as np
import copy


class FunctionTrain:

    """ 
    Need to clean up memory
    3) multipts

    Handling multiopts is not ideal. Need a way to copy multiopts, right now its memory is not freed
    """
    def __init__(self,din,filename=None,multiopts=None):



        self.dim = din
        self.opts = []
        self.onedopts = []
        self.ranks = [1]
        for ii in range(din):
            self.opts.append(None)
            self.onedopts.append(None)
            self.ranks.append(1)
            
        self.ranks.insert(din,1)
        if multiopts is None:
            self.multiopts = c3.multi_approx_opts_alloc(din)
        else:
            self.multiopts = multiopts

        self.ft = None
        if filename == None:
            self.ft = None

    def set_dim_opts(self,dim,ftype,lb=-1,ub=1,kernel_height_scale=1.0,kernel_width_scale=1.0,nparam=4,kernel_adapt_center=0):

        if self.opts[dim] is not None:
            raise AttributeError('cannot call set_dim_opts because was already called')

        if ftype == "legendre":
            self.opts.insert(dim,["poly",c3.ope_opts_alloc(c3.LEGENDRE)])
            c3.ope_opts_set_lb(self.opts[dim][1],lb)
            c3.ope_opts_set_ub(self.opts[dim][1],ub)
            c3.ope_opts_set_nparams(self.opts[dim][1],nparam)
        elif ftype == "hermite":
            self.opts.insert(dim,["poly",c3.ope_opts_alloc(c3.HERMITE)])
            c3.ope_opts_set_nparams(self.opts[dim][1],nparam)
        elif ftype == "linelm":
            x = list(np.linspace(lb,ub))
            self.opts.insert(dim,["linelm",c3.lin_elem_exp_aopts_alloc(nparam,x)])
        elif ftype == "kernel":
            x = list(np.linspace(lb,ub))
            width = nparam**(-0.2) / np.sqrt(12.0) * (ub-lb)  * kernel_width_scale
            self.opts.insert(dim,["kernel",c3.kernel_approx_opts_gauss(nparam,x,kernel_height_scale,kernel_width_scale)])
            c3.kernel_approx_opts_set_center_adapt(self.opts[dim][1],kernel_adapt_center)
        else:
            raise AttributeError('No options can be specified for function type ' + ftype)
            

    def _build_multiopts(self):
        for ii in range(self.dim):
            if self.onedopts is not None:
                c3.one_approx_opts_free(self.onedopts[ii])
            
            if self.opts[ii][0] == "poly":
                self.onedopts.insert(ii,c3.one_approx_opts_alloc(c3.POLYNOMIAL,self.opts[ii][1]))
            elif self.opts[ii][0] == "linelm":
                self.onedopts.insert(ii,c3.one_approx_opts_alloc(c3.LINELM,self.opts[ii][1]))
            elif self.opts[ii][0] == "kernel":
                self.onedopts.insert(ii,c3.one_approx_opts_alloc(c3.KERNEL,self.opts[ii][1]))
            else:
                raise AttributeError("Don't know what to do here")

            c3.multi_approx_opts_set_dim(self.multiopts,ii,self.onedopts[ii])
     
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


    def build_data_model(self,ndata,xdata,ydata,alg="AIO",obj="LS",verbose=0,\
                         opt_type="BFGS",opt_gtol=1e-10,opt_relftol=1e-10,opt_absxtol=1e-30,opt_maxiter=2000,opt_sgd_learn_rate=1e-3,\
                         adaptrank=0,roundtol=1e-5,maxrank=10,kickrank=2,\
                         kristoffel=False,regweight=1e-7,cvnparam=None,cvregweight=None,kfold=5,cvverbose=0):
        """
        Note that this overwrites multiopts, and the final rank might not be the same
        as self.rank
        """
        
        #xdata should be ndata x dim

        
        assert isinstance(xdata, np.ndarray)

        optimizer = None
        if opt_type == "BFGS":
            optimizer = c3.c3opt_create(c3.BFGS)
            c3.c3opt_set_absxtol(optimizer,opt_absxtol)
        elif opt_type == "SGD":
            optimizer = c3.c3opt_create(c3.SGD)
            c3.c3opt_set_sgd_nsamples(optimizer,xdata.shape[0])
            c3.c3opt_set_sgd_learn_rate(optimizer,1e-3)
            c3.c3opt_set_absxtol(optimizer,opt_absxtol)
        else:
            raise AttributeError('Optimizer:  ' + opt_type + " is unknown")
        
        if verbose > 1:
            c3.c3opt_set_verbose(optimizer,1)

        # Set optimization options
        c3.c3opt_set_gtol(optimizer,opt_gtol)
        c3.c3opt_set_relftol(optimizer,opt_relftol)
        c3.c3opt_set_maxiter(optimizer,opt_maxiter)

        self._build_multiopts()
        
        reg = c3.ft_regress_alloc(self.dim,self.multiopts,self.ranks)
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
            
            
        self.ft = c3.ft_regress_run(reg,optimizer,xdata.flatten(order='C'),ydata)
        
        c3.ft_regress_free(reg)
        c3.c3opt_free(optimizer)


        
    def eval(self,pt):
        return c3.function_train_eval(self.ft,pt)

    def round(self,eps=1e-14):
        c3.function_train_round(self.ft,eps,self.multiopts)
        
    def __add__(self,other,eps=1e-14):
        out = FunctionTrain(self.dim,multiopts=self.multiopts)
        out.ft = c3.function_train_sum(self.ft,other.ft)
        out.round(eps)
        return out
        
    def __mul__(self,other,eps=1e-14):
        out = FunctionTrain(self.dim,multiopts=self.multiopts)
        out.ft = c3.function_train_product(self.ft,other.ft)
        out.round(eps)
        return out

    def integrate(self):
        return c3.function_train_integrate(self.ft)

    def scale(self,a):
        c3.function_train_scale(self.ft,a)

    def norm2(self):
        return c3.function_train_norm2(self.ft)

    def expectation(self):
        return c3.function_train_integrate_weighted(self.ft)

    def close(self):


        if self.ft is not None:
            c3.function_train_free(self.ft)
            self.ft = None

            
        for ii in range(self.dim):
            if self.opts[ii] is not None:
                if (self.opts[ii][0] == "poly"):
                    c3.ope_opts_free(self.opts[ii][1])
                elif (self.opts[ii][0] == "kernel"):
                    c3.kernel_approx_opts_free(self.opts[ii][1])
                elif (self.opts[ii][0] == "linelm"):
                    c3.lin_elem_exp_aopts_free(self.opts[ii][1])
                elif (self.opts[ii][0] == "piecewise"):
                    c3.piecewise_poly_opts_free(self.opts[ii][1])

                self.opts[ii] = None

            if self.onedopts[ii] is not None:
                c3.one_approx_opts_free(self.onedopts[ii])
                self.onedopts[ii] = None

        # print ("free multiopts")
        # if self.multiopts is not None:
        #     c3.multi_approx_opts_free(self.multiopts)
        #     self.multiopts = None

        # print ("done closing")
        
