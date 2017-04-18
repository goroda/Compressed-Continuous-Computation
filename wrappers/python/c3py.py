# Compressed continuous computation in python
import c3
import numpy as np
import copy

class FunctionTrain:

    """ 
    Need to clean up memory
    1) optimizer
    2) ft
    3) multipts
    4) opts

    Handling multiopts is not ideal. Need a way to copy multiopts
    """
    def __init__(self,din,filename=None,multiopts=None):

        self.optimizer = c3.c3opt_create(c3.BFGS)

        self.dim = din
        self.opts = []
        self.ranks = [1]
        for ii in range(din):
            self.opts.append(None)
            self.ranks.append(1)
            
        self.ranks.insert(din,1)
        if multiopts is None:
            self.multiopts = c3.multi_approx_opts_alloc(din)
        else:
            self.multiopts = multiopts

        self.ft = None
        if filename == None:
            self.ft = None

    def set_dim_opts(self,dim,ftype,lb=-1,ub=1,nparam=4):

        if self.opts[dim] is not None:
            raise AttributeError('cannot call set_dim_opts because was already called')

        if ftype == "legendre":
            self.opts.insert(dim,("poly",c3.ope_opts_alloc(c3.LEGENDRE)))
            c3.ope_opts_set_lb(self.opts[dim][1],lb)
            c3.ope_opts_set_ub(self.opts[dim][1],ub)
            c3.ope_opts_set_nparams(self.opts[dim][1],nparam)
        elif ftype == "hermite":
            self.opts.insert(dim,("poly",c3.ope_opts_alloc(c3.HERMITE)))
        else:
            raise AttributeError('No options can be specified for function type ' + ftype)
            

    def set_ranks(self,ranks):

        if (isinstance(ranks,list)):
            self.ranks = copy.deepcopy(ranks)
        else:
            self.ranks = list(copy.deepcopy(ranks))

    def build_data_model(self,ndata,xdata,ydata,alg="AIO",obj="LS",adaptrank=0,\
                         roundtol=1e-5,maxrank=10,kickrank=2,verbose=0):
        """
        Note that this overwrites multiopts, and the final rank might not be the same
        as self.rank
        """
        
        #xdata should be ndata x dim
        
        assert isinstance(xdata, np.ndarray)

        if verbose > 1:
            c3.c3opt_set_verbose(self.optimizer,1)
        
        self.qmopts = []
        for ii in range(self.dim):
            if self.opts[ii][0] == "poly":
                self.qmopts.append(c3.one_approx_opts_alloc(c3.POLYNOMIAL,self.opts[ii][1]))
            elif self.opts[ii][0] == "linelm":
                self.qmopts.append(c3.one_approx_opts_alloc(c3.LINELM,self.opts[ii][1]))
            elif self.opts[ii][0] == "kernel":
                self.qmopts.append(c3.one_approx_opts_alloc(c3.KERNEL,self.opts[ii][1]))
            else:
                raise AttributeError("Don't know what to do here")

            c3.multi_approx_opts_set_dim(self.multiopts,ii,self.qmopts[ii])
                
        reg = c3.ft_regress_alloc(self.dim,self.multiopts,self.ranks)
        if alg == "AIO" and obj == "LS":
            c3.ft_regress_set_alg_and_obj(reg,c3.AIO,c3.FTLS)
        elif alg == "ALS" and obj == "LS":
            c3.ft_regress_set_alg_and_obj(reg,c3.ALS,c3.FTLS)
        else:
            raise AttributeError('Option combination of algorithm and objective not implemented ' + alg + obj)
        if adaptrank != 0:
            c3.ft_regress_set_adapt(reg,adaptrank)
            c3.ft_regress_set_roundtol(reg,roundtol)
            c3.ft_regress_set_maxrank(reg,maxrank)
            c3.ft_regress_set_kickrank(reg,kickrank)
            c3.ft_regress_set_verbose(reg,verbose)
            
        if self.ft is None:
            c3.function_train_free(self.ft)

        self.ft = c3.ft_regress_run(reg,self.optimizer,xdata.flatten(order='C'),ydata)

        c3.ft_regress_free(reg)

        
    def eval(self,pt):
        return c3.function_train_eval(self.ft,pt)

    def round(self,eps=1e-14):
        c3.function_train_round(self.ft,eps,self.multiopts)
        
    def __add__(self,other,eps=1e-14):
        out = FunctionTrain(self.dim,multiopts=self.multiopts)
        out.ft = c3.function_train_sum(self.ft,other.ft)
        out.round()
        return out
        
    def __mul__(self,other,eps=1e-14):
        out = FunctionTrain(self.dim,multiopts=self.multiopts)
        out.ft = c3.function_train_product(self.ft,other.ft)
        out.round()
        return out

    def integrate(self):
        return c3.function_train_integrate(self.ft)

    def scale(self,a):
        c3.function_train_scale(self.ft,a)

    def norm2(self):
        return c3.function_train_norm2(self.ft)

    def expectation(self):
        return c3.function_train_integrate_weighted(self.ft)
