function add(x::Array{Cdouble})
    
    return x[1]*x[2] + x[1]^2 + x[2]^2

end

function fwrap(fin_::Ptr{Void}, dim::Csize_t, x_::Ptr{Cdouble} )
    
    x = pointer_to_array(x_,dim)
    fin = unsafe_pointer_to_objref(fin_)::Function
    out = fin(x)::Float64
    return out
end

function integrate(f::Function, dim, ranksin, lb::Array{Cdouble}, 
                ub::Array{Cdouble}, adapt_max_iter=10, 
                adapt_thresh=1e-4, cross_epsilon=1e-6, verbose=1,N=100)
    
    ranks = convert(Array{Csize_t},ranksin)::Array{Csize_t}
    fc = cfunction(fwrap,Float64,(Ptr{Void}, Csize_t, Ptr{Cdouble}))
    t = ccall((:integrate_easy,"../build/src/lib_funcdecomp/libfuncdecomp"),
              Float64,(Csize_t, Ptr{Csize_t}, Ptr{Cdouble}, Ptr{Cdouble},
              Csize_t, Cdouble, Cdouble, Cint, Csize_t, Any, Ptr{Void},), 
              dim, ranks, lb, ub, adapt_max_iter, 
              adapt_thresh, cross_epsilon, verbose, N, f, fc)
    
    println("Ranks in integrate are $ranks");
    #ranksin = ranks
    return t, ranks
end

ranks = [1 1 1]
lb = [0.0 0.0]
ub = [1.0 1.0]
dim = 2
value, ranks =  integrate(add,dim, ranks, lb, ub)
println("Value of integral is $value")
println("Ranks are $ranks")
