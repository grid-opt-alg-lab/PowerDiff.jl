using PowerModels
using ForwardDiff
using SparseArrays
using LinearAlgebra
include("../src/pf/admittance_matrix.jl")
include("../src/pf/bus_injection.jl")
include("../src/graphs/laplacian.jl")
include("../src/pf/admittance_matrix.jl")


#setup network
net = make_basic_network(parse_file("/home/sam/Research/HOSProject/data/case33bw.m"))
compute_ac_pf!(net)
u = calc_basic_bus_voltage(net)
u_re,u_im = real.(u),imag.(u)

#topology params
Y = calc_basic_admittance_matrix(net)
G,B = real.(Y),imag.(Y)
Gvec,Bvec = vectorize_laplacian_weights(net)

#setup power flow equations
f(u_re,u_im,Gvec_,Bvec_) = [
    p(u_re,u_im,Gvec_,Bvec_);
    q(u_re,u_im,Gvec_,Bvec_)
]

f(u_re,u_im,Gvec,Bvec)

s_test = calc_basic_bus_injection(net)
@assert norm([real.(s_test) ; imag.(conj.(s_test))] - f(u_re,u_im,Gvec,Bvec))/norm(s_test) < 1e-6

#setup hessian
function pf_hessian(u_re,u_im,Gvec,Bvec)
    @assert length(u_re) == length(u_im) "Real and imaginary parts of voltage phasors must be the same length."
    n = length(u_re)
    # Compute the Hessians of the active power equations (third order tensor)
    hessians = zeros(n,n,2n)
    for i =1:n
        hessians[i,:] = ForwardDiff.hessian(
            x -> p(x[1:n], x[n+1:end], Gvec, Bvec)[i],
            [u_re; u_im]
        )
    end

    # Compute the Hessians of the reactive power equations (third order tensor)
    hess = ForwardDiff.hessian(
        x -> f(x[1:n], x[n+1:end], Gvec, Bvec),
        [u_re; u_im]
    )
    
    # Reshape the Hessian into a square matrix
    hess = reshape(hess, 2*n, 2*n)
    
    return hess
end
