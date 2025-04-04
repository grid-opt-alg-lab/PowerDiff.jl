"""
The full incidence matrix of an arbitrary n node network.
Note that every node has a self edge.
This assumes that there exists edges such that all nodes are connected to all other nodes.
"""
function full_incidence_matrix(n::Int)
    dims = (Int(n*(n+1)/2) ,n)
    A = BlockArray(spzeros(dims),vcat([dᵢ for dᵢ in n-1:-1:1],[n]),[n])
    for i in 1:n-1
        A[Block(i)][:,i] .= 1
        A[Block(i)][:,i+1:end] .= -I(n-i)
    end
    A[Block(n)] .= I(n)
    return sparse(A)
end

"""
Generic construction of Laplacian matrix with vectorized weights and the full incidence matrix.
"""
function laplacian(G,B,n)
    # construct the incidence matrix
    A = full_incidence_matrix(n)
    # construct the Laplacian
    W = sparse(Diagonal(G .+ B .* im))
    return A'*W*A
end

"""
Construct vectorized Laplacian weights from the admittance matrix. Assume that every node has a self edge.
"""
function vectorize_laplacian_weights(Y::AbstractMatrix{ComplexF64})

    @assert Y == transpose(Y) "The admittance matrix must be symmetric."
    
    # Make the vectorized weights
    G_full = real.(Y)
    B_full = imag.(Y)
    
    # Get the strictly off-diagonal entries of the lower triangular part of the Laplacian, excluding the diagonal.
    # Vectorize the weights.
    G_edges = -1 .* G_full[tril(ones(size(G_full)),-1) .== 1]
    B_edges = -1 .* B_full[tril(ones(size(B_full)),-1) .== 1]

    # Get the self edges
    Y_self = [sum(r) for r in eachrow(Y)]
    G_self = real.(Y_self)
    B_self = imag.(Y_self)

    # Construct the vectorized weights.
    G = vcat(G_edges,G_self)
    B = vcat(B_edges,B_self)

    @assert norm(laplacian(G,B,size(Y,1)) -Y) < 1e-6 "The Laplacian matrix is not close to the admittance matrix. The norm of the difference is $(norm(laplacian(G,B,size(Y,1)) -Y))."

    return G,B
end

"""
Construct vectorized laplacian weights from a PowerModels data dictionary.
"""
vectorize_laplacian_weights(net::Dict{String,Any}) = vectorize_laplacian_weights(calc_basic_admittance_matrix(net))

##### Vectorized, admittance-matrix-free, topology-agnostic power flow equations.

### Power flow equations with rectangular voltage phasors v ∈ ℂⁿ as states
p(v,G,B) = real.(Diagonal(conj.(v))*(laplacian(G,B,length(v)))*v)
q(v,G,B) = imag.(Diagonal(conj.(v))*(laplacian(G,B,length(v)))*v)
vm(v,G,B) = abs.(v)

### Power flow equations with real and imaginary parts of voltage phasors, v_re,v_im ∈ ℝⁿ as states
pf_eqns(v_re,v_im,G,B) = Diagonal(v_re .- im .* v_im)*(laplacian(G,B,length(v_re))*(v_re .+ im .* v_im))
p(v_re,v_im,G,B) = real.(Diagonal(v_re .- im .* v_im)*(laplacian(G,B,length(v_re))*(v_re .+ im .* v_im)))
q(v_re,v_im,G,B) = imag.(Diagonal(v_re .- im .* v_im)*(laplacian(G,B,length(v_re))*(v_re .+ im .* v_im)))
vm(v_re,v_im,G,B) = abs.(v_re + im .* v_im)
vm2(v_re,v_im,G,B) = (v_re.^2 + v_im.^2)

### Power flow equations in polar form
p_polar(vm,δ,G,B) = p(vm .* cis.(δ),G,B)
q_polar(vm,δ,G,B) = q(vm .* cis.(δ),G,B)

### Line flows
branch_flow(v,G,B) = Diagonal(full_incidence_matrix(length(v))*v)*(G .+ B .* im)
p_flow(v,G,B) = real.(branch_flow(v,G,B))
q_flow(v,G,B) = imag.(branch_flow(v,G,B))


### Jacobians of the power flow equations with phasor form of nodal voltages v ∈ ℂⁿ as states
∂p∂g(v,G,B) = ForwardDiff.jacobian(G -> p(v,G,B),G) # jacobian of real part of power injections w.r.t. conductance
∂p∂b(v,G,B) = ForwardDiff.jacobian(B -> p(v,G,B),B) # jacobian of real part of power injections w.r.t. susceptance
∂q∂g(v,G,B) = ForwardDiff.jacobian(G -> q(v,G,B),G) # jacobian of imaginary part of power injections w.r.t. conductance
∂q∂b(v,G,B) = ForwardDiff.jacobian(B -> q(v,G,B),B) # jacobian of imaginary part of power injections w.r.t. susceptance
∂vm∂g(v,G,B) = ForwardDiff.jacobian(G -> vm(v,G,B),G) # jacobian of voltage magnitudes w.r.t. conductance
∂vm∂b(v,G,B) = ForwardDiff.jacobian(B -> vm(v,G,B),B) # jacobian of voltage magnitudes w.r.t. susceptance
∂p∂vm(vm,δ,G,B) = ForwardDiff.jacobian(vm -> p(vm .* cis.(δ),G,B),vm)
∂q∂vm(vm,δ,G,B) = ForwardDiff.jacobian(vm -> q(vm .* cis.(δ),G,B),vm)
∂p∂δ(vm,δ,G,B) = ForwardDiff.jacobian(δ -> p(vm .* cis.(δ),G,B),δ)
∂q∂δ(vm,δ,G,B) = ForwardDiff.jacobian(δ -> q(vm .* cis.(δ),G,B),δ)

### Jacobians of the power flow equations with real and imaginary parts of voltage phasors, v_re,v_im ∈ ℝⁿ as states
∂p∂g(v_re,v_im,G,B) = ForwardDiff.jacobian(G -> p(v_re,v_im,G,B),G) # jacobian of real part of power injections w.r.t. conductance
∂p∂b(v_re,v_im,G,B) = ForwardDiff.jacobian(B -> p(v_re,v_im,G,B),B) # jacobian of real part of power injections w.r.t. susceptance
∂q∂g(v_re,v_im,G,B) = ForwardDiff.jacobian(G -> q(v_re,v_im,G,B),G) # jacobian of imaginary part of power injections w.r.t. conductance
∂q∂b(v_re,v_im,G,B) = ForwardDiff.jacobian(B -> q(v_re,v_im,G,B),B) # jacobian of imaginary part of power injections w.r.t. susceptance
∂vm∂g(v_re,v_im,G,B) = ForwardDiff.jacobian(G -> vm(v_re,v_im,G,B),G) # jacobian of voltage magnitudes w.r.t. conductance
∂vm∂b(v_re,v_im,G,B) = ForwardDiff.jacobian(B -> vm(v_re,v_im,G,B),B) # jacobian of voltage magnitudes w.r.t. susceptance
∂p∂v_re(v_re,v_im,G,B) = ForwardDiff.jacobian(v_re -> p(v_re,v_im,G,B),v_re) # jacobian of real part of power injections w.r.t. real part of voltage phasors
∂p∂v_im(v_re,v_im,G,B) = ForwardDiff.jacobian(v_im -> p(v_re,v_im,G,B),v_im) # jacobian of real part of power injections w.r.t. imaginary part of voltage phasors
∂q∂v_re(v_re,v_im,G,B) = ForwardDiff.jacobian(v_re -> q(v_re,v_im,G,B),v_re) # jacobian of imaginary part of power injections w.r.t. real part of voltage phasors
∂q∂v_im(v_re,v_im,G,B) = ForwardDiff.jacobian(v_im -> q(v_re,v_im,G,B),v_im) # jacobian of imaginary part of power injections w.r.t. 