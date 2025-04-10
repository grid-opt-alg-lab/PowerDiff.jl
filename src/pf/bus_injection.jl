
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