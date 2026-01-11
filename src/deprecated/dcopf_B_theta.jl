using SparseArrays, JuMP, Ipopt
using LinearAlgebra
using PowerModels
import ParametricOptInterface as POI
# using HiGHS


"""
    Bθ formulation of the DC Power Management Problem.
"""
mutable struct DCPowerManagementProblem
    problem::Model
    p # line flows
    g # generation variable
    θ # phase angle variable
    params # parameters of the problem
    cons # constraints of the problem
end

"""
    DCPowerManagementProblem(net::Dict; τ=TAU)

    Builds a DC power management problem from a network dictionary.
"""
function DCPowerManagementProblem(net::Dict; τ=TAU)
    # Calculate the demand vector
    n = length(net["bus"])  # number of buses
    load = net["load"]
    d = _make_demand_vector(load,n)
    return DCPowerManagementProblem(net, d; τ=τ)
end

"""
    DCPowerManagementProblem(net::Dict; τ=TAU)

    Assumes that only lines that are present in the network can be switched on/off.
"""
function DCPowerManagementProblem(net::Dict, d::AbstractVector; τ=TAU)
    # Parse network data
    gen, load, branch = net["gen"], net["load"], net["branch"]

    # Dimensions
    k = length(net["gen"])
    n = length(net["bus"])
    m = length(net["branch"])

    # Make generation-to-bus incidence matrix
    G_inc = _make_gen_incidence_matrix(gen, n, k)

   

    # Branch-to-bus incidence matrix
    A = calc_basic_incidence_matrix(net)

    # Calculate the susceptance matrix B, and line susceptances
    z = calc_basic_branch_series_impedance(net)
    b = imag.(inv.(z))
    z = ones(m)
    @assert norm(calc_basic_susceptance_matrix(net) - A'*Diagonal(b)*A) < 1e-6 "Susceptance matrix B does not match A'*W*A."

    # get the cost coefficients for generation
    cq = [gen[i]["cost"][1] for i in string.(1:k)]
    cl = [gen[i]["cost"][2] for i in string.(1:k)]

    # limits
    fmax = [branch[i]["rate_a"] for i in string.(1:m)]
    gmax = [gen[i]["pmax"] for i in string.(1:k)]
    gmin = [gen[i]["pmin"] for i in string.(1:k)]

    # angle difference constraints
    Δθmax = [branch[e]["angmax"] for e in string.(1:m) ]
    Δθmin = [branch[e]["angmin"] for e in string.(1:m) ]

    # calculate reference bus
    ref_ix = reference_bus(net)["bus_i"]

    return DCPowerManagementProblem(cq,cl,d,fmax,gmax,gmin,Δθmax,Δθmin,A,G_inc,b,z; τ=τ,ref_ix=ref_ix)
end



#TODO: Add Support for ParametericOptInterface.jl
"""
Builds a parameterized B-θ formulation of the DC OPF problem.

    DCPowerManagementProblem(cq, cl, d, fmax, gmax,gmin, A, G_inc, b, z; τ=TAU)

Where:
- `cq` is the quadratic cost vector for generation.
- `cl` is the linear cost vector for generation.
- `d` is the demand vector.
- `fmax` is the maximum flow vector for lines.
- `gmax` is the maximum generation vector.
- `A` is the incidence matrix.
- `G_inc` is the generation-to-bus incidence matrix.
- `b` is the susceptance parameter vector.
- `z` is the switching parameter vector for lines (active/inactive).
- `τ` is the regularization parameter (default is global varibale `TAU`).
"""
function DCPowerManagementProblem(cq,cl,d,fmax,gmax,gmin,Δθmax,Δθmin,A,G_inc,b,z;τ=TAU,ref_ix=1)
    
    # Dimensions
    k = length(cq)  # number of generators
    (m,n) = size(A) # number of branches and buses
    @assert length(fmax) == m "Length of fmax must match number of branches."
    @assert length(gmax) == k "Length of gmax must match number of generators."
    @assert length(d) == n "Length of d must match number of buses."
    
    # Create the model
    #TODO: Make POI work for better efficiency
    # model = Model(() -> POI.Optimizer(Ipopt.Optimizer()))
    model = Model(Gurobi.Optimizer)

    # parameters 
    #TODO: Make z,b, other values work efficiently as parameters.
    @variable(model, d[1:n] in Parameter.(d))  # demand vector
    # @variable(model, z[1:m] in Parameter.(z))  # switching variable for lines 
    # @variable(model, b[1:m] in Parameter.(b)) # line susceptances
    # @variable(model, fmax[1:m] in Parameter.(fmax))  # maximum flow vector
    # @variable(model, gmax[1:k] in Parameter.(gmax))  # maximum generation vector
    # @variable(model, cq[1:k] in Parameter.(cq))  # quadratic cost vector for generation
    # @variable(model, cl[1:k] in Parameter.(cl))  # linear cost vector for generation

    # build the (parameterized) susceptance matrix
    # NOTE THE -1 ↓
    W = spdiagm(-b .* z)  # Diagonal matrix of switched susceptances
    B = sparse(A' * W * A)  # Susceptance matrix


    # Define variables
    @variable(model, f[1:m])  # line flows
    @variable(model, g[1:k])  # generation variable
    @variable(model, θ[1:n])  # phase angle variable
   
    # constraints
    power_bal = @constraint(model, 
        G_inc*g .- d .== B*θ  # Power balance equation
    )
    flow_con = @constraint(model,
        f .== W * A * θ  # flow conservation equation
    )
    line_lb = @constraint(model,
        f .>= -fmax
    )
    line_ub = @constraint(model,
        f .<= fmax
    )
    gen_lb = @constraint(model,
        g .>= gmin
    )
    gen_ub = @constraint(model,
        g .<= gmax
    )   
    
    #TODO: Add phase angle constraints (phase angle difference, angle sum to zero, etc.)
    # # Reference bus phase angle constraint
    @constraint(model, θ[ref_ix] == 0.0)  # Assuming the first bus is the reference bus
    # @constraint(model, sum(θ[i] for i=1:n) == 0)
    # Phase angle difference constraints
    # Δθ = A*θ
    phase_diff = @constraint(model, Δθmin .<= A*θ .<= Δθmax)

    # Define the objective function
    @objective(model, Min,
        sum(cq[i]*g[i]^2 + cl[i]*g[i] for i in 1:k) +  #NOTE THE LACK OF 1/2
        (1/2)*τ^2*sum(f[i]^2 for i in 1:m)  # Regularization term
    )

    # Store the parameters and constraints in dictionaries
    params = Dict(
        "G_inc" => G_inc,
        "A" => A,
        "cq" => cq,
        "cl" => cl,
        "d" => d,
        "fmax" => fmax,
        "gmax" => gmax,
        "gmin" => gmin,
        "b" => b,
        "z" => z,
        "tau" => τ,
    ) 

    # Store the constraints in a dictionary
    cons = Dict(
        "power_bal" => power_bal,
        "flow_con" => flow_con,
        "line_lb" => line_lb,
        "line_ub" => line_ub,
        "gen_lb" => gen_lb,
        "gen_ub" => gen_ub,
        "phase_diff" => phase_diff
    )

    return DCPowerManagementProblem(model, f, g, θ, params, cons)

end

"""
Given an instance of a DCPowerManagementProblem, update the parameters of the model.

    update_parameters!(model::DCPowerManagementProblem, params::Dict{String,Any})

Where:
- `model` is an instance of `DCPowerManagementProblem`.
- `new_params` is a dictionary containing the parameters to update, where keys are parameter names and values are the new values for those parameters.

This function updates the parameters of the model in place, ensuring that the model reflects the new parameter values.
If a parameter key does not exist in the model's parameters, an error is raised.
"""
function update_parameters!(model::DCPowerManagementProblem, new_param::Dict{String,Any})
    for (key, value) in new_param
        if haskey(model.params, key)
            set_parameter_value.(model.params[key], value)
        else
            @info "Parameter $key not found in model parameters, skipping"
        end
    end
end

"""
Return all of the dual variables of the constraints currently in the model.
"""
function get_duals(p::DCPowerManagementProblem)
    termination_status(p.problem) ∈ [MOI.OPTIMAL,MOI.LOCALLY_SOLVED] || error("No solutions in the model")
    ν_power_bal = dual.(p.cons["power_bal"])
    ν_flow_con = dual.(p.cons["flow_con"])
    λ_flow_ub = dual.(p.cons["line_ub"])
    λ_flow_lb = dual.(p.cons["line_lb"])
    λ_gen_lb = dual.(p.cons["gen_lb"])
    λ_gen_ub = dual.(p.cons["gen_ub"])
    λ_phase_diff = dual.(p.cons["phase_diff"])
    # Return the dual variables as a dictionary
    ν = [ν_flow_con; ν_power_bal]
    λ = [λ_flow_ub; λ_flow_lb; λ_gen_ub; λ_gen_lb]
    # Return the dual variables as a dictionary
    ϕ = [λ_phase_diff]  # phase angle difference duals
    return (λ=λ, ν=ν, ϕ=ϕ)
end

"""
Computes the locational marginal prices (LMPs) of a power management problem, given the optimal solution and the network
Returns a vector of LMPs π ∈ ℝⁿ
"""
function get_lmps(p::DCPowerManagementProblem)
    termination_status(p.problem) ∈ [MOI.OPTIMAL,MOI.LOCALLY_SOLVED] || error("No solutions in the model") 
    #TODO: Is this the right notion of congestion pricing?
    return dual.(p.cons["power_bal"]) .- p.params["A"]' * (dual.(p.cons["line_ub"]) .- dual.(p.cons["line_lb"]))
end

"""
Compute the KKT operator in place for the DCPowerManagementProblem.
    the param 
"""
function compute_kkt(p::DCPowerManagementProblem)
    # Unroll the variables and parameters
    (x, d, λ, ν, Q, G, w, h, F, n, k, m) = __unroll_variables(p)
    
    K1 = Q*x + w + G'λ + F'*ν # stationarity
    K2 = spdiagm(λ)*(G*x - h) # complementary slackness
    K3 = F*x - d # primal feasibility

    # construct the kkt operator
    dim_k = n+k+ 2m+ 2k+ n
    k = sparse([
        K1;
        K2;
        K3
    ])

    return k
end

"""
Computes the sparse Jacobian of the KKT operator for the DCPowerManagementProblem.
"""
function compute_kkt_jacobian(p::DCPowerManagementProblem)
    # Unroll the variables and parameters
    (x, d, λ, ν, Q, G, w, h, F, n, k, m) = __unroll_variables(p)

    # Compute the Jacobian of the KKT operator
    dim = n + k + 2m + 2k + n
    J = spzeros(dim, dim)

    # fill in the Jacobian:: ∂k/∂z
    J = sparse([
        Q                   G'                  F';
        spdiagm(λ)*G        spdiagm(G*x-h)      spzeros(2*(m+k),n);
        F                   spzeros(n,2*(m+k))  spzeros(n,n)
    ])

    return J

end


#TODO: Set function heading to specify that this is wrt demand
"""
Computes the sensitivities of the LMPs with respect to the demand parameter the DCPowerManagementProblem.
"""
function compute_lmp_jacobian(p::DCPowerManagementProblem)
    n = length(p.θ)
    k = length(p.g)
    m = length(p.params["fmax"])

    # calculate the jacobian
    J = compute_kkt_jacobian(p)

    # calculate the sensitivities
    y = [
        zeros(n+k,n);
        zeros(2*(m+k),n);
        -Diagonal(ones(n))
    ]

    sens = -J \ y  # solve the linear system J * sens = -y

    return sens

end


"""
Compute the "congestion weights" for the DC power management problem, which describe the weights of the Laplacian that correspond to the LMP sensitivities.
    Λ_{e,e} = λ_ub_e/(f_max_e - f_e) + λ_lb_e/(f_max_e + f_e)
"""
function compute_congestion_weights(p::DCPowerManagementProblem)
    # optimal dual variables for the upper and lower line flow constraints
    λ_flow_ub = dual.(p.cons["line_ub"])
    λ_flow_lb = dual.(p.cons["line_lb"])
    
    # line flow maximum
    fmax = p.params["fmax"]

    # line flows
    fmin = -fmax
    f = value.(p.p) # line flows --- #TODO: change the name of p to f
    m = length(f)

    # compute the congestion weights
    Λ = spzeros(m,m)
    for e in 1:m
        Λ[e,e] = λ_flow_ub[e]/(fmax[e] - f[e]) + λ_flow_lb[e]/(fmax[e] + f[e])
    end

    #TODO: FINISH THIS NOW

end


"""
Extracts the primal and dual variables, and problem parameters from the DCPowerManagementProblem instance `p`.
"""
function __unroll_variables(p::DCPowerManagementProblem)
    n = length(p.θ)
    k = length(p.g)
    m = length(p.params["fmax"])

    # Collecting the variables
    θ,g = value.(p.θ), value.(p.g) 
    duals = get_duals(p)
    λ = duals[:λ]
    ν = duals[:ν][m+1:end] # exclude the flow conservation duals (for now)
    x = [θ;g]
    

    # pull out params needed
    cl = p.params["cl"]
    cq = p.params["cq"]
    d = value.(p.params["d"])
    z = p.params["z"]
    fmax = p.params["fmax"]
    gmax = p.params["gmax"]
    gmin = p.params["gmin"]
    fmax = fmax .* z  # Apply the switching variable to the flow limits
    A = p.params["A"]
    G_inc = p.params["G_inc"]
    b = p.params["b"]
    τ = p.params["tau"]

    # make the KKT matrices
    W = spdiagm(-b .* z)  # Diagonal matrix of switched susceptances
    Q = spdiagm([τ*ones(n); cq])
    G = sparse([
        W*A  spzeros(m,k);
        -W*A  spzeros(m,k);
        spzeros(k,n) spdiagm(ones(k));
        spzeros(k,n) spdiagm(-ones(k))
    ])
    w = sparse([zeros(n); cl])
    h = sparse([fmax; fmax; gmax; -gmin])
    F = sparse([
        A'*W*A  G_inc
    ])

    return (x, d, λ, ν, Q, G, w, h, F, n, k, m)
end


"""
Make the vector of demands d ∈ R^n from the load data.
"""
function _make_demand_vector(load, n)
	n_load = length(load)
	nodes = [load[i]["load_bus"] for i in string.(1:n_load)]
	d = zeros(n)
	for (ind_d, ind_n) in enumerate(nodes)
		d[ind_n] = load[string(ind_d)]["pd"]
	end
	
	return d
end

"""
make G_inc matrix (generation to node mapping matrix) such that B*g ∈ R^n
"""
function _make_gen_incidence_matrix(gen, n, k)
	nodes = [gen[i]["gen_bus"] for i in string.(1:k)]
	
	G_inc = spzeros(n, k)
	for (ind_g, ind_n) in enumerate(nodes)
		G_inc[ind_n, ind_g] = 1.0
	end
	
	return G_inc
end


