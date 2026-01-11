#### Power network  struct
TAU = 1e-2 # 1e-3

mutable struct PowerNetwork
    fq
    fl
    pmax
    gmax
    A
    B
    F
    τ
end

#### Power management problem (just DC OPF with the possibility of storage)
mutable struct PowerManagementProblem
    problem::Model
    p # line flows
    g # generation variable
    s # storage variable
    ch # charge variable
    dis # discharge variable
    params # parameters of the problem
    cons # constraints of the problem
end


"""
    PowerNetwork(fq, fl, pmax, gmax, A, B, F; τ=TAU)

Create a power network with quadratic and linear prices `fq` and `fl`, line capacities
`pmax`, generation capacities `gmax`, incidence matrix `A`, node-generator matrix `B`,
and PFDF matrix `F`. 

Optionally, quadratically penalize power flows with weight `(1/2) τ^2`.
"""
PowerNetwork(fq, fl, pmax, gmax, A, B, F; τ=TAU) =
    PowerNetwork(fq, fl, pmax, gmax, A, B, F, τ)

"""
Given a PowerModels network, return a PowerNetwork object.
"""
function PowerNetwork(net::Dict;τ=TAU)

    if !haskey(net, "basic_network") || (haskey(net,"basic_network") && net["basic_network"] == false) 
        # work as if it is not basic network

    elseif haskey(net,"basic_network") && net["basic_network"] == true 
        # work as if it is a basic network --- unchanged
        # Parse network data
        gen, load, branch = net["gen"], net["load"], net["branch"]

        # Dimensions
        k = length(net["gen"])
        n = length(net["bus"])
        m = length(net["branch"])

        # Get matrices
        F = PowerModels.calc_basic_ptdf_matrix(net)
        A = calc_basic_incidence_matrix(net)' # NOTE THE TRANSPOSE
        B = _make_B(gen, n, k)

        # Get limits
        pmax = [branch[i]["rate_a"] for i in string.(1:m)] # maximum line flow
        gmax = [gen[i]["pmax"] for i in string.(1:k)] # maximum generation

        # Get costs
        fq = [gen[i]["cost"][1] for i in string.(1:k)]
        fl = [gen[i]["cost"][2] for i in string.(1:k)]

        # Get demand
        d = _make_d(load, n)

    else 
        error("Cannot determine if the network is basic or not. Please provide a valid PowerModels network dictionary.")
    end
    
    
    return PowerNetwork(
        fq,
        fl,
        pmax,
        gmax,
        A,
        B,
        F,
        τ
    )
end


"""
    PowerManagementProblem(fq, fl, d, pmax, gmax, A, B, F; τ=TAU, ch=0, dis=0, S=0, η_c=1.0, η_d=1.0)

Set up a static DC OPF problem with the following parameters: 
- `fq`: quadratic generator costs
- `fl`: linear generator costs
- `d`: nodal demand
- `pmax`: maximum line flow limit
- `gmax`: generation capacities
- `A`: network incidence matrix
- `B`: generator-to-node matrix
- `F`: PFDF matrix
- `τ`: regularization weight used to make the problem strongly convex by adding τ ∑ᵢ pᵢ² to the objective
- `ch`: additional charge over one timestep in the batteries of the network
- `dis`: additional discharge over onetimestep in the batteries of the network
- `S`: battery-to-node matrix
- `η_c` and `η_d`: battery charge and discharge efficiencies
"""
function PowerManagementProblem(fq, fl, d, pmax, gmax, A, B, F; τ=TAU, ch=0, dis=0, S=0, η_c=1.0, η_d=1.0)
    n,m = size(A) #note this is the transpose of the A matrix in the paper
    n,k = size(B)

    #setup the JuMP model
    problem = Model(Ipopt.Optimizer)
    set_optimizer_attribute(problem, "tol", 1e-8)
    set_silent(problem)

    #define the variables
    @variable(problem,p[i=1:m])
    @variable(problem,g[i=1:k])
    
    #define the objective
    @objective(problem,Min,sum((1/2)*fq[i]*g[i]^2 + fl[i]*g[i] for i=1:k) + (1/2)*τ^2*sum(p[i]^2 for i=1:m))
    
    #define the constraints
    line_lb = @constraint(problem,line_lb,-p .≤ pmax)
    line_ub = @constraint(problem,line_ub,p .≤ pmax)
    gen_lb = @constraint(problem,gen_lb,g >= 0)
    gen_ub = @constraint(problem,gen_ub,g .≤ gmax)
    flow_con = @constraint(problem,flow_con,p .== F*(B*g .- d .- S*(ch .- dis))) # ν
    power_bal = @constraint(problem,power_bal,0 == ones(n)'*(B*g .- d .- S*(ch .- dis))) # νE

    params = (fq=fq, fl=fl, d=d, pmax=pmax, gmax=gmax, A=A, B=B, F=F, S=S, τ=τ, η_c=η_c, η_d=η_d)
   
    constraints = (line_lb=line_lb, line_ub=line_ub, gen_lb=gen_lb, gen_ub = gen_ub, flow_con = flow_con, power_bal = power_bal)
   
    return PowerManagementProblem(
        problem, 
        p, 
        g, 
        zeros(n), 
        zeros(n), 
        zeros(n), 
        params,
        constraints
    )
end

"""
Given a powermodels dictionary, make a powermanagementproblem with optional demand parameter available (d)
"""
function PowerManagementProblem(net::Dict{String,Any};d=nothing)
    net = PowerNetwork(net)
    if isnothing(d)
        return PowerManagementProblem(net.fq, net.fl, net.d, net.pmax, net.gmax, net.A, net.B, net.F; τ=net.τ)
    else
        return PowerManagementProblem(net.fq, net.fl, d, net.pmax, net.gmax, net.A, net.B, net.F; τ=net.τ)
    end
end

PowerManagementProblem(net::PowerNetwork,d::AbstractArray)= PowerManagementProblem(net.fq, net.fl, d, net.pmax, net.gmax, net.A, net.B, net.F; τ=net.τ)

"""
Get the dual problem of the power management problem
"""
function build_dual_problem(p::PowerManagementProblem)
    dual_problem = Dualization.dualize(p.problem,Gurobi.Optimizer)
    return dual_problem
end

"""
Return all of the dual variables of the constraints currently in the model.
"""
function get_duals(p::PowerManagementProblem)
    termination_status(p.problem) ∈ [MOI.OPTIMAL,MOI.LOCALLY_SOLVED] || error("No solutions in the model")
    ν_power_bal = shadow_price.(p.cons[:power_bal])
    ν_flow_con = shadow_price.(p.cons[:flow_con])
    λ_flow_ub = shadow_price.(p.cons[:line_ub])
    λ_flow_lb = shadow_price.(p.cons[:line_lb])
    return (λ=[λ_flow_ub;λ_flow_lb],ν=[ν_flow_con;ν_power_bal])
end

"""
Computes the LMPs of a power management problem, given the optimal solution and the network
Returns a vector of LMPs π ∈ ℝⁿ
"""
function get_lmps(p::PowerManagementProblem)
    termination_status(p.problem) ∈ [MOI.OPTIMAL,MOI.LOCALLY_SOLVED] || error("No solutions in the model")
    λ_bal = shadow_price.(p.cons[:power_bal])
    λ_flow_ub = shadow_price.(p.cons[:line_ub])
    λ_flow_lb = shadow_price.(p.cons[:line_lb])
    π_lmp = λ_bal .- p.params[:F]'*(λ_flow_ub .- λ_flow_lb)
    return π_lmp
end

"""
    kkt_dims(n, m, l)

Compute the dimensions of the input / output of the KKT operator for
a network with `n` nodes and `m` edges and `l` generator per node

"""
kkt_dims(n, m, l) = 4m + 3l + 1

"""
Computes the KKT operator
"""
function kkt(x,fq,fl,d,pmax,gmax,A,B,F; τ=TAU, ch=0, dis=0, S=0)
    m,n = size(F)
    n,k = size(B)
    g, p, λ_line_lb, λ_line_ub, λg_lb, λg_ub, ν, νE = unflatten_variables(x, n, m, k)

    # Lagragian is
    # L = J + λpl'(-p - pmax) + ... + λgu'(g - gmax) + v'(Ap - g - d) + νF'(p - F*(B*g - d))
    return [
        Diagonal(fq)*g + fl - λg_lb + λg_ub - B'*F'*ν + νE[1]*B'*ones(n); # stationarity wrt g
        ν + λ_line_ub - λ_line_lb + τ*p; # stationarity wrt p
        λ_line_lb .* (-p - pmax);
        λ_line_ub .* (p - pmax);
        -λg_lb .* g;
        λg_ub .* (g - gmax);
        p - F*(B*g - d .- S*ch .+ S*dis);
        ones(n)'*(B*g - d .- S*ch .+ S*dis)
    ]


end

kkt(x, net::PowerNetwork, d) =
    kkt(x, net.fq, net.fl, d, net.pmax, net.gmax, net.A, net.B, net.F; τ=net.τ)

"""
flatten_variables(P::PowerManagementProblem)

Concatenates primal and dual variables from `P` into a single vector.
"""
function flatten_variables(P::PowerManagementProblem)
    x = [
        value.(P.g) ; 
        value.(P.p)
        ]
    λ = [
        dual.(P.cons[:line_lb]) ;
        dual.(P.cons[:line_ub]) ;
        dual.(P.cons[:gen_lb]) ;
        dual.(P.cons[:gen_ub]) ;
        dual.(P.cons[:flow_con]) ;
        dual.(P.cons[:power_bal])
        ]
    return [x; λ][:, 1]
end


"""
    unflatten_variables(x, n, m, k)

Extracts primal and dual variables from `x`.
"""
function unflatten_variables(x, n, m, k)
    i = 0
    
    g =  x[i+1:i+k]
    i += k
    p = x[i+1:i+m]
    i += m
    
    λpl = x[i+1:i+m]
    i += m
    λpu = x[i+1:i+m]
    i += m
    
    λgl = x[i+1:i+k]
    i += k
    λgu = x[i+1:i+k]
    i += k
    
    ν = x[i+1:i+m]
    i += m

    νE = x[i+1:i+1]
    i += 1

    return g, p, λpl, λpu, λgl, λgu, ν, νE
end




"""
Make the vector of demands d ∈ R^n from the load data.
"""
function _make_d(load, n)
	n_load = length(load)
	nodes = [load[i]["load_bus"] for i in string.(1:n_load)]
	d = zeros(n)
	for (ind_d, ind_n) in enumerate(nodes)
		d[ind_n] = load[string(ind_d)]["pd"]
	end
	
	return d
end

"""
make B matrix (generation to node mapping matrix) such that B*g ∈ R^n
"""
function _make_B(gen, n, k)
	nodes = [gen[i]["gen_bus"] for i in string.(1:k)]
	
	B = spzeros(n, k)
	for (ind_g, ind_n) in enumerate(nodes)
		B[ind_n, ind_g] = 1.0
	end
	
	return B
end
