# Locational Marginal Price (LMP) Computation

"""
    calc_lmp(sol::DCOPFSolution, net::DCNetwork)

Compute Locational Marginal Prices from DC OPF solution.

For the B-θ formulation, LMPs are computed as:
```
LMP_i = ν_i - Σₑ (A[e,i] · bₑ · zₑ · (λ_ub_e - λ_lb_e))
```

where:
- `ν_i` is the power balance dual at bus i
- `A` is the incidence matrix
- `bₑ` is the susceptance of branch e
- `zₑ` is the switching state (1 = closed)
- `λ_ub_e`, `λ_lb_e` are flow limit duals

# Returns
Vector of LMPs (length n), one per bus.
"""
function calc_lmp(sol::DCOPFSolution, net::DCNetwork)
    # LMP = ν_bal - A' * Diagonal(b .* z) * (λ_ub - λ_lb)
    # The signs depend on constraint formulation; using standard convention
    congestion_component = net.A' * Diagonal(net.b .* net.z) * (sol.λ_ub - sol.λ_lb)
    return sol.ν_bal .- congestion_component
end

"""
    calc_lmp(prob::DCOPFProblem)

Compute LMPs from a solved DC OPF problem.

# Example
```julia
prob = DCOPFProblem(net, d)
sol = solve!(prob)
lmps = calc_lmp(prob)
```
"""
function calc_lmp(prob::DCOPFProblem)
    sol = solve!(prob)
    return calc_lmp(sol, prob.network)
end

"""
    calc_lmp_from_duals(ν_bal, λ_ub, λ_lb, A, b, z)

Low-level LMP computation from dual variables and network matrices.

Useful when you have dual variables but not a full DCOPFSolution.
"""
function calc_lmp_from_duals(
    ν_bal::AbstractVector,
    λ_ub::AbstractVector,
    λ_lb::AbstractVector,
    A::AbstractMatrix,
    b::AbstractVector,
    z::AbstractVector
)
    congestion = A' * Diagonal(b .* z) * (λ_ub - λ_lb)
    return ν_bal .- congestion
end

"""
    calc_congestion_component(sol::DCOPFSolution, net::DCNetwork)

Compute the congestion component of LMPs (contribution from binding flow constraints).

This isolates the impact of transmission congestion on nodal prices.
"""
function calc_congestion_component(sol::DCOPFSolution, net::DCNetwork)
    return net.A' * Diagonal(net.b .* net.z) * (sol.λ_ub - sol.λ_lb)
end

"""
    calc_marginal_loss_component(sol::DCOPFSolution, net::DCNetwork)

For DC OPF, there are no losses (lossless model).
Returns zeros for API compatibility with AC formulations.
"""
function calc_marginal_loss_component(sol::DCOPFSolution, net::DCNetwork)
    return zeros(net.n)
end
