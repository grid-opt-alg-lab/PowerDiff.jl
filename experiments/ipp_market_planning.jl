# Copyright 2026 Samuel Talkington and contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# IPP market aware transmission upgrade planning via Frank-Wolfe
# =============================================================================
#
# Bilevel program (weighted-hub IPP formulation):
#
#     min_{u ∈ U}     w_i' λ*(u)           (outer)
#     s.t.            λ*(u) ∈ OPF(u)        (inner: full DC OPF)
#                     fmax = fmax_0 + u
#
# The i-th IPP at bus i ∈ N assigns importance weights ω_j ≥ 0 to "hub" buses
# j ∈ H_i ⊆ N (with Σ_j ω_j = 1). The IPP's exposure vector w_i ∈ ℝ^n is
#
#     (w_i)_j = -1     if j == i               (IPP's own bus)
#     (w_i)_j = ω_j    if j ∈ H_i              (hub portfolio)
#     (w_i)_j = 0      otherwise
#
# Economic interpretation: w_i' λ = Σ_{j∈H_i} ω_j λ_j − λ_i, the basis between
# the IPP's local LMP and a weighted basket of hub LMPs. Min w'λ ≡ minimise
# the IPP's negative basis (equivalently, maximise the IPP's local-vs-hub
# premium, λ_i − Σ_j ω_j λ_j).
#
# Algorithm: Frank-Wolfe over the budget simplex
#   U = { u : u_e ≥ 0,  Σ u_e ≤ B,  u_e ≤ Δmax_e }
#
# Per FW iteration: update_fmax! → solve! → vjp!(:lmp, :fmax, w) → LMO → step.
# Matrix free gradient: O(nnz) per iteration via PowerDiff's transpose KKT solve.

using PowerDiff
using PowerModels
using LinearAlgebra
using Printf
using Statistics
using SparseArrays
using Logging
using Random
using DelimitedFiles
using CairoMakie
using LaTeXStrings

const PM = PowerModels
PM.silence()

# =============================================================================
# IPP state
# =============================================================================
#
# Each tier explicitly chooses an IPP node `ipp_node` and a hub set `hub_nodes`
# (with weights `ω`). Helpers `pick_default_ipp_hub` and `build_ipp_state` are
# provided. A degenerate setup (hub at slack, IPP at slack) leads cong_h ≡ 0
# and "do nothing" — flagged as a warning by `pick_default_ipp_hub` if it
# happens to land there.

struct IPPState
    fmax_0::Vector{Float64}
    Δmax::Vector{Float64}
    B::Float64
    H::Vector{Int}            # hub bus *original* IDs (sorted, length k)
    w::Vector{Float64}        # length n: -1 at ipp_seq, ω_j at H_seq, 0 else
    ipp_node::Int             # IPP's *original* bus ID
    ipp_seq::Int              # IPP's *sequential* bus index
    H_seq::Vector{Int}        # hub *sequential* bus indices, aligned with H and ω
    ω::Vector{Float64}        # hub weights (length k), all ≥ 0, Σω = 1
end

# Copy-with-budget convenience (used by the Pareto sweep)
function IPPState(st::IPPState; B::Float64=st.B)
    return IPPState(st.fmax_0, st.Δmax, B, st.H, st.w,
                    st.ipp_node, st.ipp_seq, st.H_seq, st.ω)
end

"""
    build_ipp_state(prob; ipp_node, hub_nodes, hub_weights=nothing,
                    B_frac=0.5, Δmax_factor=2.0)

Construct an IPPState from explicit IPP and hub bus *original* IDs. `hub_weights`
defaults to a uniform basket (1/|hub_nodes|) when not provided; otherwise must
be nonneg and sum to 1.
"""
function build_ipp_state(prob::DCOPFProblem;
                         ipp_node::Int,
                         hub_nodes::AbstractVector{<:Integer},
                         hub_weights::Union{Nothing,AbstractVector{<:Real}}=nothing,
                         B_frac::Float64=0.5,
                         Δmax_factor::Float64=2.0)
    net = prob.network
    n      = net.n
    fmax_0 = copy(net.fmax)
    Δmax   = Δmax_factor .* fmax_0
    B      = B_frac * sum(fmax_0)

    haskey(net.id_map.bus_to_idx, ipp_node) || error("IPP bus $ipp_node not in network")
    ipp_seq = net.id_map.bus_to_idx[ipp_node]

    H = collect(Int, hub_nodes)
    isempty(H) && error("hub_nodes must be non-empty")
    H_seq = Int[]
    for h in H
        haskey(net.id_map.bus_to_idx, h) || error("Hub bus $h not in network")
        seq = net.id_map.bus_to_idx[h]
        seq == ipp_seq && error("IPP node $ipp_node cannot also be a hub")
        push!(H_seq, seq)
    end
    k = length(H)
    ω = if hub_weights === nothing
        fill(1.0/k, k)
    else
        ωv = collect(Float64, hub_weights)
        length(ωv) == k || error("hub_weights length ($(length(ωv))) must match hub_nodes ($k)")
        all(ωv .≥ 0)     || error("hub_weights must be non-negative")
        isapprox(sum(ωv), 1.0; atol=1e-10) || error("hub_weights must sum to 1, got $(sum(ωv))")
        ωv
    end

    w = zeros(n)
    w[ipp_seq] = -1.0
    for (s, ωj) in zip(H_seq, ω)
        w[s] = ωj
    end

    return IPPState(fmax_0, Δmax, B, H, w, ipp_node, ipp_seq, H_seq, ω)
end

"""
    pick_default_ipp_hub(prob; k_hub=1, raw=nothing, by_zone=false)
        → (ipp_node, hub_nodes, hub_weights)

Solve the baseline OPF and pick:
- `ipp_node` (orig ID) = generator bus with the lowest baseline LMP (the most
  most congested out cheap generator — the canonical IPP whose value shows up as a
  basis discount).
- `hub_nodes` (orig IDs):
    * if `by_zone=true`: highest-LMP bus in each of the top-k_hub zones
      (by max-zone LMP), determined from `raw["bus"][...][zone]`.
    * else: top-`k_hub` highest-LMP buses overall (excluding ipp_node).
- `hub_weights` = uniform 1/k_hub.

Generator bearing buses are derived from `net.G_inc` (n×k sparse, entry
[bus_seq, gen_seq] = 1 ⇒ generator at that bus).
"""
function pick_default_ipp_hub(prob::DCOPFProblem;
                              k_hub::Int=1,
                              raw::Union{Nothing,AbstractDict}=nothing,
                              by_zone::Bool=false)
    net = prob.network
    sol = with_logger(SimpleLogger(stderr, Logging.Warn)) do
        PowerDiff.solve!(prob)
    end
    λ = sol.nu_bal

    # Sequential bus indices that have at least one generator.
    gen_seqs = unique(rowvals(net.G_inc))
    isempty(gen_seqs) && error("Network has no generators")
    # IPP at the gen bearing bus with the lowest baseline LMP.
    ipp_seq  = gen_seqs[argmin(λ[gen_seqs])]
    ipp_orig = net.id_map.bus_ids[ipp_seq]

    if by_zone
        raw === nothing && error("by_zone=true requires the PowerModels raw dict")
        # Group sequential bus indices by zone (excluding the IPP bus).
        zones = Dict{Int,Vector{Int}}()
        for (key, businfo) in raw["bus"]
            orig = key isa Integer ? Int(key) : parse(Int, string(key))
            haskey(net.id_map.bus_to_idx, orig) || continue
            seq = net.id_map.bus_to_idx[orig]
            seq == ipp_seq && continue
            z = Int(businfo["zone"])
            push!(get!(zones, z, Int[]), seq)
        end
        isempty(zones) && error("No zones found in raw bus data")
        # For each zone, take its highest-LMP bus; rank zones by that LMP; keep top k_hub.
        zone_top = [(z, seqs[argmax(λ[seqs])]) for (z, seqs) in zones]
        sort!(zone_top; by = zs -> -λ[zs[2]])
        pick = zone_top[1:min(k_hub, length(zone_top))]
        hub_seqs = [zs[2] for zs in pick]
    else
        sorted = sortperm(λ; rev=true)
        kept = filter(s -> s != ipp_seq, sorted)
        length(kept) ≥ k_hub || error("Not enough non-IPP buses to pick $k_hub hubs")
        hub_seqs = kept[1:k_hub]
    end
    hub_orig = [net.id_map.bus_ids[s] for s in hub_seqs]
    hub_weights = fill(1.0 / k_hub, k_hub)

    # Sanity warnings
    if ipp_seq == net.ref_bus
        @warn "Default IPP node landed on the slack bus" ipp_orig=ipp_orig
    end
    if any(s -> s == net.ref_bus, hub_seqs)
        @warn "A default hub landed on the slack bus (cong ≡ 0)" hub_origs=hub_orig
    end

    return ipp_orig, hub_orig, hub_weights
end

# =============================================================================
# Frank-Wolfe core
# =============================================================================

"""
Linear minimisation oracle over U = {fmax = fmax_0 + Δ : Δ_e ≥ 0,
Σ Δ_e ≤ B, Δ_e ≤ Δmax_e}.

Vertices: (a) fmax_0 (no upgrade) or (b) fmax_0 + min(B, Δmax[e*])·e_{e*}.
Returns vertex `v` (in fmax-space) and the chosen edge index e*.
"""
function lmo_budget!(v::Vector{Float64}, g::Vector{Float64},
                     fmax_0::Vector{Float64}, Δmax::Vector{Float64}, B::Float64)
    e_star = argmin(g)
    copyto!(v, fmax_0)
    if g[e_star] < 0
        v[e_star] += min(B, Δmax[e_star])
    end
    return v, e_star
end

"""
    fw_ipp!(prob, st; max_iters, tol, capex_α, capex_c, demand_periods, verbose)

Frank-Wolfe outer loop. Mutates `prob` (via `update_fmax!`) and returns
(fmax_star, history).

- `capex_α > 0`: adds α·c'(fmax-fmax_0) penalty to the objective and gradient.
- `demand_periods::Vector{Vector{Float64}}`: averages gradient and objective
  across periods using `update_demand!`. Each entry is a length-`n` demand
  vector. If `nothing`, runs a single period FW with the prob's current `d`.
"""
function _eval_obj!(prob, st, fmax_try, capex_α, capex_c, demand_periods)
    update_fmax!(prob, fmax_try)
    obj = 0.0
    if demand_periods === nothing
        sol = with_logger(SimpleLogger(stderr, Logging.Warn)) do
            PowerDiff.solve!(prob)
        end
        obj = dot(st.w, sol.nu_bal)
    else
        T = length(demand_periods)
        for t in 1:T
            update_demand!(prob, demand_periods[t])
            sol = with_logger(SimpleLogger(stderr, Logging.Warn)) do
                PowerDiff.solve!(prob)
            end
            obj += dot(st.w, sol.nu_bal) / T
        end
    end
    if capex_α > 0 && capex_c !== nothing
        obj += capex_α * dot(capex_c, fmax_try .- st.fmax_0)
    end
    return obj
end

function fw_ipp!(prob::DCOPFProblem, st::IPPState;
                  max_iters::Int=50, tol::Float64=1e-6,
                  capex_α::Float64=0.0, capex_c::Union{Nothing,Vector{Float64}}=nothing,
                  demand_periods::Union{Nothing,Vector{Vector{Float64}}}=nothing,
                  step_rule::Symbol=:armijo,   # :armijo (backtrack) or :simple (2/(k+2))
                  verbose::Bool=true)
    m = prob.network.m
    # Save the caller's demand so we can restore it on exit when the multi
    # period branch overwrites prob.d via update_demand!.
    d_entry = demand_periods === nothing ? nothing : copy(prob.d)
    fmax_k = copy(st.fmax_0)
    update_fmax!(prob, fmax_k)

    g    = zeros(m)
    g_t  = zeros(m)
    work = zeros(kkt_dims(prob))
    v    = similar(fmax_k)
    fmax_try = similar(fmax_k)

    obj_hist = Float64[]
    gap_hist = Float64[]
    estar_hist = Int[]
    γ_hist = Float64[]
    Δnorm_hist = Float64[]
    fmax_hist = zeros(m, 0)

    # Best iterate tracking (handles non-convex w'λ from active set jumps)
    fmax_best = copy(fmax_k)
    obj_best  = Inf

    if verbose
        @printf("  %-5s  %-13s  %-13s  %-9s  %-7s  %-7s  %-13s\n",
                "Iter", "Objective", "FW Gap", "Σ Δ", "γ", "e*", "Best obj")
        println("  " * "-"^80)
    end

    for k in 0:max_iters-1
        obj = 0.0
        fill!(g, 0.0)
        if demand_periods === nothing
            sol = with_logger(SimpleLogger(stderr, Logging.Warn)) do
                PowerDiff.solve!(prob)
            end
            obj = dot(st.w, sol.nu_bal)
            with_logger(SimpleLogger(stderr, Logging.Warn)) do
                vjp!(g, prob, :lmp, :fmax, st.w; work=work)
            end
        else
            T = length(demand_periods)
            for t in 1:T
                update_demand!(prob, demand_periods[t])
                sol = with_logger(SimpleLogger(stderr, Logging.Warn)) do
                    PowerDiff.solve!(prob)
                end
                obj += dot(st.w, sol.nu_bal) / T
                fill!(g_t, 0.0)
                with_logger(SimpleLogger(stderr, Logging.Warn)) do
                    vjp!(g_t, prob, :lmp, :fmax, st.w; work=work)
                end
                @. g += g_t / T
            end
        end

        if capex_α > 0 && capex_c !== nothing
            obj += capex_α * dot(capex_c, fmax_k .- st.fmax_0)
            @. g += capex_α * capex_c
        end

        # Track best iterate
        if obj < obj_best
            obj_best = obj
            fmax_best .= fmax_k
        end

        _, e_star = lmo_budget!(v, g, st.fmax_0, st.Δmax, st.B)
        gap = dot(g, fmax_k .- v)

        push!(obj_hist, obj)
        push!(gap_hist, gap)
        push!(estar_hist, e_star)
        push!(Δnorm_hist, sum(fmax_k .- st.fmax_0))
        fmax_hist = hcat(fmax_hist, copy(fmax_k))

        if gap ≤ tol * (1 + abs(obj))
            if verbose
                γ_disp = isempty(γ_hist) ? NaN : γ_hist[end]
                @printf("  %-5d  %-13.6f  %-13.4e  %-9.4f  %-7.4f  %-7d  %-13.6f\n",
                        k, obj, gap, Δnorm_hist[end], γ_disp, e_star, obj_best)
                println("  Converged at iteration $k (gap ≤ tol).")
            end
            break
        end

        # Step. Armijo: backtrack from γ_full = 2/(k+2), halving until obj
        # decreases. Handles active set kinks where the linearization stops
        # being valid mid step.
        γ = if step_rule == :armijo
            γ_try = 2.0 / (k + 2)
            best_γ = γ_try
            best_δ = +Inf
            descent = false
            for _ in 1:8
                @. fmax_try = (1 - γ_try) * fmax_k + γ_try * v
                obj_try = _eval_obj!(prob, st, fmax_try, capex_α, capex_c, demand_periods)
                if obj_try < obj
                    best_γ = γ_try; best_δ = obj_try - obj
                    descent = true
                    break
                else
                    if obj_try - obj < best_δ
                        best_γ = γ_try; best_δ = obj_try - obj
                    end
                    γ_try /= 2.0
                end
            end
            descent || @warn "Armijo backtrack exhausted at iter $k without descent; taking smallest γ" γ=best_γ Δobj=best_δ
            best_γ
        else
            2.0 / (k + 2)
        end

        push!(γ_hist, γ)

        if verbose
            @printf("  %-5d  %-13.6f  %-13.4e  %-9.4f  %-7.4f  %-7d  %-13.6f\n",
                    k, obj, gap, Δnorm_hist[end], γ, e_star, obj_best)
        end

        @. fmax_k = (1 - γ) * fmax_k + γ * v
        update_fmax!(prob, fmax_k)
    end

    # Restore prob to best iterate; restore caller demand if multi period.
    update_fmax!(prob, fmax_best)
    if d_entry !== nothing
        update_demand!(prob, d_entry)
    end
    invalidate!(prob.cache)

    history = (
        obj=obj_hist, gap=gap_hist, estar=estar_hist,
        γ=γ_hist, Δnorm=Δnorm_hist, fmax_hist=fmax_hist,
        obj_best=obj_best,
    )
    return fmax_best, history
end

# =============================================================================
# CSV writer
# =============================================================================

function write_history_csv(history, path::String)
    open(path, "w") do io
        println(io, "iter,objective,fw_gap,delta_norm,gamma,e_star")
        K = length(history.obj)
        for k in 1:K
            γ = k == 1 ? NaN : history.γ[k-1]
            @printf(io, "%d,%.10f,%.10e,%.10f,%.10f,%d\n",
                    k-1, history.obj[k], history.gap[k],
                    history.Δnorm[k], γ, history.estar[k])
        end
    end
end

# =============================================================================
# Tier 1: 3-bus pedagogical demo
# =============================================================================

"""
3-bus congested network with quadratic generation cost (so LMPs are smooth in
fmax — `cq=0` would give piecewise-constant LMPs and zero gradient within an
active set, which is mathematically correct but not a useful demo).

Cheap gen at bus 1, expensive at bus 2, single load at bus 3. Line 1→3
saturates at fmax=0.5; relaxing it routes more cheap power to bus 3.
"""
function build_3bus()
    n, m, k = 3, 2, 2
    A = sparse([
        1.0  0.0 -1.0;   # Line 1: 1→3 (congested)
        0.0  1.0 -1.0    # Line 2: 2→3
    ])
    G_inc = sparse([
        1.0 0.0;
        0.0 1.0;
        0.0 0.0
    ])
    b = [-10.0, -10.0]
    net = DCNetwork(n, m, k, A, G_inc, b;
        fmax=[0.5, 10.0],
        gmax=[2.0, 2.0], gmin=[0.0, 0.0],
        cl=[10.0, 50.0], cq=[1.0, 1.0],   # quadratic → smooth LMPs
        ref_bus=1, tau=0.0)
    return net
end

function run_tier1(; outdir::String=@__DIR__)
    println("\n" * "="^65)
    println("Tier 1: 3-bus pedagogical demo")
    println("="^65)

    net = build_3bus()
    d   = [0.05, 0.05, 1.0]
    prob = DCOPFProblem(net, d)

    sol = with_logger(SimpleLogger(stderr, Logging.Warn)) do
        PowerDiff.solve!(prob)
    end
    lmp_base = copy(sol.nu_bal)

    println("\nBaseline LMPs (bus 1, 2, 3): ", round.(lmp_base, digits=4))
    println("  λ_3 - λ_1 = ", round(lmp_base[3] - lmp_base[1], digits=4),
            "   ← congestion rent")

    # ── IPP at bus 1 (cheap exporter); hub at bus 3 (load pocket) ─────────────
    # Weights: w[1]=-1 (IPP), w[3]=+1 (single hub, ω=1), w[2]=0.
    ipp_node = 1
    hub_nodes = [3]
    println("\nIPP setup:")
    println("  IPP node i  = $ipp_node  (cheap exporter, bus 1)")
    println("  Hub set H_i = $hub_nodes  (load pocket, bus 3); ω = [1.0]")
    println("  w_i = [-1, 0, +1]    (weight at bus 2 is 0)")
    println("  IPP basis = w_i'λ = λ_3 - λ_1;  IPP premium = -w_i'λ = λ_1 - λ_3")

    # ── Full Jacobian display + FD verification ────────────────────────────────
    dlmp_dfmax = calc_sensitivity(prob, :lmp, :fmax)
    println("\nFull ∂λ/∂fmax (3 buses × 2 lines):")
    show(stdout, "text/plain", round.(Matrix(dlmp_dfmax), digits=4)); println()

    ε = 1e-5
    fd = zeros(3, 2)
    fmax_baseline = [0.5, 10.0]
    for e in 1:2
        fmax_p = copy(fmax_baseline)   # always perturb from the baseline, not last iterate
        fmax_p[e] += ε
        update_fmax!(prob, fmax_p)
        sol_p = with_logger(SimpleLogger(stderr, Logging.Warn)) do
            PowerDiff.solve!(prob)
        end
        fd[:, e] = (sol_p.nu_bal .- lmp_base) ./ ε
    end
    update_fmax!(prob, fmax_baseline)  # restore
    with_logger(SimpleLogger(stderr, Logging.Warn)) do
        PowerDiff.solve!(prob)
    end

    println("\nFinite difference reference:")
    show(stdout, "text/plain", round.(fd, digits=4)); println()
    err = maximum(abs.(Matrix(dlmp_dfmax) .- fd))
    @printf("\nmax |analytical − FD| = %.3e   (tol 1e-3)\n", err)
    err < 1e-3 || @warn "Tier 1 FD verification failed" err

    # Reset cache (so the in loop VJP takes the matrix free path, not the cached matrix)
    invalidate!(prob.cache)

    # ── Frank-Wolfe ────────────────────────────────────────────────────────────
    println("\nFrank-Wolfe (B = 1.5, Δmax = [2,2]):")
    # Build IPP state with explicit overrides for the 3-bus pedagogical sizing.
    st_auto = build_ipp_state(prob; ipp_node=ipp_node, hub_nodes=hub_nodes)
    st = IPPState(copy(prob.network.fmax), [2.0, 2.0], 1.5, st_auto.H, st_auto.w,
                  st_auto.ipp_node, st_auto.ipp_seq, st_auto.H_seq, st_auto.ω)
    @assert st.w == [-1.0, 0.0, +1.0] "3-bus weight vector mismatch: $(st.w)"
    fmax_star, hist = fw_ipp!(prob, st; max_iters=10, tol=1e-8)

    println("\nUpgrade plan:")
    @printf("  Branch  fmax_0   fmax*    Δ\n")
    for e in 1:2
        @printf("  %-7d  %-7.3f  %-7.3f  %.3f\n",
                e, st.fmax_0[e], fmax_star[e], fmax_star[e]-st.fmax_0[e])
    end

    sol_star = with_logger(SimpleLogger(stderr, Logging.Warn)) do
        PowerDiff.solve!(prob)
    end
    println("\nLMPs (baseline → optimized):")
    for i in 1:3
        @printf("  bus %d:  %.4f → %.4f   Δ = %+.4f\n",
                i, lmp_base[i], sol_star.nu_bal[i], sol_star.nu_bal[i]-lmp_base[i])
    end
    @printf("\nObjective w_i'λ (basis):  baseline = %.4f, optimized = %.4f\n",
            dot(st.w, lmp_base), dot(st.w, sol_star.nu_bal))
    @printf("IPP premium -w_i'λ:        baseline = %.4f, optimized = %.4f  (Δ = %+.4f)\n",
            -dot(st.w, lmp_base), -dot(st.w, sol_star.nu_bal),
            dot(st.w, lmp_base) - dot(st.w, sol_star.nu_bal))

    # ── Plot ───────────────────────────────────────────────────────────────────
    plot_3bus(hist, lmp_base, sol_star.nu_bal,
              joinpath(outdir, "ipp_market_planning_3bus");
              ipp_seq=st.ipp_seq, hub_seqs=st.H_seq)

    write_history_csv(hist, joinpath(outdir, "ipp_history_3bus.csv"))
    println()
    return prob, st, fmax_star, hist
end

# =============================================================================
# Tier 2: case14
# =============================================================================

function run_tier2(; outdir::String=@__DIR__,
                     ipp_node::Union{Nothing,Int}=nothing,
                     hub_nodes::Union{Nothing,Vector{Int}}=nothing,
                     fmax_scale::Float64=0.10)
    println("\n" * "="^65)
    println("Tier 2: case14, rate_a × $(fmax_scale)  (single period + capex Pareto)")
    println("="^65)

    case_path = joinpath(dirname(pathof(PM)), "..", "test", "data", "matpower", "case14.m")
    raw = PM.parse_file(case_path)
    PM.make_basic_network!(raw)        # populates rate_a defaults
    # case14 ships with very loose flow limits (max loading ~15% on the binding
    # line at default rate_a). Scale by 0.10 to bring 2 lines to the bound and
    # produce meaningful LMP variation. This is the "stressed" scenario IPPs
    # actually care about — peak loading, outages, etc.
    for (_, br) in raw["branch"]
        br["rate_a"] *= fmax_scale
    end
    net = DCNetwork(raw)
    # Break generator degeneracy. Without this the KKT system is singular at
    # the optimum (multiple gens at upper bound), Tikhonov regularization kicks
    # in, and the matrix free VJP returns essentially zero gradient.
    for i in eachindex(net.gmax)
        if net.gmax[i] > 0.01
            net.gmax[i] *= 3.0
            net.gmin[i] = max(net.gmin[i], 0.01)
        end
    end
    d   = calc_demand_vector(net)
    n, m = net.n, net.m
    println("Network: $n buses, $m branches, $(net.k) gens.  Total demand = $(round(sum(d), digits=3))")

    prob = DCOPFProblem(net, d)

    # Pick IPP node and single hub (k=1) by default; allow override.
    ipp, hubs, hub_w = if ipp_node === nothing && hub_nodes === nothing
        pick_default_ipp_hub(prob; k_hub=1)
    elseif ipp_node !== nothing && hub_nodes !== nothing
        ipp_node, hub_nodes, fill(1.0/length(hub_nodes), length(hub_nodes))
    else
        error("Pass both ipp_node and hub_nodes, or neither")
    end
    st = build_ipp_state(prob; ipp_node=ipp, hub_nodes=hubs, hub_weights=hub_w,
                         B_frac=0.5, Δmax_factor=2.0)

    # Re-solve at fmax_0 to get baseline LMPs (pick_default_ipp_hub already solved once)
    sol_base = with_logger(SimpleLogger(stderr, Logging.Warn)) do
        PowerDiff.solve!(prob)
    end
    lmp_base = copy(sol_base.nu_bal)

    println("\nIPP setup:")
    println("  IPP node i = $(st.ipp_node)   (lowest-LMP gen bus, λ_i = $(round(lmp_base[st.ipp_seq], digits=4)))")
    println("  Hub set H_i = $(st.H)        ω = $(round.(st.ω, digits=3))   (single hub, k=1)")
    println("  λ at hub: ", round.(lmp_base[st.H_seq], digits=4))
    println("  Baseline basis  w_i'λ = $(round(dot(st.w, lmp_base), digits=4))   (= λ_h - λ_i)")
    println("  Budget B = $(round(st.B, digits=3))")

    # Reset cache so in loop VJP is matrix free
    invalidate!(prob.cache)

    # ── Pure spread (capex_α = 0) ─────────────────────────────────────────────
    println("\nPure spread: min w'λ")
    fmax_star, hist = fw_ipp!(prob, st; max_iters=80, tol=1e-7)

    sol_star = with_logger(SimpleLogger(stderr, Logging.Warn)) do
        PowerDiff.solve!(prob)
    end

    println("\nUpgrade plan (top 6 by Δ):")
    Δ = fmax_star .- st.fmax_0
    perm = sortperm(Δ; rev=true)
    @printf("  Branch  fmax_0   fmax*    Δ        %% increase\n")
    for e in perm[1:min(6, m)]
        Δ[e] < 1e-6 && break
        @printf("  %-7d  %-7.3f  %-7.3f  %-7.3f  %+5.0f%%\n",
                net.id_map.branch_ids[e], st.fmax_0[e], fmax_star[e], Δ[e],
                100*Δ[e]/st.fmax_0[e])
    end
    @printf("\nTotal Δ = %.3f  (budget B = %.3f, %.0f%% used)\n",
            sum(Δ), st.B, 100*sum(Δ)/st.B)
    @printf("Objective w_i'λ (basis):  baseline = %.4f, optimized = %.4f\n",
            dot(st.w, lmp_base), dot(st.w, sol_star.nu_bal))
    @printf("IPP premium -w_i'λ improvement: %+.4f\n",
            dot(st.w, lmp_base) - dot(st.w, sol_star.nu_bal))

    # FD verification at fmax* on top-5 branches
    println("\nFD verification at fmax* (top 5 by |Δ|):")
    obj_star = dot(st.w, sol_star.nu_bal)
    g_star = zeros(m); work = zeros(kkt_dims(prob))
    invalidate!(prob.cache)
    PowerDiff.solve!(prob)
    with_logger(SimpleLogger(stderr, Logging.Warn)) do
        vjp!(g_star, prob, :lmp, :fmax, st.w; work=work)
    end
    ε = 1e-5
    @printf("  Branch    VJP           FD            |Δ|\n")
    for e in perm[1:min(5, m)]
        Δ[e] < 1e-6 && break
        fmax_p = copy(fmax_star); fmax_p[e] += ε
        update_fmax!(prob, fmax_p)
        sol_p = with_logger(SimpleLogger(stderr, Logging.Warn)) do
            PowerDiff.solve!(prob)
        end
        fd_e = (dot(st.w, sol_p.nu_bal) - obj_star) / ε
        @printf("  %-8d  %-+13.4e  %-+13.4e  %.2e\n", e, g_star[e], fd_e, abs(g_star[e]-fd_e))
    end
    update_fmax!(prob, fmax_star)

    # Plot
    plot_results(hist, prob, st, lmp_base, sol_star.nu_bal, fmax_star,
                 joinpath(outdir, "ipp_market_planning_case14"))
    write_history_csv(hist, joinpath(outdir, "ipp_history_case14.csv"))

    # ── Budget-Aware Pareto Sweep ─────────────────────────────────────────────
    # Sweep the transmission budget B (constraint Σ Δ ≤ B), holding the
    # objective fixed at min wᵀλ. This is the cleanest "same machinery, no new
    # sensitivities" demo — each FW run reuses identical gradients/VJPs, only
    # the LMO's budget cap changes. Produces a smooth, monotone frontier.
    #
    # Alternative: sweep capex weight α in `min wᵀλ + α·cᵀ(f̄−f̄₀)`. With
    # uniform `c=ones(m)`, α uniformly shifts the gradient — the LMO always
    # picks the same `argmin(g)` branch but stops committing once `g[e_star]+α
    # ≥ 0`. Combined with active set kinks, this gives a piecewise-constant
    # Pareto with only a couple of distinct (Σ Δ, obj) "best iterates" — not
    # presentation-friendly. The capex_α machinery is still available in
    # `fw_ipp!` for callers that want it.
    println("\nBudget aware Pareto: sweep transmission budget B")
    B_frac_grid = [0.015, 0.025, 0.04, 0.07, 0.15]
    spread_grid = Float64[]
    Δnorm_grid  = Float64[]
    B_grid      = Float64[]
    for B_frac in B_frac_grid
        update_fmax!(prob, st.fmax_0); invalidate!(prob.cache)
        B_α = B_frac * sum(st.fmax_0)
        st_B = IPPState(st; B=B_α)
        fmax_B, _ = fw_ipp!(prob, st_B; max_iters=80, tol=1e-7, verbose=false)
        sol_B = with_logger(SimpleLogger(stderr, Logging.Warn)) do
            PowerDiff.solve!(prob)
        end
        push!(spread_grid, dot(st.w, sol_B.nu_bal))
        push!(Δnorm_grid, sum(fmax_B .- st.fmax_0))
        push!(B_grid, B_α)
        @printf("  B_frac = %4.2f (B = %5.2f):  Σ Δ = %.3f,  w_i'λ = %+.4f,  IPP premium = %+.4f\n",
                B_frac, B_α, Δnorm_grid[end], spread_grid[end], -spread_grid[end])
    end

    plot_pareto(B_grid, spread_grid, Δnorm_grid,
                joinpath(outdir, "ipp_market_planning_case14_pareto"))

    # restore prob to fmax_star for downstream
    update_fmax!(prob, fmax_star)
    println()
    return prob, st, fmax_star, hist
end

# =============================================================================
# Tier 4: RTS-GMLC
# =============================================================================

const RTS_PATH = expanduser("~/Datasets/RTS-GMLC/RTS_Data/FormattedData/MATPOWER/RTS_GMLC.m")
const RTS_LOAD_CSV = expanduser("~/Datasets/RTS-GMLC/RTS_Data/timeseries_data_files/Load/DAY_AHEAD_regional_Load.csv")

function load_rts_gmlc()
    isfile(RTS_PATH) || error("RTS_GMLC.m not at $RTS_PATH")
    raw = PM.parse_file(RTS_PATH)
    if !isempty(raw["dcline"])
        empty!(raw["dcline"])           # PowerModels DC line workaround
    end
    PM.make_basic_network!(raw)         # populates rate_a defaults & sequential IDs
    return raw
end

"""
Read DAY_AHEAD_regional_Load.csv and compute, for each hour of day h ∈ 1:24,
the mean total system load over the year. Return a 24-vector of multipliers
(scaled so the annual mean = 1.0).
"""
function rts_hourly_multipliers()
    isfile(RTS_LOAD_CSV) || error("RTS load CSV not at $RTS_LOAD_CSV")
    data, _ = readdlm(RTS_LOAD_CSV, ','; header=true)
    period = Int.(data[:, 4])      # 1..24
    z1 = data[:, 5]; z2 = data[:, 6]; z3 = data[:, 7]
    total = z1 .+ z2 .+ z3
    mults = zeros(24)
    for h in 1:24
        mask = period .== h
        mults[h] = mean(total[mask])
    end
    annual_mean = mean(mults)
    return mults ./ annual_mean
end

function run_tier4(; outdir::String=@__DIR__,
                     ipp_node::Union{Nothing,Int}=nothing,
                     hub_nodes::Union{Nothing,Vector{Int}}=nothing,
                     k_hub::Int=3,
                     run_multi_period::Bool=true)
    println("\n" * "="^65)
    println("Tier 4: RTS-GMLC (73 buses, 120 branches)")
    println("="^65)

    raw = load_rts_gmlc()
    # Tighten flow limits so congestion is meaningful (RTS-GMLC ships generous limits)
    for (_, br) in raw["branch"]
        br["rate_a"] *= 0.5
    end
    net = DCNetwork(raw)
    # Break gen degeneracy so the KKT system is non singular at the optimum.
    for i in eachindex(net.gmax)
        if net.gmax[i] > 0.01
            net.gmax[i] *= 1.5
            net.gmin[i] = max(net.gmin[i], 0.01)
        end
    end
    d   = calc_demand_vector(net)
    n, m = net.n, net.m
    println("Network: $n buses, $m branches, $(net.k) gens.  Total demand = $(round(sum(d), digits=3))")

    prob = DCOPFProblem(net, d)

    # Multi-hub portfolio: one hub per zone, ω = 1/k_hub each.
    ipp, hubs, hub_w = if ipp_node === nothing && hub_nodes === nothing
        pick_default_ipp_hub(prob; k_hub=k_hub, raw=raw, by_zone=true)
    elseif ipp_node !== nothing && hub_nodes !== nothing
        ipp_node, hub_nodes, fill(1.0/length(hub_nodes), length(hub_nodes))
    else
        error("Pass both ipp_node and hub_nodes, or neither")
    end
    st = build_ipp_state(prob; ipp_node=ipp, hub_nodes=hubs, hub_weights=hub_w,
                         B_frac=0.3, Δmax_factor=2.0)

    sol_base = with_logger(SimpleLogger(stderr, Logging.Warn)) do
        PowerDiff.solve!(prob)
    end
    lmp_base = copy(sol_base.nu_bal)
    println("\nIPP setup (multi-hub portfolio k=$(length(st.H))):")
    println("  IPP node i  = $(st.ipp_node)   (lowest-LMP gen bus, λ_i = $(round(lmp_base[st.ipp_seq], digits=4)))")
    println("  Hub set H_i = $(st.H)")
    println("  Hub weights ω = $(round.(st.ω, digits=3))   (Σω = $(round(sum(st.ω), digits=3)))")
    println("  λ at hubs : ", round.(lmp_base[st.H_seq], digits=4))
    println("  Baseline basis  w_i'λ = $(round(dot(st.w, lmp_base), digits=4))   (= Σ ω_j λ_j - λ_i)")
    println("  Budget B  = $(round(st.B, digits=3))")

    invalidate!(prob.cache)

    # ── Single period (peak hour proxy = baseline d) ──────────────────────────
    println("\nSingle period FW:")
    t0 = time()
    fmax_star, hist = fw_ipp!(prob, st; max_iters=60, tol=1e-7)
    elapsed_single = time() - t0
    println("\nSingle period wall time: ", round(elapsed_single, digits=1), " s, ",
            length(hist.obj), " iters")

    sol_star = with_logger(SimpleLogger(stderr, Logging.Warn)) do
        PowerDiff.solve!(prob)
    end
    Δ = fmax_star .- st.fmax_0
    perm = sortperm(Δ; rev=true)
    println("\nTop 8 upgraded branches:")
    @printf("  Branch ID  fmax_0   fmax*    Δ        %% increase\n")
    cnt = 0
    for e in perm
        Δ[e] < 1e-4 && break
        cnt += 1; cnt > 8 && break
        @printf("  %-9d  %-7.3f  %-7.3f  %-7.3f  %+5.0f%%\n",
                net.id_map.branch_ids[e], st.fmax_0[e], fmax_star[e], Δ[e],
                100*Δ[e]/st.fmax_0[e])
    end
    @printf("\nTotal Δ used = %.3f  (budget %.3f)\n", sum(Δ), st.B)
    @printf("Objective w_i'λ (basis):  baseline = %.4f, optimized = %.4f\n",
            dot(st.w, lmp_base), dot(st.w, sol_star.nu_bal))
    @printf("IPP premium improvement = %+.4f\n",
            dot(st.w, lmp_base) - dot(st.w, sol_star.nu_bal))

    plot_results(hist, prob, st, lmp_base, sol_star.nu_bal, fmax_star,
                 joinpath(outdir, "ipp_market_planning_rts_gmlc"))
    write_history_csv(hist, joinpath(outdir, "ipp_history_rts_gmlc.csv"))

    if !run_multi_period
        return prob, st, fmax_star, hist
    end

    # ── Multi period (12 hour of day means from RTS-GMLC time series) ────────
    println("\nMulti period FW (12 representative hours from RTS-GMLC time series):")
    update_fmax!(prob, st.fmax_0); invalidate!(prob.cache)

    mults24 = rts_hourly_multipliers()
    # Subsample 12 (every other hour starting at 1, so 1,3,5,...,23)
    sample_hours = collect(1:2:24)
    demand_periods = Vector{Vector{Float64}}()
    for h in sample_hours
        push!(demand_periods, mults24[h] .* d)
    end
    println("  Hourly multipliers (12 periods): ", round.(mults24[sample_hours], digits=3))

    t0 = time()
    fmax_star_mp, hist_mp = fw_ipp!(prob, st; max_iters=40, tol=1e-7,
                                     demand_periods=demand_periods)
    elapsed_mp = time() - t0
    println("\nMulti period wall time: ", round(elapsed_mp, digits=1), " s")

    Δ_mp = fmax_star_mp .- st.fmax_0
    perm_mp = sortperm(Δ_mp; rev=true)
    println("\nTop 8 upgraded branches (multi period):")
    @printf("  Branch ID  fmax_0   fmax*    Δ        %% increase\n")
    cnt = 0
    for e in perm_mp
        Δ_mp[e] < 1e-4 && break
        cnt += 1; cnt > 8 && break
        @printf("  %-9d  %-7.3f  %-7.3f  %-7.3f  %+5.0f%%\n",
                net.id_map.branch_ids[e], st.fmax_0[e], fmax_star_mp[e], Δ_mp[e],
                100*Δ_mp[e]/st.fmax_0[e])
    end

    # restore at peak hour for plotting
    update_fmax!(prob, fmax_star_mp); invalidate!(prob.cache)
    update_demand!(prob, d)
    sol_mp_final = with_logger(SimpleLogger(stderr, Logging.Warn)) do
        PowerDiff.solve!(prob)
    end

    plot_results(hist_mp, prob, st, lmp_base, sol_mp_final.nu_bal, fmax_star_mp,
                 joinpath(outdir, "ipp_market_planning_rts_gmlc_multi"))
    write_history_csv(hist_mp, joinpath(outdir, "ipp_history_rts_gmlc_multi.csv"))

    println()
    return prob, st, (single=fmax_star, multi=fmax_star_mp), (single=hist, multi=hist_mp)
end

# =============================================================================
# Plotting
# =============================================================================

function plot_3bus(history, lmp_base, lmp_star, savepath::String;
                   ipp_seq::Int=1, hub_seqs::Vector{Int}=[3])
    set_theme!(theme_minimal())
    fig = Figure(size=(1050, 420))
    iters = 0:length(history.obj)-1

    ax_a = Axis(fig[1, 1]; xlabel="FW iteration  k",
                ylabel=L"w_i^{\!\top}\!\lambda \;=\; \sum_{j\in H_i}\!\omega_j\,\lambda_j - \lambda_i",
                title=L"\text{(a)}\;\;\text{FW objective (basis)}")
    lines!(ax_a, iters, history.obj; color=:steelblue, linewidth=2)
    scatter!(ax_a, iters, history.obj; color=:steelblue, markersize=8)

    ax_b = Axis(fig[1, 2]; xlabel="FW iteration  k",
                ylabel=L"g(u_k)^{\!\top}\!(u_k - v_k)",
                title=L"\text{(b)}\;\;\text{FW duality gap}", yscale=log10)
    pos_gaps = max.(abs.(history.gap), 1e-12)
    lines!(ax_b, iters, pos_gaps; color=:firebrick, linewidth=2)
    scatter!(ax_b, iters, pos_gaps; color=:firebrick, markersize=8)

    ax_c = Axis(fig[1, 3]; xlabel="Bus index  j",
                ylabel=L"\lambda_j\;\;[\$/\mathrm{MWh}]",
                title=L"\text{(c)}\;\;\text{LMP: baseline vs.\ optimized}",
                xticks=1:3)
    barpos1 = [1, 2, 3] .- 0.18
    barpos2 = [1, 2, 3] .+ 0.18
    bp1 = barplot!(ax_c, barpos1, lmp_base; width=0.32, color=:gray70)
    bp2 = barplot!(ax_c, barpos2, lmp_star; width=0.32, color=:steelblue)
    vl_ipp = vlines!(ax_c, [Float64(ipp_seq)]; color=(:darkorange, 0.85),
                     linestyle=:dot, linewidth=2.5)
    vl_hub = vlines!(ax_c, Float64.(hub_seqs); color=:firebrick,
                     linestyle=:dash, linewidth=1.8)

    # Single horizontal legend below all panels (no overlap with bars).
    Legend(fig[2, 1:3],
           [bp1, bp2, vl_ipp, vl_hub],
           [L"\text{Baseline}\;\;\lambda_j(\overline{f}_0)",
            L"\text{Optimized}\;\;\lambda_j(\overline{f}_0 + u_\star)",
            L"\text{IPP node}\;\;i\;\;\;(w_i = -1)",
            L"\text{Hub}\;\;h \in H_i\;\;\;(w_h = \omega_h)"],
           orientation=:horizontal, framevisible=false, nbanks=1,
           colgap=18, labelsize=12)

    Label(fig[0, :], L"\text{3-bus}: \;\; \text{IPP}\,@\,\text{bus}\,%$(ipp_seq), \;\; H_i = %$(hub_seqs), \;\; \omega = [1.0]";
          fontsize=15, font=:bold)

    rowsize!(fig.layout, 2, Relative(0.10))

    save(savepath * ".pdf", fig)
    save(savepath * ".png", fig; px_per_unit=2)
    println("  Figure saved: $(savepath).{pdf,png}")
end

function plot_results(history, prob, st::IPPState, lmp_base, lmp_star, fmax_star,
                       savepath::String)
    set_theme!(theme_minimal())
    n, m = prob.network.n, prob.network.m
    fig = Figure(size=(1300, 980))
    iters = 0:length(history.obj)-1

    # ── (a) FW objective (basis) ──────────────────────────────────────────────
    ax_a = Axis(fig[1, 1]; xlabel="FW iteration  k",
                ylabel=L"w_i^{\!\top}\!\lambda \;=\; \sum_{j\in H_i}\!\omega_j\,\lambda_j - \lambda_i",
                title=L"\text{(a)}\;\;\text{FW objective (IPP basis)}")
    lines!(ax_a, iters, history.obj; color=:steelblue, linewidth=2)
    scatter!(ax_a, iters, history.obj; color=:steelblue, markersize=5)

    # ── (b) FW duality gap (log) ──────────────────────────────────────────────
    pos_gaps = max.(abs.(history.gap), 1e-12)
    ax_b = Axis(fig[1, 2]; xlabel="FW iteration  k",
                ylabel=L"g(u_k)^{\!\top}\!(u_k - v_k)",
                title=L"\text{(b)}\;\;\text{FW duality gap (active set transitions cause spikes)}",
                yscale=log10)
    lines!(ax_b, iters, pos_gaps; color=:firebrick, linewidth=2)
    scatter!(ax_b, iters, pos_gaps; color=:firebrick, markersize=5)

    # ── (c) Heatmap of u_e / f̄_{0,e} over iterations ─────────────────────────
    # u_e = f̄_e - f̄_{0,e} is the per-branch capacity upgrade. Normalised by the
    # baseline so different branches compare on a common 0–2 scale.
    ax_c = Axis(fig[2, 1]; xlabel="FW iteration  k",
                ylabel="Branch sequential index  e",
                title=L"\text{(c)}\;\;\text{Upgrade fraction}\;\;u_e/\overline{f}_{0,e}")
    u_hist = (history.fmax_hist .- st.fmax_0) ./ st.fmax_0
    hm = heatmap!(ax_c, iters, 1:m, u_hist'; colormap=:viridis)

    # ── (d) LMP comparison ────────────────────────────────────────────────────
    xtick_vals = n ≤ 20 ? (1:n) : (5:5:n)
    ax_d = Axis(fig[2, 2]; xlabel="Bus sequential index  j",
                ylabel=L"\lambda_j\;\;[\$/\mathrm{MWh}]",
                title=L"\text{(d)}\;\;\text{LMP: baseline vs.\ optimized}",
                xticks=xtick_vals)
    barwidth = n ≤ 20 ? 0.35 : 0.45
    barpos1 = collect(1:n) .- 0.2
    barpos2 = collect(1:n) .+ 0.2
    bp1 = barplot!(ax_d, barpos1, lmp_base; width=barwidth, color=:gray70)
    bp2 = barplot!(ax_d, barpos2, lmp_star; width=barwidth, color=:steelblue)
    vl_ipp = vlines!(ax_d, [Float64(st.ipp_seq)]; color=(:darkorange, 0.85),
                     linestyle=:dot, linewidth=2.5)
    vl_hub = vlines!(ax_d, Float64.(st.H_seq); color=:firebrick,
                     linestyle=:dash, linewidth=1.8)

    # ── Row 3: horizontal colorbar (under panel c) + legend (under panel d) ──
    # Both horizontal so the heatmap colorbar's label cannot collide with panel
    # (d)'s y-axis (which previously caused the 'LMP' / 'Δ/fmax_0' overlap).
    Colorbar(fig[3, 1], hm; vertical=false,
             label=L"u_e/\overline{f}_{0,e}\;\;\;(\text{fractional upgrade})",
             height=14, width=Relative(0.85))

    Legend(fig[3, 2],
           [bp1, bp2, vl_ipp, vl_hub],
           [L"\text{Baseline}\;\;\lambda_j(\overline{f}_0)",
            L"\text{Optimized}\;\;\lambda_j(\overline{f}_0 + u_\star)",
            L"\text{IPP node}\;\;i\;\;\;(w_i = -1)",
            L"\text{Hub}\;\;h \in H_i\;\;\;(w_h = \omega_h)"],
           orientation=:horizontal, framevisible=false, nbanks=2,
           colgap=18, labelsize=12)

    # ── Title ────────────────────────────────────────────────────────────────
    Σu = round(sum(fmax_star .- st.fmax_0), digits=2)
    ω_str = length(st.ω) == 1 ? "1.0" : string(round.(st.ω, digits=3))
    Label(fig[0, :],
          L"\text{IPP}\,@\,\text{bus}\,%$(st.ipp_node) \;\;|\;\; H_i = %$(st.H) \;\;|\;\; \omega = %$(ω_str) \;\;|\;\; B = %$(round(st.B, digits=2)) \;\;|\;\; \sum_e u_e = %$(Σu)";
          fontsize=14, font=:bold)

    rowsize!(fig.layout, 3, Relative(0.10))
    colgap!(fig.layout, 30)
    rowgap!(fig.layout, 18)

    save(savepath * ".pdf", fig)
    save(savepath * ".png", fig; px_per_unit=2)
    println("  Figure saved: $(savepath).{pdf,png}")
end

function plot_pareto(B_grid, spread_grid, Δnorm_grid, savepath::String)
    set_theme!(theme_minimal())
    fig = Figure(size=(960, 380))

    # Profit gain relative to the smallest-budget run.
    # Premium gain Π_i(B) − Π_i(B_min), where Π_i = -w_i'λ = λ_i − Σ ω_j λ_j.
    # Reference is the smallest-budget run, so the leftmost point is exactly 0.
    premium_gain = -spread_grid .+ spread_grid[1]

    ax_a = Axis(fig[1, 1];
                xlabel=L"\text{Total upgrade}~\sum_e u_e~[\mathrm{MW}]",
                ylabel=L"\Delta \Pi_i(B) = \Pi_i(B) - \Pi_i(B_{\min}),~~\Pi_i = \lambda_i - \sum_{j \in H_i}\omega_j \lambda_j",
                title=L"\text{(a) IPP per-MWh premium gain vs. upgrade}",
                ytickformat = vals -> [@sprintf("%.1f", v) for v in vals])
    lines!(ax_a, Δnorm_grid, premium_gain; color=:gray60, linewidth=1.5)
    sc_a = scatter!(ax_a, Δnorm_grid, premium_gain; color=B_grid,
                    colormap=:viridis, markersize=14,
                    strokecolor=:black, strokewidth=0.5)

    ax_b = Axis(fig[1, 2]; xlabel=L"\text{Budget}~B~[\mathrm{MW}]",
                ylabel=L"\sum_e u_e~~(\text{used})",
                title=L"\text{(b) Budget saturation}")
    lines!(ax_b, B_grid, Δnorm_grid; color=:gray60, linewidth=1.5)
    scatter!(ax_b, B_grid, Δnorm_grid; color=B_grid,
             colormap=:viridis, markersize=14,
             strokecolor=:black, strokewidth=0.5)

    Colorbar(fig[1, 3], sc_a; label=L"\text{Budget}~B", width=12)
    Label(fig[0, :], L"\text{case14: budget aware Pareto frontier (premium}~\Pi_i = \lambda_i - \sum_{j \in H_i}\omega_j \lambda_j\text{)}";
          fontsize=14, font=:bold)

    save(savepath * ".pdf", fig)
    save(savepath * ".png", fig; px_per_unit=2)
    println("  Figure saved: $(savepath).{pdf,png}")
end

# =============================================================================
# main
# =============================================================================

function main(; tiers=[1, 2, 4])
    Random.seed!(42)
    1 in tiers && run_tier1()
    2 in tiers && run_tier2()
    4 in tiers && run_tier4()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
