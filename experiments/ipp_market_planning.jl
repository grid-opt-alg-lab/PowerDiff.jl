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
# IPP Market-Aware Transmission Upgrade Planning via Frank-Wolfe
# =============================================================================
#
# Bilevel program:
#
#     min_{fmax ∈ U}  w' λ*(fmax)        (outer)
#     s.t.            λ*(fmax) ∈ OPF(fmax)  (inner: full DC OPF)
#                     w_i = +1 if i ∈ H, w_i = -1 otherwise
#
# An IPP holds a PPA settling at a hub bus set H ⊆ N, paid local LMPs at
# non-hub generation. Minimising w' λ ≡ maximising the IPP's basis revenue.
#
# Algorithm: Frank-Wolfe over the budget simplex
#   U = { fmax = fmax_0 + Δ : Δ_e ≥ 0,  Σ Δ_e ≤ B,  Δ_e ≤ Δmax_e }
#
# Per FW iteration: update_fmax! → solve! → vjp!(:lmp, :fmax, w) → LMO → step.
# Matrix-free gradient: O(nnz) per iteration via PowerDiff's transpose KKT solve.

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

const PM = PowerModels
PM.silence()

# =============================================================================
# Hub policy (configurable at top of script)
# =============================================================================
#
# Set HUB_OVERRIDE to a Vector{Int} of original bus IDs to pin the hub. Default
# `nothing` triggers auto-detection: at fmax_0 (after tightening), the bus with
# the highest LMP is chosen as the hub. This is the load-pocket that the IPP
# pays through its PPA, so upgrades that relieve its congestion increase IPP
# basis revenue.
#
# Note for Pablo's slide: choosing H = {slack} (e.g., bus 1 in case14) is
# pedagogically clean (cong_slack ≡ 0 by PTDF convention) but produces a
# DEGENERATE demo — the optimizer says "do nothing" because the IPP already
# benefits from the status quo. We surface both runs in the deck.

const HUB_OVERRIDE = nothing   # nothing = auto-detect; or Vector{Int} of orig IDs

# =============================================================================
# IPP State
# =============================================================================

struct IPPState
    fmax_0::Vector{Float64}
    Δmax::Vector{Float64}
    B::Float64
    H::Vector{Int}            # hub bus *original* IDs (translated to seq for w)
    w::Vector{Float64}         # length n, ±1 in sequential-index order
end

function build_ipp_state(prob::DCOPFProblem;
                          hub_override=nothing,
                          B_frac::Float64=0.5,
                          Δmax_factor::Float64=2.0)
    net = prob.network
    n, m = net.n, net.m
    fmax_0 = copy(net.fmax)
    Δmax   = Δmax_factor .* fmax_0
    B      = B_frac * sum(fmax_0)

    # Solve once at baseline to detect hub
    sol = with_logger(SimpleLogger(stderr, Logging.Error)) do
        PowerDiff.solve!(prob)
    end

    H_orig = if hub_override !== nothing
        collect(hub_override)
    else
        seq = argmax(sol.nu_bal)
        [net.id_map.bus_ids[seq]]
    end

    w = fill(-1.0, n)
    for h in H_orig
        haskey(net.id_map.bus_to_idx, h) || error("Hub bus $h not in network")
        w[net.id_map.bus_to_idx[h]] = 1.0
    end

    return IPPState(fmax_0, Δmax, B, H_orig, w)
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
  vector. If `nothing`, runs a single-period FW with the prob's current `d`.
"""
# Re-solve the OPF at fmax_try (mutates prob) and return scalar objective.
# Used by the Armijo line search inside fw_ipp!.
function _eval_obj!(prob, st, fmax_try, capex_α, capex_c, demand_periods)
    update_fmax!(prob, fmax_try)
    obj = 0.0
    if demand_periods === nothing
        sol = with_logger(SimpleLogger(stderr, Logging.Error)) do
            PowerDiff.solve!(prob)
        end
        obj = dot(st.w, sol.nu_bal)
    else
        T = length(demand_periods)
        for t in 1:T
            update_demand!(prob, demand_periods[t])
            sol = with_logger(SimpleLogger(stderr, Logging.Error)) do
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

    # Best iterate tracking (handles non-convex w'λ from active-set jumps)
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
            sol = with_logger(SimpleLogger(stderr, Logging.Error)) do
                PowerDiff.solve!(prob)
            end
            obj = dot(st.w, sol.nu_bal)
            with_logger(SimpleLogger(stderr, Logging.Error)) do
                vjp!(g, prob, :lmp, :fmax, st.w; work=work)
            end
        else
            T = length(demand_periods)
            for t in 1:T
                update_demand!(prob, demand_periods[t])
                sol = with_logger(SimpleLogger(stderr, Logging.Error)) do
                    PowerDiff.solve!(prob)
                end
                obj += dot(st.w, sol.nu_bal) / T
                fill!(g_t, 0.0)
                with_logger(SimpleLogger(stderr, Logging.Error)) do
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

        # Step
        γ = if step_rule == :armijo
            # Backtrack from γ_full = 2/(k+2). Halve until obj decreases (or
            # we exhaust 8 backtracks). This handles active-set kinks where
            # the linearization stops being valid mid-step.
            γ_try = 2.0 / (k + 2)
            best_γ = γ_try
            best_δ = +Inf
            for _ in 1:8
                @. fmax_try = (1 - γ_try) * fmax_k + γ_try * v
                obj_try = _eval_obj!(prob, st, fmax_try, capex_α, capex_c, demand_periods)
                if obj_try < obj + 0.0
                    best_γ = γ_try; best_δ = obj_try - obj
                    break
                else
                    if obj_try - obj < best_δ
                        best_γ = γ_try; best_δ = obj_try - obj
                    end
                    γ_try /= 2.0
                end
            end
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

    # Restore prob to best iterate
    update_fmax!(prob, fmax_best)
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

    sol = with_logger(SimpleLogger(stderr, Logging.Error)) do
        PowerDiff.solve!(prob)
    end
    lmp_base = copy(sol.nu_bal)

    println("\nBaseline LMPs (bus 1, 2, 3): ", round.(lmp_base, digits=4))
    println("  λ_3 - λ_1 = ", round(lmp_base[3] - lmp_base[1], digits=4),
            "   ← congestion rent")

    # ── Hub = bus 3 (load pocket; the IPP pays this LMP via PPA) ──────────────
    H_orig = [3]
    w = [-1.0, -1.0, +1.0]
    println("\nIPP setup:")
    println("  Hub H = $H_orig  (bus 3, the load pocket)")
    println("  w     = $w        (+1 hub, -1 elsewhere)")
    println("  IPP profit = -w'λ = (λ_1 + λ_2) - λ_3")

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
        sol_p = with_logger(SimpleLogger(stderr, Logging.Error)) do
            PowerDiff.solve!(prob)
        end
        fd[:, e] = (sol_p.nu_bal .- lmp_base) ./ ε
    end
    update_fmax!(prob, fmax_baseline)  # restore
    with_logger(SimpleLogger(stderr, Logging.Error)) do
        PowerDiff.solve!(prob)
    end

    println("\nFinite-difference reference:")
    show(stdout, "text/plain", round.(fd, digits=4)); println()
    err = maximum(abs.(Matrix(dlmp_dfmax) .- fd))
    @printf("\nmax |analytical − FD| = %.3e   (tol 1e-3)\n", err)
    err < 1e-3 || @warn "Tier 1 FD verification failed" err

    # Reset cache (so the in-loop VJP takes the matrix-free path, not the cached matrix)
    invalidate!(prob.cache)

    # ── Frank-Wolfe ────────────────────────────────────────────────────────────
    println("\nFrank-Wolfe (B = 1.5, Δmax = [2,2]):")
    st = IPPState(copy(prob.network.fmax), [2.0, 2.0], 1.5, H_orig, w)
    fmax_star, hist = fw_ipp!(prob, st; max_iters=10, tol=1e-8)

    println("\nUpgrade plan:")
    @printf("  Branch  fmax_0   fmax*    Δ\n")
    for e in 1:2
        @printf("  %-7d  %-7.3f  %-7.3f  %.3f\n",
                e, st.fmax_0[e], fmax_star[e], fmax_star[e]-st.fmax_0[e])
    end

    sol_star = with_logger(SimpleLogger(stderr, Logging.Error)) do
        PowerDiff.solve!(prob)
    end
    println("\nLMPs (baseline → optimized):")
    for i in 1:3
        @printf("  bus %d:  %.4f → %.4f   Δ = %+.4f\n",
                i, lmp_base[i], sol_star.nu_bal[i], sol_star.nu_bal[i]-lmp_base[i])
    end
    @printf("\nObjective w'λ:  baseline = %.4f, optimized = %.4f\n",
            dot(w, lmp_base), dot(w, sol_star.nu_bal))
    @printf("IPP profit -w'λ:  baseline = %.4f, optimized = %.4f  (Δ = %+.4f)\n",
            -dot(w, lmp_base), -dot(w, sol_star.nu_bal),
            dot(w, lmp_base) - dot(w, sol_star.nu_bal))

    # ── Plot ───────────────────────────────────────────────────────────────────
    plot_3bus(hist, lmp_base, sol_star.nu_bal,
              joinpath(outdir, "ipp_market_planning_3bus"))

    write_history_csv(hist, joinpath(outdir, "ipp_history_3bus.csv"))
    println()
    return prob, st, fmax_star, hist
end

# =============================================================================
# Tier 2: case14
# =============================================================================

function run_tier2(; outdir::String=@__DIR__,
                     hub_override=HUB_OVERRIDE,
                     fmax_scale::Float64=0.10)
    println("\n" * "="^65)
    println("Tier 2: case14, rate_a × $(fmax_scale)  (single-period + capex Pareto)")
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
    # Break generator degeneracy (per MEMORY.md and experiments/lmp_switching_fw.jl
    # pattern). Without this the KKT system is singular at the optimum (multiple
    # gens at upper bound), Tikhonov regularization kicks in, and the matrix-free
    # VJP returns essentially zero gradient.
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
    st = build_ipp_state(prob; hub_override=hub_override, B_frac=0.5, Δmax_factor=2.0)

    # Re-solve at fmax_0 to get baseline LMPs (build_ipp_state already solved once)
    sol_base = with_logger(SimpleLogger(stderr, Logging.Error)) do
        PowerDiff.solve!(prob)
    end
    lmp_base = copy(sol_base.nu_bal)

    println("\nIPP setup:")
    println("  Hub H = $(st.H)  (auto-detected highest-LMP bus)")
    println("  λ at hub: ", round.(lmp_base[[net.id_map.bus_to_idx[h] for h in st.H]], digits=4))
    println("  Mean λ at non-hub: ", round(mean(lmp_base[[i for i in 1:n if !(net.id_map.bus_ids[i] in st.H)]]), digits=4))
    println("  Budget B = $(round(st.B, digits=3))")

    # Reset cache so in-loop VJP is matrix-free
    invalidate!(prob.cache)

    # ── Pure spread (capex_α = 0) ─────────────────────────────────────────────
    println("\nPure spread: min w'λ")
    fmax_star, hist = fw_ipp!(prob, st; max_iters=80, tol=1e-7)

    sol_star = with_logger(SimpleLogger(stderr, Logging.Error)) do
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
    @printf("Objective w'λ:  baseline = %.4f, optimized = %.4f\n",
            dot(st.w, lmp_base), dot(st.w, sol_star.nu_bal))
    @printf("IPP profit -w'λ improvement: %+.4f\n",
            dot(st.w, lmp_base) - dot(st.w, sol_star.nu_bal))

    # FD verification at fmax* on top-5 branches
    println("\nFD verification at fmax* (top 5 by |Δ|):")
    obj_star = dot(st.w, sol_star.nu_bal)
    g_star = zeros(m); work = zeros(kkt_dims(prob))
    invalidate!(prob.cache)
    PowerDiff.solve!(prob)
    with_logger(SimpleLogger(stderr, Logging.Error)) do
        vjp!(g_star, prob, :lmp, :fmax, st.w; work=work)
    end
    ε = 1e-5
    @printf("  Branch    VJP           FD            |Δ|\n")
    for e in perm[1:min(5, m)]
        Δ[e] < 1e-6 && break
        fmax_p = copy(fmax_star); fmax_p[e] += ε
        update_fmax!(prob, fmax_p)
        sol_p = with_logger(SimpleLogger(stderr, Logging.Error)) do
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

    # ── Capex-Aware Pareto Sweep ──────────────────────────────────────────────
    println("\nCapex-aware Pareto: min w'λ + α·c'(fmax-fmax_0)")
    capex_c = ones(m)
    α_grid = [0.0, 0.5, 1.0, 2.0, 5.0]
    spread_grid  = Float64[]
    Δnorm_grid   = Float64[]
    for α in α_grid
        update_fmax!(prob, st.fmax_0); invalidate!(prob.cache)
        st_α = IPPState(st.fmax_0, st.Δmax, st.B, st.H, st.w)
        fmax_α, _ = fw_ipp!(prob, st_α; max_iters=60, tol=1e-7,
                            capex_α=α, capex_c=capex_c, verbose=false)
        sol_α = with_logger(SimpleLogger(stderr, Logging.Error)) do
            PowerDiff.solve!(prob)
        end
        push!(spread_grid, dot(st.w, sol_α.nu_bal))
        push!(Δnorm_grid, sum(fmax_α .- st.fmax_0))
        @printf("  α = %4.1f:  Σ Δ = %.3f,  w'λ = %+.4f,  IPP profit = %+.4f\n",
                α, Δnorm_grid[end], spread_grid[end], -spread_grid[end])
    end

    plot_pareto(α_grid, spread_grid, Δnorm_grid,
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
Read DAY_AHEAD_regional_Load.csv and compute, for each hour-of-day h ∈ 1:24,
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
                     hub_override=HUB_OVERRIDE,
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
    # Break gen degeneracy (per MEMORY.md / lmp_switching_fw.jl)
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
    st = build_ipp_state(prob; hub_override=hub_override, B_frac=0.3, Δmax_factor=2.0)

    sol_base = with_logger(SimpleLogger(stderr, Logging.Error)) do
        PowerDiff.solve!(prob)
    end
    lmp_base = copy(sol_base.nu_bal)
    println("\nIPP setup:")
    println("  Hub H = $(st.H)  (auto-detected highest-LMP bus)")
    println("  λ at hub: ", round.(lmp_base[[net.id_map.bus_to_idx[h] for h in st.H]], digits=4))
    println("  Mean λ all buses: ", round(mean(lmp_base), digits=4))
    println("  Budget B = $(round(st.B, digits=3))")

    invalidate!(prob.cache)

    # ── Single-period (peak hour proxy = baseline d) ──────────────────────────
    println("\nSingle-period FW:")
    t0 = time()
    fmax_star, hist = fw_ipp!(prob, st; max_iters=60, tol=1e-7)
    elapsed_single = time() - t0
    println("\nSingle-period wall time: ", round(elapsed_single, digits=1), " s, ",
            length(hist.obj), " iters")

    sol_star = with_logger(SimpleLogger(stderr, Logging.Error)) do
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
    @printf("Objective w'λ:  baseline = %.4f, optimized = %.4f\n",
            dot(st.w, lmp_base), dot(st.w, sol_star.nu_bal))
    @printf("IPP profit improvement = %+.4f\n",
            dot(st.w, lmp_base) - dot(st.w, sol_star.nu_bal))

    plot_results(hist, prob, st, lmp_base, sol_star.nu_bal, fmax_star,
                 joinpath(outdir, "ipp_market_planning_rts_gmlc"))
    write_history_csv(hist, joinpath(outdir, "ipp_history_rts_gmlc.csv"))

    if !run_multi_period
        return prob, st, fmax_star, hist
    end

    # ── Multi-period (12 hour-of-day means from RTS-GMLC time series) ────────
    println("\nMulti-period FW (12 representative hours from RTS-GMLC time series):")
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
    println("\nMulti-period wall time: ", round(elapsed_mp, digits=1), " s")

    Δ_mp = fmax_star_mp .- st.fmax_0
    perm_mp = sortperm(Δ_mp; rev=true)
    println("\nTop 8 upgraded branches (multi-period):")
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
    sol_mp_final = with_logger(SimpleLogger(stderr, Logging.Error)) do
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

function plot_3bus(history, lmp_base, lmp_star, savepath::String)
    set_theme!(theme_minimal())
    fig = Figure(size=(900, 360))
    iters = 0:length(history.obj)-1

    ax_a = Axis(fig[1, 1]; xlabel="Iteration", ylabel="Objective  w'λ",
                title="(a) FW convergence")
    lines!(ax_a, iters, history.obj; color=:steelblue, linewidth=2)
    scatter!(ax_a, iters, history.obj; color=:steelblue, markersize=8)

    ax_b = Axis(fig[1, 2]; xlabel="Iteration", ylabel="FW gap",
                title="(b) Duality gap", yscale=log10)
    pos_gaps = max.(abs.(history.gap), 1e-12)
    lines!(ax_b, iters, pos_gaps; color=:firebrick, linewidth=2)
    scatter!(ax_b, iters, pos_gaps; color=:firebrick, markersize=8)

    ax_c = Axis(fig[1, 3]; xlabel="Bus", ylabel="LMP",
                title="(c) LMP: baseline vs optimized",
                xticks=1:3)
    barpos1 = [1, 2, 3] .- 0.18
    barpos2 = [1, 2, 3] .+ 0.18
    barplot!(ax_c, barpos1, lmp_base; width=0.32, color=:gray70, label="Baseline")
    barplot!(ax_c, barpos2, lmp_star; width=0.32, color=:steelblue, label="Optimized")
    axislegend(ax_c; position=:lt)

    Label(fig[0, :], "Tier 1: 3-bus pedagogical (Hub = bus 3)";
          fontsize=14, font=:bold)

    save(savepath * ".pdf", fig)
    save(savepath * ".png", fig; px_per_unit=2)
    println("  Figure saved: $(savepath).{pdf,png}")
end

function plot_results(history, prob, st::IPPState, lmp_base, lmp_star, fmax_star,
                       savepath::String)
    set_theme!(theme_minimal())
    n, m = prob.network.n, prob.network.m
    fig = Figure(size=(1100, 800))
    iters = 0:length(history.obj)-1

    # (a) Objective
    ax_a = Axis(fig[1, 1]; xlabel="Iteration", ylabel="Objective  w'λ",
                title="(a) FW objective")
    lines!(ax_a, iters, history.obj; color=:steelblue, linewidth=2)
    scatter!(ax_a, iters, history.obj; color=:steelblue, markersize=5)

    # (b) FW gap (log)
    pos_gaps = max.(abs.(history.gap), 1e-12)
    ax_b = Axis(fig[1, 2]; xlabel="Iteration", ylabel="FW gap",
                title="(b) Duality gap", yscale=log10)
    lines!(ax_b, iters, pos_gaps; color=:firebrick, linewidth=2)
    scatter!(ax_b, iters, pos_gaps; color=:firebrick, markersize=5)

    # (c) Heatmap of Δfmax / fmax_0 over iterations
    ax_c = Axis(fig[2, 1]; xlabel="Iteration", ylabel="Branch (sequential idx)",
                title="(c) Upgrade evolution Δ/fmax_0")
    Δhist = (history.fmax_hist .- st.fmax_0) ./ st.fmax_0
    hm = heatmap!(ax_c, iters, 1:m, Δhist'; colormap=:viridis)
    Colorbar(fig[2, 1, Right()], hm; label="Δ/fmax₀", width=12)

    # (d) LMP comparison
    xtick_vals = n ≤ 20 ? (1:n) : (5:5:n)
    ax_d = Axis(fig[2, 2]; xlabel="Bus (sequential idx)", ylabel="LMP",
                title="(d) LMP: baseline vs optimized",
                xticks=xtick_vals)
    barwidth = n ≤ 20 ? 0.35 : 0.45
    barpos1 = collect(1:n) .- 0.2
    barpos2 = collect(1:n) .+ 0.2
    barplot!(ax_d, barpos1, lmp_base; width=barwidth, color=:gray70, label="Baseline")
    barplot!(ax_d, barpos2, lmp_star; width=barwidth, color=:steelblue, label="Optimized")
    # Annotate hub bus
    for h in st.H
        seq = prob.network.id_map.bus_to_idx[h]
        vlines!(ax_d, [seq]; color=:firebrick, linestyle=:dash, linewidth=1.5)
    end
    axislegend(ax_d; position=:rt)

    Label(fig[0, :], "Hub: $(st.H)   |   B = $(round(st.B, digits=2))   |   Δ used = $(round(sum(fmax_star .- st.fmax_0), digits=2))";
          fontsize=13, font=:bold)

    save(savepath * ".pdf", fig)
    save(savepath * ".png", fig; px_per_unit=2)
    println("  Figure saved: $(savepath).{pdf,png}")
end

function plot_pareto(α_grid, spread_grid, Δnorm_grid, savepath::String)
    set_theme!(theme_minimal())
    fig = Figure(size=(900, 380))

    ax_a = Axis(fig[1, 1]; xlabel="Total Σ Δ (transmission upgrade)",
                ylabel="IPP profit  -w'λ",
                title="(a) Profit-vs-capex frontier")
    lines!(ax_a, Δnorm_grid, -spread_grid; color=:steelblue, linewidth=2)
    scatter!(ax_a, Δnorm_grid, -spread_grid; color=:steelblue, markersize=10)
    for (i, α) in enumerate(α_grid)
        text!(ax_a, "α=$(α)", position=(Δnorm_grid[i], -spread_grid[i]),
              fontsize=10, offset=(8, -8))
    end

    ax_b = Axis(fig[1, 2]; xlabel="Capex weight α", ylabel="Σ Δ used",
                title="(b) Upgrade reduction with capex penalty")
    lines!(ax_b, α_grid, Δnorm_grid; color=:darkorange, linewidth=2)
    scatter!(ax_b, α_grid, Δnorm_grid; color=:darkorange, markersize=10)

    Label(fig[0, :], "case14: capex-aware Pareto frontier"; fontsize=14, font=:bold)

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
