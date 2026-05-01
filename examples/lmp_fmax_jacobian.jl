# @author: Samuel Talkington 04/26/2026
# Documentation: https://samueltalkington.com/research/powerdiff/
# This file runs standalone with `julia lmp_fmax_jacobian.jl`.

# Jacobian of LMPs w.r.t. transmission line capacities (fmax)
# for a 3-bus DC OPF, with a finite-difference cross-check.
#
#         line 1 (fmax = 0.5, binding)
#   bus 1 ─────────────────────── bus 3   ← 1.0 pu load
#                                /
#         line 2 (fmax = 10.0)  /
#   bus 2 ───────────────────────
#
# Cheap gen at bus 1 (cl = 10), expensive gen at bus 2 (cl = 50). Line 1
# saturates, so cheap power is rationed and bus 3 prices off the expensive
# gen. Increasing fmax[1] lets more cheap power reach bus 3, dropping its
# LMP — i.e. ∂LMP[3]/∂fmax[1] is meaningfully negative.

# Install PowerDiff into a temporary environment
import Pkg
Pkg.activate(; temp=true)
Pkg.add(url="https://github.com/grid-opt-alg-lab/PowerDiff.jl")

using PowerDiff

# ---------------------------------------------------------------------------
# 1. Build the network programmatically (no MATPOWER file needed).
# ---------------------------------------------------------------------------
n, m, k = 3, 2, 2            # number of buses, lines, and generators.

# Line incidence matrix
A = [1.0  0.0 -1.0;          # line 1: bus 1 -> bus 3
     0.0  1.0 -1.0]          # line 2: bus 2 -> bus 3

# Generator incidence matrix
G_inc = [1.0 0.0;            # gen 1 at bus 1
         0.0 1.0;            # gen 2 at bus 2
         0.0 0.0]            # bus 3: load only

# Line susceptances
b = [-10.0, -10.0]                  # susceptances (Im(1/z) < 0 for inductive)

net = DCNetwork(n, m, k, A, G_inc, b;
                fmax    = [0.5, 10.0],
                gmax    = [2.0, 2.0],
                cq      = [1.0, 1.0],     # quadratic cost -> smooth LMPs
                cl      = [10.0, 50.0],
                ref_bus = 1,
                tau     = 0.0)

# Small loads at buses 1 and 2, larger load at bus 3.
d = [0.05, 0.05, 1.0]

prob = DCOPFProblem(net, d)
PowerDiff.solve!(prob)

println("Base case LMPs: ", calc_lmp(prob))

# ---------------------------------------------------------------------------
# 2. Analytical Jacobian ∂LMP/∂fmax (n × m).
# ---------------------------------------------------------------------------
dlmp_dfmax = calc_sensitivity(prob, :lmp, :fmax)

println("\n∂LMP/∂fmax (rows = buses, cols = branches):")
display(Matrix(dlmp_dfmax))
println("\nrow bus IDs:    ", dlmp_dfmax.row_to_id)
println("col branch IDs: ", dlmp_dfmax.col_to_id)

# ---------------------------------------------------------------------------
# 3. Verification: Show that the finite difference approximation is close to the exact derivatives we just computed.
# ---------------------------------------------------------------------------
ε        = 1e-5
fd       = zeros(n, m)
lmp_base = calc_lmp(prob)

for e in 1:m
    fmax_p = copy(net.fmax)
    fmax_p[e] += ε
    net_p = DCNetwork(n, m, k, A, G_inc, b;
                      fmax    = fmax_p,
                      gmax    = net.gmax,
                      cl      = net.cl, cq = net.cq,
                      ref_bus = 1,
                      tau     = 0.0)
    prob_p = DCOPFProblem(net_p, d)
    PowerDiff.solve!(prob_p)
    fd[:, e] = (calc_lmp(prob_p) .- lmp_base) ./ ε
end

println("\nFinite difference reference:")
display(fd)
println("\nmax |analytical − FD| = ", maximum(abs.(Matrix(dlmp_dfmax) .- fd)))
