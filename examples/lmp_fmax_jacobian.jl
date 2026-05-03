# @author: Samuel Talkington 04/26/2026
# Documentation: https://samueltalkington.com/research/powerdiff/
# Run with `julia lmp_fmax_jacobian.jl`.

# Jacobian of LMPs w.r.t. transmission line capacities (fmax)
# for a 3 bus DC OPF, with a finite difference cross check via update_fmax!.
#
#         line 1 (fmax = 0.5, binding)
#   bus 1 ─────────────────────── bus 3   ← 1.0 pu load
#                                /
#         line 2 (fmax = 10.0)  /
#   bus 2 ───────────────────────
#
# Cheap gen at bus 1 (cl = 10), expensive gen at bus 2 (cl = 50). Line 1
# saturates, so cheap power is rationed and bus 3 prices off the expensive
# gen. Raising fmax[1] lets more cheap power reach bus 3, dropping its LMP —
# ∂LMP[3]/∂fmax[1] is meaningfully negative.

import Pkg
Pkg.activate(; temp=true)
Pkg.add(url="https://github.com/grid-opt-alg-lab/PowerDiff.jl")

using PowerDiff

n, m, k = 3, 2, 2

A = [1.0  0.0 -1.0;
     0.0  1.0 -1.0]

G_inc = [1.0 0.0;
         0.0 1.0;
         0.0 0.0]

b = [-10.0, -10.0]

net = DCNetwork(n, m, k, A, G_inc, b;
                fmax    = [0.5, 10.0],
                gmax    = [2.0, 2.0],
                cq      = [1.0, 1.0],
                cl      = [10.0, 50.0],
                ref_bus = 1,
                tau     = 0.0)

d = [0.05, 0.05, 1.0]

prob = DCOPFProblem(net, d)
PowerDiff.solve!(prob)

println("Base case LMPs: ", calc_lmp(prob))

# Analytical Jacobian ∂LMP/∂fmax (n × m).
dlmp_dfmax = calc_sensitivity(prob, :lmp, :fmax)

println("\n∂LMP/∂fmax (rows = buses, cols = branches):")
display(Matrix(dlmp_dfmax))
println("\nrow bus IDs:    ", dlmp_dfmax.row_to_id)
println("col branch IDs: ", dlmp_dfmax.col_to_id)

# Finite difference reference. Reuse `prob` via update_fmax! — no JuMP rebuild
# per perturbation. Restore the base limits between branches so each FD column
# perturbs from the same base point.
ε        = 1e-5
fd       = zeros(n, m)
fmax_base = copy(net.fmax)
lmp_base  = calc_lmp(prob)

for e in 1:m
    fmax_p = copy(fmax_base)
    fmax_p[e] += ε
    update_fmax!(prob, fmax_p)
    PowerDiff.solve!(prob)
    fd[:, e] = (calc_lmp(prob) .- lmp_base) ./ ε
end
update_fmax!(prob, fmax_base)
PowerDiff.solve!(prob)

println("\nFinite difference reference:")
display(fd)
println("\nmax |analytical − FD| = ", maximum(abs.(Matrix(dlmp_dfmax) .- fd)))
