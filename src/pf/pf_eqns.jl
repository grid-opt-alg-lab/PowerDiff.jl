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
# Power Flow Equations (Vectorized, Topology-Agnostic)
# =============================================================================
#
# Provides differentiable power flow equations using vectorized admittance
# representation. Equations are parameterized by G (conductance) and B (susceptance)
# vectors rather than a full Y matrix, enabling efficient implicit differentiation.

# =============================================================================
# Power Flow Equations - Complex Voltage Form
# =============================================================================

"""
    p(v, G, B)

Real power injection at each bus given complex voltage phasors.

Uses the vectorized admittance representation:
    P = Re(diag(v̄) * Y * v) where Y = A' * diag(G + jB) * A
"""
p(v, G, B) = real.(Diagonal(conj.(v)) * laplacian(G, B, length(v)) * v)

"""
    q(v, G, B)

Reactive power injection at each bus given complex voltage phasors.
"""
q(v, G, B) = imag.(Diagonal(conj.(v)) * laplacian(G, B, length(v)) * v)

"""
    vm(v, G, B)

Voltage magnitudes (G, B are unused but kept for interface consistency).
"""
vm(v, G, B) = abs.(v)

# =============================================================================
# Power Flow Equations - Rectangular Voltage Form
# =============================================================================

"""
    pf_eqns(v_re, v_im, G, B)

Complex power injection S = P + jQ in rectangular coordinates.
"""
pf_eqns(v_re, v_im, G, B) = Diagonal(v_re .- im .* v_im) * (laplacian(G, B, length(v_re)) * (v_re .+ im .* v_im))

"""
    p(v_re, v_im, G, B)

Real power injection in rectangular coordinates.
"""
p(v_re, v_im, G, B) = real.(pf_eqns(v_re, v_im, G, B))

"""
    q(v_re, v_im, G, B)

Reactive power injection in rectangular coordinates.
"""
q(v_re, v_im, G, B) = imag.(pf_eqns(v_re, v_im, G, B))

"""
    vm(v_re, v_im, G, B)

Voltage magnitudes from rectangular coordinates.
"""
vm(v_re, v_im, G, B) = hypot.(v_re, v_im)

"""
    vm2(v_re, v_im, G, B)

Squared voltage magnitudes from rectangular coordinates.
"""
vm2(v_re, v_im, G, B) = v_re.^2 .+ v_im.^2

# =============================================================================
# Power Flow Equations - Polar Form
# =============================================================================

"""
    p_polar(vm, δ, G, B)

Real power injection in polar coordinates (vm = voltage magnitude, δ = phase angle).
"""
p_polar(vm, δ, G, B) = p(vm .* cis.(δ), G, B)

"""
    q_polar(vm, δ, G, B)

Reactive power injection in polar coordinates.
"""
q_polar(vm, δ, G, B) = q(vm .* cis.(δ), G, B)

# =============================================================================
# Branch Flow Equations
# =============================================================================

"""
    branch_flow(v, G, B)

Complex branch current flows: I_branch = diag(A*v) * (G + jB)

Note: This uses the full incidence matrix and returns currents for all edges.
"""
branch_flow(v, G, B) = Diagonal(full_incidence_matrix(length(v)) * v) * (G .+ B .* im)

"""
    p_flow(v, G, B)

Real part of branch power flows.
"""
p_flow(v, G, B) = real.(branch_flow(v, G, B))

"""
    q_flow(v, G, B)

Reactive part of branch power flows.
"""
q_flow(v, G, B) = imag.(branch_flow(v, G, B))

# =============================================================================
# Power Flow Jacobians - Complex Voltage Form
# =============================================================================

"""Jacobian ∂P/∂G (real power w.r.t. conductance)"""
∂p∂g(v, G, B) = ForwardDiff.jacobian(G -> p(v, G, B), G)

"""Jacobian ∂P/∂B (real power w.r.t. susceptance)"""
∂p∂b(v, G, B) = ForwardDiff.jacobian(B -> p(v, G, B), B)

"""Jacobian ∂Q/∂G (reactive power w.r.t. conductance)"""
∂q∂g(v, G, B) = ForwardDiff.jacobian(G -> q(v, G, B), G)

"""Jacobian ∂Q/∂B (reactive power w.r.t. susceptance)"""
∂q∂b(v, G, B) = ForwardDiff.jacobian(B -> q(v, G, B), B)

"""Jacobian ∂|V|/∂G (voltage magnitude w.r.t. conductance) - always zero"""
∂vm∂g(v, G, B) = ForwardDiff.jacobian(G -> vm(v, G, B), G)

"""Jacobian ∂|V|/∂B (voltage magnitude w.r.t. susceptance) - always zero"""
∂vm∂b(v, G, B) = ForwardDiff.jacobian(B -> vm(v, G, B), B)

# =============================================================================
# Power Flow Jacobians - Polar Form
# =============================================================================

"""Jacobian ∂P/∂|V| in polar coordinates"""
∂p∂vm(vm, δ, G, B) = ForwardDiff.jacobian(vm -> p(vm .* cis.(δ), G, B), vm)

"""Jacobian ∂Q/∂|V| in polar coordinates"""
∂q∂vm(vm, δ, G, B) = ForwardDiff.jacobian(vm -> q(vm .* cis.(δ), G, B), vm)

"""Jacobian ∂P/∂δ in polar coordinates"""
∂p∂δ(vm, δ, G, B) = ForwardDiff.jacobian(δ -> p(vm .* cis.(δ), G, B), δ)

"""Jacobian ∂Q/∂δ in polar coordinates"""
∂q∂δ(vm, δ, G, B) = ForwardDiff.jacobian(δ -> q(vm .* cis.(δ), G, B), δ)

# =============================================================================
# Power Flow Jacobians - Rectangular Form
# =============================================================================

"""Jacobian ∂P/∂G in rectangular coordinates"""
∂p∂g(v_re, v_im, G, B) = ForwardDiff.jacobian(G -> p(v_re, v_im, G, B), G)

"""Jacobian ∂P/∂B in rectangular coordinates"""
∂p∂b(v_re, v_im, G, B) = ForwardDiff.jacobian(B -> p(v_re, v_im, G, B), B)

"""Jacobian ∂Q/∂G in rectangular coordinates"""
∂q∂g(v_re, v_im, G, B) = ForwardDiff.jacobian(G -> q(v_re, v_im, G, B), G)

"""Jacobian ∂Q/∂B in rectangular coordinates"""
∂q∂b(v_re, v_im, G, B) = ForwardDiff.jacobian(B -> q(v_re, v_im, G, B), B)

"""Jacobian ∂|V|/∂G in rectangular coordinates"""
∂vm∂g(v_re, v_im, G, B) = ForwardDiff.jacobian(G -> vm(v_re, v_im, G, B), G)

"""Jacobian ∂|V|/∂B in rectangular coordinates"""
∂vm∂b(v_re, v_im, G, B) = ForwardDiff.jacobian(B -> vm(v_re, v_im, G, B), B)

"""Jacobian ∂P/∂v_re (real power w.r.t. real voltage)"""
∂p∂v_re(v_re, v_im, G, B) = ForwardDiff.jacobian(v_re -> p(v_re, v_im, G, B), v_re)

"""Jacobian ∂P/∂v_im (real power w.r.t. imaginary voltage)"""
∂p∂v_im(v_re, v_im, G, B) = ForwardDiff.jacobian(v_im -> p(v_re, v_im, G, B), v_im)

"""Jacobian ∂Q/∂v_re (reactive power w.r.t. real voltage)"""
∂q∂v_re(v_re, v_im, G, B) = ForwardDiff.jacobian(v_re -> q(v_re, v_im, G, B), v_re)

"""Jacobian ∂Q/∂v_im (reactive power w.r.t. imaginary voltage)"""
∂q∂v_im(v_re, v_im, G, B) = ForwardDiff.jacobian(v_im -> q(v_re, v_im, G, B), v_im)
