# Voltage-Power Sensitivity Analysis for AC Power Flow
#
# Computes analytical sensitivity coefficients ∂v/∂p and ∂v/∂q using
# implicit differentiation on the AC power flow equations.
#
# Reference:
# K. Christakou, et al., "Efficient Computation of Sensitivity Coefficients
# of Node Voltages and Line Currents in Unbalanced Radial Electrical
# Distribution Networks", IEEE Trans. Smart Grid, vol. 4, no. 2, pp. 741-750, 2013.

using LinearAlgebra
using SparseArrays

# =============================================================================
# Voltage-Power Sensitivity
# =============================================================================

"""
    calc_voltage_power_sensitivities(v, Y; idx_slack=1, full=true)

Compute sensitivity of bus voltages with respect to active and reactive power injections.

Uses implicit differentiation on the AC power flow equations. The power flow
equations in rectangular coordinates are linearized around the operating point
to compute the sensitivity matrices.

# Arguments
- `v::Vector{ComplexF64}`: Voltage phasors at all buses
- `Y::AbstractMatrix{ComplexF64}`: Bus admittance matrix

# Keyword Arguments
- `idx_slack::Int=1`: Index of the slack (reference) bus
- `full::Bool=true`: If true, include zero rows/columns for slack bus

# Returns
NamedTuple with fields:
- `∂v_∂p`: Complex phasor voltage sensitivity to active power (n × n)
- `∂v_∂q`: Complex phasor voltage sensitivity to reactive power (n × n)
- `∂vm_∂p`: Voltage magnitude sensitivity to active power (n × n)
- `∂vm_∂q`: Voltage magnitude sensitivity to reactive power (n × n)

# Example
```julia
Y = calc_basic_admittance_matrix(net)
v = calc_basic_bus_voltage(net)
sens = calc_voltage_power_sensitivities(v, Y)
# How does voltage at bus 3 change when active power at bus 2 increases?
dvdp = sens.∂vm_∂p[3, 2]
```
"""
function calc_voltage_power_sensitivities(
    v::AbstractVector{ComplexF64},
    Y::AbstractMatrix{ComplexF64};
    idx_slack::Int=1,
    full::Bool=true
)
    ∂v_∂p, ∂vm_∂p = calc_voltage_active_power_sensitivities(v, Y; idx_slack=idx_slack, full=full)
    ∂v_∂q, ∂vm_∂q = calc_voltage_reactive_power_sensitivities(v, Y; idx_slack=idx_slack, full=full)

    return VoltagePowerSensitivity(∂v_∂p, ∂v_∂q, ∂vm_∂p, ∂vm_∂q)
end

"""
    calc_voltage_power_sensitivities(net::Dict; full=true)

Compute voltage-power sensitivities from a solved PowerModels network.

The network must be a basic network with a solved power flow.
"""
function calc_voltage_power_sensitivities(net::Dict; full::Bool=true)
    @assert haskey(net, "basic_network") && net["basic_network"] "Network must be a basic network"

    Y = PM.calc_basic_admittance_matrix(net)
    v = PM.calc_basic_bus_voltage(net)
    idx_slack = _find_slack_bus(net)

    return calc_voltage_power_sensitivities(v, Y; idx_slack=idx_slack, full=full)
end

"""
    calc_voltage_power_sensitivities(state::ACPowerFlowState; full=true)

Compute voltage-power sensitivities from an ACPowerFlowState.

This method provides a unified interface consistent with DC OPF sensitivities.
"""
function calc_voltage_power_sensitivities(state::ACPowerFlowState; full::Bool=true)
    return calc_voltage_power_sensitivities(state.v, state.Y; idx_slack=state.idx_slack, full=full)
end

"""
    calc_voltage_active_power_sensitivities(v, Y; idx_slack=1, full=true)

Compute sensitivity of bus voltages with respect to active power injections.

# Returns
Tuple (∂v_∂p, ∂vm_∂p) where:
- `∂v_∂p`: Complex phasor sensitivity ∂v/∂p (n × n or (n-1) × (n-1) if full=false)
- `∂vm_∂p`: Magnitude sensitivity ∂|v|/∂p
"""
function calc_voltage_active_power_sensitivities(
    v::AbstractVector{ComplexF64},
    Y::AbstractMatrix{ComplexF64};
    idx_slack::Int=1,
    full::Bool=true
)
    # Build the linearized system matrix
    A = _build_voltage_sensitivity_matrix(v, Y, idx_slack)

    # Remove slack bus from voltage vector
    v_ = v[1:end .!= idx_slack]
    d = length(v_)

    # Solve for each column (unit power perturbation at bus k)
    ∂v_∂p = Matrix{ComplexF64}(undef, d, d)
    ∂vm_∂p = Matrix{Float64}(undef, d, d)

    for k in 1:d
        if abs(v_[k]) > 1e-6
            # Right-hand side: unit perturbation in active power at bus k
            b = zeros(2d)
            b[k] = 1.0

            # Solve the linear system
            x = A \ b

            # Convert to voltage sensitivities
            # x = [∂v_re; ∂v_im], so ∂v = ∂v_re - im*∂v_im (conjugate convention)
            ∂v_∂p[:, k] = x[1:d] - im * x[d+1:2d]

            # Magnitude sensitivity: ∂|v|/∂p = Re(∂v/∂p · conj(v)) / |v|
            ∂vm_∂p[:, k] = real.(∂v_∂p[:, k] .* conj.(v_)) ./ abs.(v_)
        else
            ∂v_∂p[:, k] .= 0
            ∂vm_∂p[:, k] .= 0
        end
    end

    # Optionally expand to include slack bus as zeros
    if full
        ∂v_∂p = _insert_slack_zeros(∂v_∂p, idx_slack, ComplexF64)
        ∂vm_∂p = _insert_slack_zeros(∂vm_∂p, idx_slack, Float64)
    end

    return ∂v_∂p, ∂vm_∂p
end

"""
    calc_voltage_reactive_power_sensitivities(v, Y; idx_slack=1, full=true)

Compute sensitivity of bus voltages with respect to reactive power injections.

# Returns
Tuple (∂v_∂q, ∂vm_∂q) where:
- `∂v_∂q`: Complex phasor sensitivity ∂v/∂q
- `∂vm_∂q`: Magnitude sensitivity ∂|v|/∂q
"""
function calc_voltage_reactive_power_sensitivities(
    v::AbstractVector{ComplexF64},
    Y::AbstractMatrix{ComplexF64};
    idx_slack::Int=1,
    full::Bool=true
)
    # Build the linearized system matrix
    A = _build_voltage_sensitivity_matrix(v, Y, idx_slack)

    # Remove slack bus from voltage vector
    v_ = v[1:end .!= idx_slack]
    d = length(v_)

    # Solve for each column (unit reactive power perturbation at bus k)
    ∂v_∂q = Matrix{ComplexF64}(undef, d, d)
    ∂vm_∂q = Matrix{Float64}(undef, d, d)

    for k in 1:d
        if abs(v_[k]) > 1e-6
            # Right-hand side: unit perturbation in reactive power at bus k
            # Note: power flow equation is p + jq = conj(v) * Y * v
            # Derivative w.r.t. q has opposite sign in imaginary part
            b = zeros(2d)
            b[d + k] = -1.0  # Note the negative sign

            # Solve the linear system
            x = A \ b

            # Convert to voltage sensitivities
            ∂v_∂q[:, k] = x[1:d] - im * x[d+1:2d]

            # Magnitude sensitivity
            ∂vm_∂q[:, k] = real.(∂v_∂q[:, k] .* conj.(v_)) ./ abs.(v_)
        else
            ∂v_∂q[:, k] .= 0
            ∂vm_∂q[:, k] .= 0
        end
    end

    # Optionally expand to include slack bus as zeros
    if full
        ∂v_∂q = _insert_slack_zeros(∂v_∂q, idx_slack, ComplexF64)
        ∂vm_∂q = _insert_slack_zeros(∂vm_∂q, idx_slack, Float64)
    end

    return ∂v_∂q, ∂vm_∂q
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    _build_voltage_sensitivity_matrix(v, Y, idx_slack)

Build the linearized system matrix A for voltage-power sensitivity computation.

The matrix A relates perturbations in (v_re, v_im) to perturbations in (p, q)
based on the power flow equations:
    p = Re(conj(v) * (Y * v))
    q = Im(conj(v) * (Y * v))

The matrix has structure:
    A = [∂p/∂v_re  ∂p/∂v_im]
        [∂q/∂v_re  ∂q/∂v_im]

evaluated at the operating point (v, Y).
"""
function _build_voltage_sensitivity_matrix(
    v::AbstractVector{ComplexF64},
    Y::AbstractMatrix{ComplexF64},
    idx_slack::Int
)
    # Remove slack bus
    v_ = v[1:end .!= idx_slack]
    Y_ = Y[1:end .!= idx_slack, 1:end .!= idx_slack]

    # Current injection at non-slack buses: I = Y * v
    I = (Y * v)[1:end .!= idx_slack]

    d = length(v_)

    # H = Diag(conj(v)) * Y (used in power flow Jacobian)
    H = Diagonal(conj.(v_)) * Y_

    # Build the 2d × 2d Jacobian matrix
    A = spzeros(2d, 2d)

    for i in 1:d
        for j in 1:d
            if i == j
                # Diagonal blocks include current injection terms
                A[i, j] = real(I[i]) + real(H[i, j])
                A[d+i, j] = imag(I[i]) + imag(H[i, j])
                A[i, d+j] = imag(I[i]) - imag(H[i, j])
                A[d+i, d+j] = real(H[i, j]) - real(I[i])
            else
                # Off-diagonal blocks
                A[i, j] = real(H[i, j])
                A[d+i, j] = imag(H[i, j])
                A[i, d+j] = -imag(H[i, j])
                A[d+i, d+j] = real(H[i, j])
            end
        end
    end

    return A
end

"""
Insert zero rows and columns for the slack bus into a reduced matrix.
"""
function _insert_slack_zeros(K::Matrix{T}, idx_slack::Int, ::Type{T}) where T
    d = size(K, 1)
    n = d + 1

    K_full = zeros(T, n, n)

    # Copy values, skipping slack index
    row_idx = 1
    for i in 1:n
        if i == idx_slack
            continue
        end
        col_idx = 1
        for j in 1:n
            if j == idx_slack
                continue
            end
            K_full[i, j] = K[row_idx, col_idx]
            col_idx += 1
        end
        row_idx += 1
    end

    return K_full
end

# NOTE: _find_slack_bus(net::Dict) is defined in types/ac_network.jl
# NOTE: ACPowerFlowState(net::Dict) constructor is defined in types/ac_network.jl
