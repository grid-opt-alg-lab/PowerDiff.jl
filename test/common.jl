# =============================================================================
# Common Test Setup and Utilities
# =============================================================================

using Test
using LinearAlgebra
using SparseArrays
using Statistics
using PowerModelsDiff
using PowerModels
using ForwardDiff
using Ipopt
using JuMP: MOI

# PowerModels test data directory
const PM_DATA_DIR = joinpath(dirname(pathof(PowerModels)), "..", "test", "data", "matpower")

"""
    load_test_case(case_name::String)

Load and prepare a PowerModels test case.
Returns a basic network dictionary or nothing if not found.
"""
function load_test_case(case_name::String)
    case_path = joinpath(PM_DATA_DIR, case_name)
    if isfile(case_path)
        raw = PowerModels.parse_file(case_path)
        return PowerModels.make_basic_network(raw)
    else
        @warn "Test case not found: $case_path"
        return nothing
    end
end

"""
    load_raw_case(case_name::String)

Load a raw (non-basic) PowerModels network.
"""
function load_raw_case(case_name::String)
    case_path = joinpath(PM_DATA_DIR, case_name)
    if isfile(case_path)
        return PowerModels.parse_file(case_path)
    else
        return nothing
    end
end

"""
    create_2bus_network(; fmax=100.0, gmax=10.0, cl=10.0, cq=0.0, tau=0.0)

Create a minimal 2-bus test network.
"""
function create_2bus_network(; fmax=100.0, gmax=10.0, cl=10.0, cq=0.0, tau=0.0)
    n, m, k = 2, 1, 1
    A = sparse([1.0 -1.0])
    G_inc = sparse(reshape([1.0, 0.0], 2, 1))
    b = [-10.0]

    return DCNetwork(n, m, k, A, G_inc, b;
        fmax=[fmax], gmax=[gmax], gmin=[0.0],
        cl=[cl], cq=[cq], ref_bus=1, tau=tau)
end

"""
    create_3bus_congested_network()

Create a 3-bus network with congestion on line 1→3.
"""
function create_3bus_congested_network()
    n, m, k = 3, 2, 2
    A = sparse([
        1.0  0.0 -1.0;   # Line 1: 1→3 (congested)
        0.0  1.0 -1.0    # Line 2: 2→3
    ])
    G_inc = sparse([
        1.0 0.0;   # Gen 1 at bus 1 (cheap)
        0.0 1.0;   # Gen 2 at bus 2 (expensive)
        0.0 0.0    # No gen at bus 3 (load)
    ])
    b = [-10.0, -10.0]

    return DCNetwork(n, m, k, A, G_inc, b;
        fmax=[0.5, 10.0],  # Line 1→3 constrained
        gmax=[10.0, 10.0], gmin=[0.0, 0.0],
        cl=[10.0, 50.0], cq=[0.0, 0.0],
        ref_bus=1, tau=0.0)
end
