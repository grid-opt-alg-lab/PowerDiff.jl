# This MRE attempts to solve a DC OPF problem using PowerDiff. 
# If PowerDiff fails, it falls back to solving the same problem using PowerModels to ensure that the issue is isolated to PowerDiff.
# It provides two test cases: "RTS_GMLC.m" and "IEEE300.m". Comment out either/or to switch between them.
# RTS should succeed with PowerDiff, while IEEE300 fails.

using Test
using LinearAlgebra
using SparseArrays
using Statistics
using PowerDiff
using PowerModels
using ForwardDiff
using Ipopt
using JuMP
using JSON


function main()
    #testCase = "MRE_matpowerFiles/RTS_GMLC.m"
    testCase = "MRE_matpowerFiles/WECC240.m"
    net = PowerModels.parse_file(testCase)
    d = PowerDiff.calc_demand_vector(net)
    dc_net = PowerDiff.DCNetwork(net)
    prob = PowerDiff.DCOPFProblem(dc_net, d)
    try
        solve!(prob)
        println("OPF with PowerDiff succeeded.")
    catch e
        @warn "Solve failed: $e"
        solve_opf(testCase, DCPPowerModel, Ipopt.Optimizer)
        println("OPF with PowerDiff failed, but PowerModels succeeded.")
    end
    
end
Base.invokelatest(main)