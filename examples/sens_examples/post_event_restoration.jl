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

using PowerDiff

PowerDiff.silence()

const CASE_PATH = joinpath(@__DIR__, "matPowerFiles", "IEEE300.m")
const DAMAGED_BRANCH_IDS = [3, 6, 16, 17, 37]
const DAMAGED_SWITCH_STATE = 0.01
const LOCAL_SWITCH_STEP = 0.01
const BASE_MVA = 100.0

"""Solve the IEEE300 DC OPF at supplied continuous branch switching states."""
function solve_at_switching(network_data, demand, switching)
    network = DCNetwork(network_data)
    network.sw .= switching
    problem = DCOPFProblem(network, demand)
    solution = solve!(problem)
    return problem, solution
end

function main()
    network_data = PowerDiff.parse_file(CASE_PATH)
    base_network = DCNetwork(network_data)
    demand = calc_demand_vector(network_data)
    healthy_switching = ones(base_network.m)

    _, healthy_solution = solve_at_switching(network_data, demand, healthy_switching)

    # These five branch IDs form a synthetic damage scenario on IEEE300. A
    # continuous switching state of 0.01 represents a nearly open branch while
    # retaining a well-defined local switching derivative.
    damaged_indices = [base_network.id_map.branch_to_idx[id] for id in DAMAGED_BRANCH_IDS]
    damaged_switching = copy(healthy_switching)
    damaged_switching[damaged_indices] .= DAMAGED_SWITCH_STATE

    damaged_problem, damaged_solution =
        solve_at_switching(network_data, demand, damaged_switching)
    damaged_shed = sum(damaged_solution.psh)

    damaged_shed > 1e-6 || error("The selected damage scenario did not cause load shedding.")

    # If total_shed = 1' * psh, this VJP directly returns
    #
    #     d(total_shed) / d(sw)
    #
    # for all 411 branches, without constructing the full 300-by-411 Jacobian.
    total_shed_sensitivity = PowerDiff.vjp(
        damaged_problem,
        :psh,
        :sw,
        ones(base_network.n),
    )

    # Convert signed local derivatives into nonnegative restoration-priority
    # scores. These normalized weights are probability-like screening weights,
    # not calibrated probabilities that a branch is the best full repair.
    priority_score = Dict(
        branch_id => max(
            0.0,
            -total_shed_sensitivity[base_network.id_map.branch_to_idx[branch_id]],
        )
        for branch_id in DAMAGED_BRANCH_IDS
    )
    score_sum = sum(values(priority_score))
    score_sum > 0.0 || error("No damaged branch has a beneficial local switching sensitivity.")

    priority_weight = Dict(
        branch_id => priority_score[branch_id] / score_sum
        for branch_id in DAMAGED_BRANCH_IDS
    )
    priority_order = sort(
        DAMAGED_BRANCH_IDS;
        by=branch_id -> priority_weight[branch_id],
        rev=true,
    )

    println("Post-event transmission restoration on IEEE300")
    println("Network: $(base_network.n) buses, $(base_network.m) branches, $(base_network.k) generators")
    println("Damaged branches: ", join(DAMAGED_BRANCH_IDS, ", "))
    println("Healthy-grid unserved demand: ", round(BASE_MVA * sum(healthy_solution.psh); digits=3), " MW")
    println("Damaged-grid unserved demand: ", round(BASE_MVA * damaged_shed; digits=3), " MW")
    println()
    println("Local sensitivity priorities and explicit repair checks")
    println("branch | dS/dsw (MW) | priority | predicted +0.01 | actual +0.01 | full repair")

    full_recovery = Dict{Int,Float64}()
    for branch_id in priority_order
        branch_index = base_network.id_map.branch_to_idx[branch_id]
        sensitivity_mw = BASE_MVA * total_shed_sensitivity[branch_index]
        priority_percent = 100.0 * priority_weight[branch_id]

        # Check the local derivative by moving the continuous switching state
        # from 0.01 to 0.02.
        local_switching = copy(damaged_switching)
        local_switching[branch_index] += LOCAL_SWITCH_STEP
        _, local_solution = solve_at_switching(network_data, demand, local_switching)
        actual_local_recovery = BASE_MVA * (damaged_shed - sum(local_solution.psh))
        predicted_local_recovery = -sensitivity_mw * LOCAL_SWITCH_STEP

        # A complete repair is a large change, so verify it with a fresh OPF.
        restored_switching = copy(damaged_switching)
        restored_switching[branch_index] = 1.0
        _, restored_solution = solve_at_switching(network_data, demand, restored_switching)
        actual_full_recovery = BASE_MVA * (damaged_shed - sum(restored_solution.psh))
        full_recovery[branch_id] = actual_full_recovery

        println(
            lpad(branch_id, 6), " | ",
            lpad(round(sensitivity_mw; digits=3), 12), " | ",
            lpad("$(round(priority_percent; digits=1))%", 8), " | ",
            lpad(round(predicted_local_recovery; digits=3), 15), " | ",
            lpad(round(actual_local_recovery; digits=3), 12), " | ",
            round(actual_full_recovery; digits=3), " MW",
        )
    end

    highest_priority_branch = first(priority_order)
    best_realized_branch = argmax(full_recovery)
    println()
    println(
        "Branch $highest_priority_branch has the highest local ",
        "sensitivity-based restoration priority.",
    )
    println("Best full repair after explicit OPF re-solves: branch $best_realized_branch.")
end

main()
