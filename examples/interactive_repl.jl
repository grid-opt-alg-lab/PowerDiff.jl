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

using PowerDiff, PowerModels
const PM = PowerModels

case_path = joinpath(dirname(pathof(PM)), "..", "test", "data", "matpower", "case14.m")
pm_data = PM.parse_file(case_path)
PM.make_basic_network!(pm_data)

net = DCNetwork(pm_data)
d = PowerDiff.calc_demand_vector(pm_data)

pf = DCPowerFlowState(net, d)

prob = DCOPFProblem(net, d)
PowerDiff.solve!(prob)

n_bus = size(net.A, 2)
n_branch = size(net.A, 1)
n_gen = length(net.gmax)

println()
println("=== Ready ===")
println("  net  :: DCNetwork     ($n_bus buses, $n_branch branches, $n_gen gens)")
println("  d    :: demand vector  (length $(length(d)))")
println("  pf   :: DCPowerFlowState")
println("  prob :: DCOPFProblem   (solved)")
println()
println("Try these:")
println("  calc_sensitivity(pf, :va, :d)       # voltage angles w.r.t. demand")
println("  calc_sensitivity(pf, :f, :d)        # branch flows w.r.t. demand")
println("  calc_sensitivity(pf, :f, :sw)       # branch flows w.r.t. switching")
println("  calc_sensitivity(prob, :pg, :d)     # generation w.r.t. demand")
println("  calc_sensitivity(prob, :lmp, :d)    # LMPs w.r.t. demand")
println("  calc_sensitivity(prob, :f, :d)      # OPF flows w.r.t. demand")
println("  calc_sensitivity(prob, :va, :sw)    # OPF angles w.r.t. switching")
println("  calc_sensitivity(prob, :pg, :cq)    # generation w.r.t. cost (quad)")
println("  calc_sensitivity(prob, :f, :fmax)   # flows w.r.t. flow limits")
