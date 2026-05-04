using BenchmarkTools
using PowerDiff

const PM = PowerDiff.PM
const SUITE = BenchmarkGroup()

PM.silence()
PowerDiff.silence()

function _load_benchmark_case()
    pm_dir = joinpath(dirname(pathof(PM)), "..", "test", "data", "matpower")
    for case_name in ("case30.m", "case24.m", "case14.m", "case9.m", "case5.m")
        case_path = joinpath(pm_dir, case_name)
        isfile(case_path) || continue
        raw = PM.parse_file(case_path)
        return case_name, PM.make_basic_network(raw)
    end
    error("No bundled PowerModels MATPOWER benchmark case found")
end

case_name, net_data = _load_benchmark_case()
prob = DCOPFProblem(net_data)
sol = solve!(prob)

SUITE["dc_opf"] = BenchmarkGroup()
SUITE["dc_opf"]["kkt_jacobian"] = BenchmarkGroup()
kkt_suite = SUITE["dc_opf"]["kkt_jacobian"][case_name] = BenchmarkGroup()
kkt_suite["full"] = @benchmarkable PowerDiff.calc_kkt_jacobian($prob; sol=$sol)
kkt_suite["demand"] = @benchmarkable PowerDiff.calc_kkt_jacobian_demand($(prob.network), $(prob.d), $sol)
kkt_suite["flowlimit"] = @benchmarkable PowerDiff.calc_kkt_jacobian_flowlimit($prob, $sol)
kkt_suite["cost_linear"] = @benchmarkable PowerDiff.calc_kkt_jacobian_cost_linear($(prob.network))
kkt_suite["cost_quadratic"] = @benchmarkable PowerDiff.calc_kkt_jacobian_cost_quadratic($prob, $sol)
kkt_suite["susceptance"] = @benchmarkable PowerDiff.calc_kkt_jacobian_susceptance($prob, $sol)
