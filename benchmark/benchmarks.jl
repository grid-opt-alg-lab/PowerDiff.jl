using BenchmarkTools
using PowerDiff
import PowerModels

const PM = PowerModels
const SUITE = BenchmarkGroup()

PM.silence()
PowerDiff.silence()

function _parse_benchmark_case(case_path)
    if isdefined(PowerDiff, :parse_file)
        return PowerDiff.parse_file(case_path)
    end
    return PM.make_basic_network(PM.parse_file(case_path))
end

function _load_benchmark_case()
    pm_dir = joinpath(dirname(pathof(PM)), "..", "test", "data", "matpower")
    for case_name in ("case30.m", "case24.m", "case14.m", "case9.m", "case5.m")
        case_path = joinpath(pm_dir, case_name)
        isfile(case_path) || continue
        net_data = _parse_benchmark_case(case_path)
        return case_name, case_path, net_data
    end
    error("No bundled PowerModels MATPOWER benchmark case found")
end

case_name, case_path, net_data = _load_benchmark_case()
prob = DCOPFProblem(net_data)
sol = solve!(prob)
ac_prob = ACOPFProblem(deepcopy(net_data); silent=true)
ac_sol = solve!(ac_prob)

SUITE["parser"] = BenchmarkGroup()
SUITE["parser"][case_name] = @benchmarkable _parse_benchmark_case($case_path)

SUITE["dc_opf"] = BenchmarkGroup()
SUITE["dc_opf"]["kkt_jacobian"] = BenchmarkGroup()
kkt_suite = SUITE["dc_opf"]["kkt_jacobian"][case_name] = BenchmarkGroup()
kkt_suite["full"] = @benchmarkable PowerDiff.calc_kkt_jacobian($prob; sol=$sol)
kkt_suite["demand"] = @benchmarkable PowerDiff.calc_kkt_jacobian_demand($(prob.network), $(prob.d), $sol)
kkt_suite["flowlimit"] = @benchmarkable PowerDiff.calc_kkt_jacobian_flowlimit($prob, $sol)
kkt_suite["cost_linear"] = @benchmarkable PowerDiff.calc_kkt_jacobian_cost_linear($(prob.network))
kkt_suite["cost_quadratic"] = @benchmarkable PowerDiff.calc_kkt_jacobian_cost_quadratic($prob, $sol)
kkt_suite["susceptance"] = @benchmarkable PowerDiff.calc_kkt_jacobian_susceptance($prob, $sol)

SUITE["ac_opf"] = BenchmarkGroup()
SUITE["ac_opf"]["kkt_jacobian"] = BenchmarkGroup()
SUITE["ac_opf"]["kkt_jacobian"][case_name] =
    @benchmarkable PowerDiff.calc_kkt_jacobian($ac_prob; sol=$ac_sol)
SUITE["ac_opf"]["kkt_param"] = BenchmarkGroup()
SUITE["ac_opf"]["kkt_param"][case_name] = BenchmarkGroup()
SUITE["ac_opf"]["kkt_param"][case_name]["switching"] =
    @benchmarkable PowerDiff.calc_kkt_jacobian_param($ac_prob, $ac_sol, :sw)
