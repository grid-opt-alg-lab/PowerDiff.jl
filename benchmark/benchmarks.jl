using BenchmarkTools
using PowerDiff

const SUITE = BenchmarkGroup()
const CASE_NAME = "pglib_opf_case300_ieee.m"
const CASE_PATH = joinpath(PowerDiff.get_path(:pglib), CASE_NAME)

PowerDiff.silence()

function _parse_benchmark_case(case_path)
    return PowerDiff.parse_file(case_path)
end

net_data = _parse_benchmark_case(CASE_PATH)
prob = DCOPFProblem(net_data)
sol = solve!(prob)
ac_prob = ACOPFProblem(deepcopy(net_data); silent=true)
ac_sol = solve!(ac_prob)

SUITE["parser"] = BenchmarkGroup()
SUITE["parser"][CASE_NAME] = @benchmarkable _parse_benchmark_case($CASE_PATH)

SUITE["dc_opf"] = BenchmarkGroup()
SUITE["dc_opf"]["kkt_jacobian"] = BenchmarkGroup()
kkt_suite = SUITE["dc_opf"]["kkt_jacobian"][CASE_NAME] = BenchmarkGroup()
kkt_suite["full"] = @benchmarkable PowerDiff.calc_kkt_jacobian($prob; sol=$sol)
kkt_suite["demand"] = @benchmarkable PowerDiff.calc_kkt_jacobian_demand($(prob.network), $(prob.d), $sol)
kkt_suite["flowlimit"] = @benchmarkable PowerDiff.calc_kkt_jacobian_flowlimit($prob, $sol)
kkt_suite["cost_linear"] = @benchmarkable PowerDiff.calc_kkt_jacobian_cost_linear($(prob.network))
kkt_suite["cost_quadratic"] = @benchmarkable PowerDiff.calc_kkt_jacobian_cost_quadratic($prob, $sol)
kkt_suite["susceptance"] = @benchmarkable PowerDiff.calc_kkt_jacobian_susceptance($prob, $sol)

SUITE["ac_opf"] = BenchmarkGroup()
SUITE["ac_opf"]["kkt_jacobian"] = BenchmarkGroup()
SUITE["ac_opf"]["kkt_jacobian"][CASE_NAME] =
    @benchmarkable PowerDiff.calc_kkt_jacobian($ac_prob; sol=$ac_sol)
SUITE["ac_opf"]["kkt_param"] = BenchmarkGroup()
SUITE["ac_opf"]["kkt_param"][CASE_NAME] = BenchmarkGroup()
SUITE["ac_opf"]["kkt_param"][CASE_NAME]["switching"] =
    @benchmarkable PowerDiff.calc_kkt_jacobian_param($ac_prob, $ac_sol, :sw)
