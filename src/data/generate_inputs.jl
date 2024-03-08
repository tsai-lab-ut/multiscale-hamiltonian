"""
Generate inputs.
"""

using Distributed  # for parallel computing
addprocs(40)

include("../utils/parsing_utils.jl")
include("../utils/logging_utils.jl")
include("../utils/saving_utils.jl")
@everywhere include("HMC.jl")
@everywhere include("../utils/ode_solver.jl")
@everywhere include("../problems/problems.jl")

using ArgParse
using TOML
@everywhere using .HMC


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "toml_config"
            help = "TOML config file"
            arg_type = String
            required = true
        "--output_dir"
            help = "output directory"
            arg_type = String
            default = "."
    end

    return parse_args(s)
end

function get_solver(config::Dict{String, Any}, prob::SeparableHamiltonianSystem)
    solver = (v0, x0) -> ode_solve(
        (ddx, dx, x, p, t) -> compute_ddx!(prob, ddx, dx, x), 
        METHODS[config["method"]], v0, x0, 0., config["dt"], config["nsteps"], false)
    return solver
end

function main()
    parsed_args = parse_commandline()
    config = TOML.parsefile(parsed_args["toml_config"])
    output_dir = parsed_args["output_dir"]
    
    # create logger 
    logger = get_default_logger(output_dir)
    global_logger(logger)

    # save a copy of config file in the output directory 
    save_toml_config(output_dir, config)

    @info "Using $(nworkers()) workers ..."

    @info "Instantiating problem ..."
    prob = get_problem(config["problem"])

    @info "Instantiating integrator ..."
    phi_dt = get_solver(config["integration"], prob)

    @info "Generating initial state ..."
    v0, x0 = initial_condition(prob, config["integration"]["use_float64x4"] ? Float64x4 : Float64)
    H0 = compute_H(prob, v0, x0)
    @info "v0: $v0 \nx0: $x0\nH0: $(compute_H(prob, v0, x0))\nK0: $(compute_K(prob, v0))\nU0: $(compute_U(prob, x0))"

    @info "Sampling chains!"
    alg_cfg = config["algorithm"]
    if alg_cfg["_name_"] == "hmc-H0"
        transition = (v, x) -> hmc_H0_transition(v, x, mass(prob), H0,
            x -> compute_U(prob, x), (v, x) -> compute_H(prob, v, x), phi_dt;
            nsteps=alg_cfg["n_steps_per_trans"],
            sigma=alg_cfg["sigma"],
            with_rejection=alg_cfg["with_rejection"]
        )
        elapsed_time = @elapsed res = HMC.chain_ensemble(v0, x0, transition; 
            num_chains=alg_cfg["n_chains"], 
            num_transitions=alg_cfg["n_trans_per_chain"]
        )
    elseif alg_cfg["_name_"] == "hmc"
        transition = (v, x) -> hmc_transition(v, x, mass(prob),
            (v, x) -> compute_H(prob, v, x), phi_dt;
            nsteps=alg_cfg["n_steps_per_trans"],
            beta=alg_cfg["beta"],
            with_rejection=alg_cfg["with_rejection"]
        )
        elapsed_time = @elapsed res = HMC.chain_ensemble(v0, x0, transition; 
            num_chains=alg_cfg["n_chains"],
            num_transitions=alg_cfg["n_trans_per_chain"]
        )
    elseif alg_cfg["_name_"] == "trajensemble"
        transition = (v, x) -> hmc_H0_transition(v, x, mass(prob), H0,
            x -> compute_U(prob, x), (v, x) -> compute_H(prob, v, x), phi_dt;
            nsteps=alg_cfg["n_steps_per_chain"],
            sigma=alg_cfg["sigma"],
            with_rejection=false
        )
        elapsed_time = @elapsed res = HMC.chain_ensemble(v0, x0, transition; 
            num_chains=alg_cfg["n_chains"], 
            num_transitions=1
        )
    end
    @info "Done generating inputs. Elapsed time = $elapsed_time seconds. Number of samples = $(length(res))."
    
    filepath = joinpath(output_dir, "U0.csv")
    @info "Saving results at $filepath ..."
    save_csv(filepath, res)
end

main()