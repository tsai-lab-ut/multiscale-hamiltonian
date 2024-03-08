"""
Generate targets given the file of inputs U0.csv.
"""


using Distributed  # for parallel computing
addprocs(40)

include("../utils/parsing_utils.jl")
include("../utils/logging_utils.jl")
include("../utils/saving_utils.jl")
@everywhere include("../utils/ode_solver.jl")
@everywhere include("../problems/problems.jl")

using ArgParse
using ProgressMeter
using TOML


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
        METHODS[config["method"]], v0, x0, 0., config["Delta_t"], config["nsteps"], false)
    return solver
end

function main()
    parsed_args = parse_commandline()
    config = TOML.parsefile(parsed_args["toml_config"])
    output_dir = parsed_args["output_dir"]
    
    # assert the output directory contains U0.csv 
    init_states_file = joinpath(output_dir, "U0.csv")
    @assert isfile(init_states_file) "Output directory does not contain U0.csv!"

    # create logger 
    logger = get_default_logger(output_dir)
    global_logger(logger)

    # save a copy of config file in the output directory 
    save_toml_config(output_dir, config)

    @info "Using $(nworkers()) workers ..."

    @info "Instantiating problem ..."
    prob = get_problem(config["problem"])

    @info "Instantiating integrator ..."
    phi_Dt = get_solver(config["integration"], prob)

    # for each step n
    for n in 1:config["integration"]["N"]
        input_file = joinpath(output_dir, "U$(n-1).csv")
        output_file = joinpath(output_dir, "U$n.csv")

        if isfile(output_file)
            @info "U$n.csv already exists."
            continue
        end
        
        @info "Computing n=$n ..."
        # read inputs from U_{n-1}.csv 
        V_init, X_init = read_csv(input_file, config["integration"]["use_float64x4"] ? Float64x4 : Float64)
        
        # compute outputs and save in U_{n}.csv 
        elapsed_time = @elapsed res = @showprogress pmap(phi_Dt, eachslice(V_init, dims=2), eachslice(X_init, dims=2))
        @everywhere GC.gc()  # <-- important! 
        V_final = hcat([v for (v, x) in res]...)
        X_final = hcat([v for (v, x) in res]...)
        save_csv(output_file, V_final, X_final)

        @info "Done. Elapsed time = $elapsed_time seconds."
    end
end

main()
