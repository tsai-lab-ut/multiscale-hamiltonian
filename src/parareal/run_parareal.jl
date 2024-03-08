"""
Run parareal algorithm. 
"""

using Distributed  # for parallel computing
addprocs(40)

include("../utils/parsing_utils.jl")
include("../utils/logging_utils.jl")
include("../utils/saving_utils.jl")
# include("../utils/python_model.jl")
include("Parareal.jl")
@everywhere include("../utils/ode_solver.jl")
@everywhere include("../problems/problems.jl")

using ArgParse
using TOML
using .Parareal


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

@everywhere function ode_solve_wrapper(
    v0::AbstractArray{T, 1}, 
    x0::AbstractArray{T, 1};
    func::Function,
    method::String, 
    H::Float64,
    nsteps::Integer,
    T2::Type) where T<:AbstractFloat

    if T != T2
        v0 = convert.(T2, v0)
        x0 = convert.(T2, x0)
    end
    v, x = ode_solve(func, METHODS[method], v0, x0, 0., H, nsteps, false)
    if T != T2
        v = convert.(T, v)
        x = convert.(T, x)
    end
    return v, x
end

function get_solvers(config::Dict{String, Any}, prob::SeparableHamiltonianSystem)
    fine_solve = (v0, x0) -> ode_solve_wrapper(
        v0, x0; 
        func=(ddx, dx, x, p, t) -> compute_ddx!(prob, ddx, dx, x), 
        method=config["fine_method"],
        H=config["Delta_t"],
        nsteps=config["Nf"], 
        T2=Float64x4
    )

    if ~haskey(config, "nn_ckpt_path")
        coarse_solve = (v0, x0) -> ode_solve_wrapper(
            v0, x0; 
            func=(ddx, dx, x, p, t) -> compute_ddx!(prob, ddx, dx, x), 
            method=config["coarse_method"],
            H=config["Delta_t"],
            nsteps=config["Nc"], 
            T2=config["use_float64x4"] ? Float64x4 : Float64
        )
    else
        nn_func = load_nn(config["nn_ckpt_path"])
        coarse_solve = (v0, x0) -> nn_solve(v0, x0, nn_func)
    end

    return fine_solve, coarse_solve
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

    @info "Instantiating integrators ..."
    fine_solve, coarse_solve = get_solvers(config["integration"], prob)

    @info "Generating initial state ..."
    v0, x0 = initial_condition(prob, config["integration"]["use_float64x4"] ? Float64x4 : Float64)
    @info "v0: $v0 \nx0: $x0\nH0: $(compute_H(prob, v0, x0))\nK0: $(compute_K(prob, v0))\nU0: $(compute_U(prob, x0))"

    @info "Running parareal!"
    alg_name = config["algorithm"]["_name_"]
    alg_kwargs = to_kwargs(config["algorithm"])
    if alg_name == "plain"
        elapsed_time = @elapsed Parareal.plain(
            v0, x0, fine_solve, coarse_solve, config["integration"]["N"], config["integration"]["niters"];
            output_dir=output_dir, alg_kwargs...)
    elseif alg_name == "procrustes"
        Lambda = (v, x) -> construct_z(prob, v, x)
        Theta = (v, x, corrector) -> correct_phase(prob, v, x, corrector)
        elapsed_time = @elapsed Parareal.procrustes(
            v0, x0, fine_solve, coarse_solve, config["integration"]["N"], config["integration"]["niters"],
            Lambda, Theta; output_dir=output_dir, alg_kwargs...)
    elseif alg_name == "interpolative"
        elapsed_time = @elapsed Parareal.interpolative(
            v0, x0, fine_solve, coarse_solve, config["integration"]["N"], config["integration"]["niters"];
            output_dir=output_dir, alg_kwargs...)
    elseif alg_name == "sequential"
        elapsed_time = @elapsed Parareal.plain(
            v0, x0, fine_solve, fine_solve, config["integration"]["N"], 0;
            output_dir=output_dir, alg_kwargs...)
    end
    @info "Done running parareal. Elapsed time = $elapsed_time seconds."
end

main()
