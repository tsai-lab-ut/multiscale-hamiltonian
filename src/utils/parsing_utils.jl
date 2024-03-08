"""
Config parsing utils.
"""

function to_kwargs(config::Dict{String, Any})
    kwargs = Dict()
    for (k, v) in config
        if k != "_name_"
            kwargs[Symbol(k)] = v
        end
    end
    return kwargs
end

function get_problem(config::Dict{String, Any})
    name = config["_name_"]
    kwargs = to_kwargs(config)
    
    prob = Dict(
        "fpu"=>FPU, 
        "1body"=>OneBodyKepler, 
        "2body"=>TwoBodyKepler, 
        "argoncrystal"=>ArgonCrystal,
        "nbody"=>NBody)[name](; kwargs...)

    return prob
end
