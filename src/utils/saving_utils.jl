"""
Saving utils.
"""

using DataFrames, CSV
using Logging
using MultiFloats 
using TOML


function save_toml_config(dir, config)
    filepath = joinpath(dir, "config.toml")
    @info "Saving config file at $filepath ..."

    mkpath(dir)
    open(filepath, "w") do io
        TOML.print(io, config, sorted=true)
    end
end


function save_config(dir, config)
    mkpath(dir)
    CSV.write(joinpath(dir, "config.csv"), config)
end


"""Save P and Q matrices to a csv file."""
function save_csv(
    filepath::String, 
    P::AbstractArray{T, 2},  # shape = (d, N)
    Q::AbstractArray{T, 2},  # shape = (d, N)
    ) where T<:AbstractFloat
    
    df_P = DataFrame(P', "p" .* string.(1:size(P, 1)))
    df_Q = DataFrame(Q', "q" .* string.(1:size(Q, 1)))     
    
    mkpath(dirname(filepath))
    CSV.write(filepath, hcat(df_P, df_Q))
end

"""Save (p, q) tuples to a csv file."""
function save_csv(
    filepath::String, 
    tuples::Vector{Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}},
    ) where T<:AbstractFloat
    
    P = hcat([p for (p, q) in tuples]...)
    Q = hcat([q for (p, q) in tuples]...)

    df_P = DataFrame(P', "p" .* string.(1:size(P, 1)))
    df_Q = DataFrame(Q', "q" .* string.(1:size(Q, 1)))     
    
    mkpath(dirname(filepath))
    CSV.write(filepath, hcat(df_P, df_Q))
end

"""Save a dictionary of vectors to a csv file."""
function save_csv(
    filepath::String, 
    dict::Dict{String, Vector})
    
    mkpath(dirname(filepath))
    CSV.write(filepath, DataFrame(dict))
end

"""Read P and Q matrices from a csv file."""
function read_csv(filepath::String, dtype::Type)
    
    if dtype == Float64x4
        df = CSV.read(filepath, DataFrame, types=BigFloat)
        df = convert.(Float64x4, df) 
    else
        df = CSV.read(filepath, DataFrame, types=dtype)
    end
    
    dim = Int(ncol(df)/2)
    df_P = df[:, 1:dim]
    df_Q = df[:, dim+1:end]    
    
    P = Matrix(df_P)'
    Q = Matrix(df_Q)'
    
    return P, Q
end
