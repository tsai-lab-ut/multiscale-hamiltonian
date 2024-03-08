"""
Hamiltonian Monte Carlo methods.
"""

module HMC

using Distributed
using Logging
using ProgressMeter
using Random
include("sampling.jl")


"""HMC-H0 transition function."""
function hmc_H0_transition(
    v::AbstractArray{T, 1}, 
    x::AbstractArray{T, 1},
    mass::AbstractArray{Float64, 1},
    H0::T,
    compute_U::Function,
    compute_H::Function,
    phi_dt::Function;
    nsteps::Int=1,
    sigma::Float64=1e-2,
    with_rejection::Bool=false
    ) where T<:AbstractFloat
        
    res = Vector{Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}}()
        
    # step 1: momentum refreshment   
    K0 = H0 - compute_U(x)
    K_new = 0
    while K_new <= 0
        K_new = normal(K0, abs(sigma*H0))  # TODO: reconsider definition of std 
    end
    vhat = nSphereSampling(length(v))
    v = vhat ./ sqrt.(mass) * sqrt(2 * K_new)

    # step 2: integration in time with accept/reject mechanism
    for n in 1:nsteps
        v_temp, x_temp = phi_dt(v, x)

        if with_rejection
            dH = compute_H(v_temp, x_temp) - compute_H(v, x)  # TODO: should H be dimensionless? 
            acceptance = min(1, exp(-dH))
            gamma = bernoulli(acceptance)
            if gamma == 0
                v = -v
                @info "Reject. Acceptance rate = $acceptance."
            else
                v = v_temp
                x = x_temp
            end
        else
            v = v_temp
            x = x_temp
        end
        push!(res, (v, x))
    end 

    return res
end


"""HMC transition function."""
function hmc_transition(
    v::AbstractArray{T, 1}, 
    x::AbstractArray{T, 1},
    mass::AbstractArray{Float64, 1},
    compute_H::Function,
    phi_dt::Function;
    nsteps::Int=1,
    beta::Float64=1.,
    with_rejection::Bool=false
    ) where T<:AbstractFloat

    res = Vector{Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}}()

    # step 1: momentum refreshment
    vhat = randn(length(v))
    v = vhat ./ sqrt.(mass) / sqrt(beta)

    # step 2: integration in time with accept/reject mechanism
    for n in 1:nsteps
        v_temp, x_temp = phi_dt(v, x)

        if with_rejection
            dH = compute_H(v_temp, x_temp) - compute_H(v, x)  # TODO: should H be dimensionless? 
            acceptance = min(1, exp(-dH))
            gamma = bernoulli(acceptance)
            if gamma == 0
                v = -v
                @info "Reject. Acceptance rate = $acceptance."
            else
                v = v_temp
                x = x_temp
            end
        else
            v = v_temp
            x = x_temp
        end
        push!(res, (v, x))
    end 

    return res
end

    
function chain(
    v0::AbstractArray{T, 1}, 
    x0::AbstractArray{T, 1}, 
    transition_func::Function,
    num_transitions::Int, 
    seed::Int
    ) where T<:AbstractFloat

    Random.seed!(seed)        
    samples = Vector{Tuple{AbstractArray{T, 1}, AbstractArray{T, 1}}}()
    v = v0
    x = x0

    for i in 1:num_transitions
        res = transition_func(v, x)
        samples = vcat(samples, res)
        v, x = res[end]
    end
    return samples 
end


function chain_ensemble(
    v0::AbstractArray{T, 1}, 
    x0::AbstractArray{T, 1}, 
    transition_func::Function;
    num_chains::Int=1, 
    num_transitions::Int=1
    ) where T<:AbstractFloat

    seeds = 1:num_chains
    res = @showprogress pmap(s->chain(v0, x0, transition_func, num_transitions, s), seeds)

    return vcat(res...)
end

export hmc_H0_transition, hmc_transition, chain_ensemble

end