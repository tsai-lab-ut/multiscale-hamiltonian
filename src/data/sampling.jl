"""
Sampling functions.
"""

using Distributions


"""Sample a random point on a unit n-sphere."""
function nSphereSampling(n::Int)

    # generate n Gaussian random variables 
    x = randn(n)
    
    # normalize 
    x /= sqrt(sum(abs2, x)) 
    
    return x
end

"""Generate uniformly distributed random points on a unit n-sphere."""
function nSphereSampling(n::Int, n_samples::Int)
    
    # generate n Gaussian random variables 
    X = randn(n, n_samples)
    
    # normalize 
    X = X ./ sqrt.(sum(abs2, X, dims=1)) 
    
    return X
end

"""Generate uniformly distributed random points in a unit n-ball."""
function nBallSampling(n::Int, n_samples::Int)
    
    # generate n Gaussian random variables 
    X = randn(n, n_samples)
    
    # generate random radius 
    r = rand(1, n_samples) .^ (1/n)
    
    # normalize the vector and multiply by radius 
    X = X ./ sqrt.(sum(abs2, X, dims=1)) 
    X = X .* r
    
    return X
end

"""Generate uniformly distributed random points in a unit n-cube."""
function nCubeSampling(n::Int, n_samples::Int)
    
    X = rand(n, n_samples) .- 0.5
    
    return X
end

"""Generate uniformly distributed random points in a n-dimensional spherical shell (1 < r < 1+epsilon)."""
function nShellSampling(n::Int, n_samples::Int, epsilon::T) where T<:AbstractFloat
    
    # generate n Gaussian random variables 
    X = randn(n, n_samples)
    
    # generate random radius between [1, 1+epsilon]
    u = rand(1, n_samples)
    r = (u * (1. + epsilon)^n + (1 .- u)) .^ (1/n)
    
    # normalize the vector and multiply by radius 
    X = X ./ sqrt.(sum(abs2, X, dims=1)) 
    X = X .* r
    
    return X
end

"""Generate a random variable from a normal distribution."""
function normal(mean::T, std::T) where T<:AbstractFloat
    return randn() * std .+ mean
end

"""Generate random variables from a normal distribution."""
function normal(mean::T, std::T, n_samples::Int) where T<:AbstractFloat
    return randn(n_samples) * std .+ mean
end

"""Generate a Bernoulli random variable."""
function bernoulli(p::T) where T<:AbstractFloat
    dist = Binomial(1, Float64(p))
    return rand(dist)
end

"""Generate random variables from a Bernoulli distribution."""
function bernoulli(p::T, n_samples::Int) where T<:AbstractFloat
    dist = Binomial(1, Float64(p))
    return rand(dist, n_samples)
end