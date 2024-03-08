"""
Two-body Kepler problem in 3D
"""

Base.@kwdef struct TwoBodyKepler <: SeparableHamiltonianSystem
    g12::Float64 = 1e-5
    ecc1::Float64 = 0.4
    ecc2::Float64 = 0.5
end

function initial_condition(prob::TwoBodyKepler, T::Type)
    e1 = convert(T, prob.ecc1)
    e2 = convert(T, prob.ecc2)

    x0 = [1-e1,     0.,                     0.,     cos(pi/4)*(1-e2),   0.,                     sin(pi/4)*(1-e2)]
    v0 = [0.,       sqrt((1+e1)/(1-e1)),    0.,     0.,                 sqrt((1+e2)/(1-e2)),    0.]

    return v0, x0
end 

function nondimensionalize(prob::TwoBodyKepler, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return v, x
end

function dimensionalize(prob::TwoBodyKepler, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return v, x
end

function compute_ddx!(prob::TwoBodyKepler, ddx, dx, x)   
    x1 = @view x[1:3]
    x2 = @view x[4:6]

    ddx[1:3] = - x1 ./ norm(x1)^3 - prob.g12*(x1-x2) ./ norm(x1-x2)^3
    ddx[4:6] = - x2 ./ norm(x2)^3 + prob.g12*(x1-x2) ./ norm(x1-x2)^3

    nothing 
end

function compute_ddx(prob::TwoBodyKepler, x::AbstractArray{T, 1}) where T<:AbstractFloat    
    x1 = @view x[1:3]
    x2 = @view x[4:6]
    
    ddx = zero(x)
    ddx[1:3] = - x1 ./ norm(x1)^3 - prob.g12*(x1-x2) ./ norm(x1-x2)^3
    ddx[4:6] = - x2 ./ norm(x2)^3 + prob.g12*(x1-x2) ./ norm(x1-x2)^3
    
    return ddx
end

function compute_K(prob::TwoBodyKepler, v::AbstractArray{T, 1}) where T<:AbstractFloat
    return 0.5 * v' * v
end

function compute_U(prob::TwoBodyKepler, x::AbstractArray{T, 1}) where T<:AbstractFloat
    x1 = @view x[1:3]
    x2 = @view x[4:6]
    U = - (1/norm(x1) + 1/norm(x2) + prob.g12/norm(x1-x2))
    return U
end
