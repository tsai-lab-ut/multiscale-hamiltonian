"""
One-body Kepler problem in 2D
"""

Base.@kwdef struct OneBodyKepler <: SeparableHamiltonianSystem 
    ecc::Float64 = 0.5  # eccentricity
end

function initial_condition(prob::OneBodyKepler, T::Type)
    ecc = convert(T, prob.ecc)
    x0 = [1-ecc,    0.]
    v0 = [0.,       sqrt((1+ecc)/(1-ecc))]
    return v0, x0
end

function nondimensionalize(prob::OneBodyKepler, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return v, x
end

function dimensionalize(prob::OneBodyKepler, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return v, x
end

function compute_ddx!(prob::OneBodyKepler, ddx, dx, x)   
    ddx[:] = - x ./ norm(x)^3
    nothing
end

function compute_ddx(prob::OneBodyKepler, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return - x ./ norm(x)^3
end

function compute_K(prob::OneBodyKepler, v::AbstractArray{T, 1}) where T<:AbstractFloat
    return 0.5 * v' * v
end

function compute_U(prob::OneBodyKepler, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return - 1. / norm(x)
end


# function construct_z(v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
#     H = compute_H(p, q)
#     z = [sqrt(2. / sqrt(q'*q)); p] 
#     # z /= sqrt(-2. * H)
#     return z, H
# end

# function recover_canonical_vars(
#     z::AbstractArray{T, 1}, 
#     q_init::AbstractArray{T, 1}, 
#     H_init::T, 
#     L_init::T) where T<:AbstractFloat
#     # p = z[2:end] * sqrt(-2. * H_init)
#     # radius = 1. / (- H_init * z[1]^2)
#     p = z[2:end]
#     radius = 2. / (z[1]^2)
#     # q = q_init / sqrt(q_init'*q_init) * radius
#     p_sq = p'*p
#     fac = maximum([0., p_sq * radius^2 - L_init^2])
#     fac = sqrt(fac) 
#     if fac == 0.
#         println("fac:", fac, " radius:", radius)
#     end
#     q_candidate1 = (L_init * [p[2]; -p[1]] + fac * p) / p_sq
#     q_candidate2 = (L_init * [p[2]; -p[1]] - fac * p) / p_sq
#     if q_init'*q_candidate1 < q_init'*q_candidate2
#         q = q_candidate2
#     else
#         q = q_candidate1
#     end
#     return p, q
# end


# Lambda = (p, q) -> construct_z(p, q)[1]
# Lambda = (p, q) -> [p; q]
# Lambda = (p, q) -> [p/sqrt(p'*p); q/sqrt(q'*q)]

# Theta = function (p, q, Omega)
#     z, E = construct_z(p, q)
#     return recover_canonical_vars(Omega(z), q, E, q[1]*p[2]-q[2]*p[1])
# end
# Theta = function (p, q, Omega)
#     dim = length(p)
#     p_norm = sqrt(p'*p)
#     q_norm = sqrt(q'*q)
#     znew = Omega([p; q])
#     # znew = Omega([p/p_norm; q/q_norm])
#     pnew, qnew = znew[1:dim], znew[dim+1:end]
#     # pnew *= p_norm
#     # qnew *= q_norm
#     return pnew, qnew
# end