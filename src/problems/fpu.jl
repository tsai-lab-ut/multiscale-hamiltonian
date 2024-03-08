"""
Fermi-Pasta-Ulam Problem 
"""

using StaticArrays
using NL2sol
# #using NonlinearSolve

Base.@kwdef struct FPU <: SeparableHamiltonianSystem
    omega::Float64      # frequency of stiff springs  
    m::Int64 = 3        # number of stiff springs
    c0::Float64 = 0.5 * omega^2
end

mass(prob::FPU) = ones(2*prob.m) 

function initial_condition(prob::FPU, T::Type)
    omega = convert(T, prob.omega)
    sqrt2 = sqrt(convert(T, 2.0))

    q0 = zeros(T, 2*prob.m)
    p0 = zeros(T, 2*prob.m)
    q0[1] = (1 - 1/omega)/sqrt2
    q0[2] = (1 + 1/omega)/sqrt2
    p0[2] = sqrt2

    return p0, q0
end 

function nondimensionalize(prob::FPU, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return v, x
end

function dimensionalize(prob::FPU, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return v, x
end

function compute_ddx!(prob::FPU, ddx, dx, x)     
    
    ddx[1] = prob.c0 * (x[2] - x[1]) - 4 * x[1].^3
    ddx[2] = - prob.c0 * (x[2] - x[1]) + 4 * (x[3] - x[2]).^3
    ddx[3] = prob.c0 * (x[4] - x[3]) - 4 * (x[3] - x[2]).^3 
    ddx[4] = - prob.c0 * (x[4] - x[3]) + 4 * (x[5] - x[4]).^3
    ddx[5] = prob.c0 * (x[6] - x[5]) - 4 * (x[5] - x[4]).^3 
    ddx[6] = - prob.c0 * (x[6] - x[5]) + 4 * (- x[6]).^3
    
    nothing 
end

function compute_ddx_static(prob::FPU, dx, x, params, t) 
    
    ddx1 = prob.c0 * (x[2] - x[1]) - 4 * x[1].^3
    ddx2 = - prob.c0 * (x[2] - x[1]) + 4 * (x[3] - x[2]).^3
    ddx3 = prob.c0 * (x[4] - x[3]) - 4 * (x[3] - x[2]).^3 
    ddx4 = - prob.c0 * (x[4] - x[3]) + 4 * (x[5] - x[4]).^3
    ddx5 = prob.c0 * (x[6] - x[5]) - 4 * (x[5] - x[4]).^3 
    ddx6 = - prob.c0 * (x[6] - x[5]) + 4 * (- x[6]).^3
    
    SA[ddx1, ddx2, ddx3, ddx4, ddx5, ddx6]
end

function compute_ddx(prob::FPU, x::AbstractArray{T, 1}) where T<:AbstractFloat
    
    dx_stiff = x[2:2:end] - x[1:2:end]
    x_pad = vcat([0], x, [0]) 
    dx_soft = x_pad[2:2:end] - x_pad[1:2:end]
    
    a_r = - prob.c0 * dx_stiff + 4 * dx_soft[2:end].^3
    a_l = prob.c0 * dx_stiff - 4 * dx_soft[1:end-1].^3
    
    ddx = zero(x)
    ddx[1:2:end] = a_l
    ddx[2:2:end] = a_r
    
    return ddx
end

function compute_K(prob::FPU, v::AbstractArray{T, 1}) where T<:AbstractFloat
    return 0.5 * v' * v
end

function compute_U(prob::FPU, x::AbstractArray{T, 1}) where T<:AbstractFloat
    omega = convert(T, prob.omega)

    dx_stiff = x[2:2:end] - x[1:2:end]
    x_pad = vcat([0], x, [0]) 
    dx_soft = x_pad[2:2:end] - x_pad[1:2:end]
    
    U = 0.25 * omega^2 * sum(dx_stiff.^2) + sum(dx_soft.^4) 
    return U
end

"""Compute energy of stiff springs."""
function compute_I(prob::FPU, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    omega = convert(T, prob.omega)

    dx_stiff = x[2:2:end] - x[1:2:end]
    dv_stiff = v[2:2:end] - v[1:2:end]
    
    return 0.25 * dv_stiff.^2 + 0.25 * omega^2 * dx_stiff.^2
end

function construct_z(prob::FPU, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    m = prob.m

    z = zeros(T, 4*m+1)
    z[1:2*m] = v
    z[2*m+1:end] = compute_diffx(prob, x)    
    
    return z        
end

function recover_v_x(prob::FPU, z::AbstractArray{T, 1}, x_guess::AbstractArray{T, 1}) where T<:AbstractFloat
    m = prob.m

    v = z[1:2*m]
    diffx = z[2*m+1:end]
    
    if T != Float64
        diffx = convert.(Float64, diffx)
        x_guess = convert.(Float64, x_guess)
    end 
    
    function residual(x, r)
        r[:] = compute_diffx(prob, x) .- diffx
        return r
    end

    function jacobian(x, jac)
        jac[:, :] = Jac_diffx(prob, x)
        return jac
    end

    res = nl2sol(residual, jacobian, x_guess, 2*m+1; quiet=true)
    x = res.minimum

    #f(u, p) = compute_dq(u) - dq
    #jac(u, p) = Jac_dq(u)
    #prob = NonlinearProblem(NonlinearFunction(f; jac=jac), q_guess, nothing)
    #sol = solve(prob, NewtonRaphson())
    #q = sol.u
    # println("residual:", sol.resid)

    if T != Float64
        x = convert.(T, x)
    end

    return v, x
end

correct_phase(prob::FPU, v, x, corrector) = recover_v_x(prob, corrector(construct_z(prob, v, x)), x)

function compute_diffx(prob::FPU, x::AbstractArray{T, 1}) where T<:AbstractFloat
    m = prob.m
    omega = convert(T, prob. omega)
    sqrt2 = sqrt(convert(T, 2.0))

    diffx = zeros(T, 2*m+1)

    dx_stiff = x[2:2:end] - x[1:2:end]
    x_pad = vcat([0], x, [0])
    dx_soft = x_pad[2:2:end] - x_pad[1:2:end]
    
    diffx[1:m] = omega / sqrt2 * dx_stiff
    diffx[m+1:end] = sqrt2 * dx_soft.^2
    
    return diffx
end 

function Jac_diffx(prob::FPU, x::AbstractArray{T, 1}) where T<:AbstractFloat
    m = prob.m
    omega = convert(T, prob.omega)
    sqrt2 = sqrt(convert(T, 2.0))

    J = zeros(T, (2*m+1, 2*m))
    
    x_pad = vcat([0], x, [0]) 
    dx_soft = x_pad[2:2:end] - x_pad[1:2:end]
    
    for i in 1:m
        J[i, 2*i-1] = - omega / sqrt2
        J[i, 2*i] = omega / sqrt2
    end 
    
    for i in 2:m
        J[m+i, 2*i-2] = - 2 * sqrt2 * dx_soft[i]
        J[m+i, 2*i-1] = 2 * sqrt2 * dx_soft[i]
    end
    
    J[m+1, 1] = 2 * sqrt2 * dx_soft[1]
    J[end, end] = - 2 * sqrt2 * dx_soft[end]
    
    return J
end
