"""
N-body problem (the outer solar system)
"""

Base.@kwdef struct NBody <: SeparableHamiltonianSystem
    G::Float64 = 2.95912208286e-4
    m0::Float64 = 1.00000597682
    m1::Float64 = 0.000954786104043
    m2::Float64 = 0.000285583733151
    m3::Float64 = 0.0000437273164546
    m4::Float64 = 0.0000517759138449
    m5::Float64 = 1. / (1.3e8)
    MASS::Vector{Float64} = [m0, m1, m2, m3, m4, m5]
end

function initial_condition(prob::NBody, T::Type)
    x0 = zeros(T, 18)
    v0 = zeros(T, 18)

    x0[4:6] =   [-3.5023653,      -3.8169847,     -1.5507963    ]
    x0[7:9] =   [9.0755314,       -3.0458353,     -1.6483708    ]
    x0[10:12] = [8.3101420,       -16.2901086,    -7.2521278    ]
    x0[13:15] = [11.4707666,      -25.7294829,    -10.8169456   ]
    x0[16:18] = [-15.5387357,     -25.2225594,    -3.1902382    ]

    v0[4:6] =   [0.00565429,      -0.00412490,    -0.00190589   ]
    v0[7:9] =   [0.00168318,      0.00483525,     0.00192462    ]
    v0[10:12] = [0.00354178,      0.00137102,     0.00055029    ]
    v0[13:15] = [0.00288930,      0.00114527,     0.00039677    ]
    v0[16:18] = [0.00276725,      -0.00170702,    -0.00136504   ]

    return v0, x0
end 

function nondimensionalize(prob::NBody, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return v, x
end

function dimensionalize(prob::NBody, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return v, x
end

function compute_ddx!(prob::NBody, ddx, dx, x)   
    for i in 0:5
        xi = @view x[3*i+1:3*i+3]
        ddx[3*i+1:3*i+3] = zeros(3)
        for j in 0:5
            if j != i
                xj = @view x[3*j+1:3*j+3]
                ddx[3*i+1:3*i+3] += prob.MASS[j+1] * (xi-xj) ./ norm(xi-xj)^3
            end
        end
        ddx[3*i+1:3*i+3] *= -prob.G
    end
    nothing 
end

function compute_ddx(prob::NBody, x::AbstractArray{T, 1}) where T<:AbstractFloat        
    ddx = zero(x)

    for i in 0:5
        xi = x[3*i+1:3*i+3]
        ddxi = zeros(3)
        for j in 0:5
            if j != i
                xj = x[3*j+1:3*j+3]
                ddxi += prob.MASS[j+1] * (xi-xj) ./ norm(xi-xj)^3
            end
        end
        ddxi *= -prob.G
        ddx[3*i+1:3*i+3] = ddxi
    end

    return ddx
end

function compute_K(prob::NBody, v::AbstractArray{T, 1}) where T<:AbstractFloat
    return 0.5 * v' * (v .* reshape(repeat(prob.MASS, 1, 3)', length(v)))
end

function compute_U(prob::NBody, x::AbstractArray{T, 1}) where T<:AbstractFloat
    U = 0.
    for i in 1:5
        xi = x[3*i+1:3*i+3]
        for j in 0:(i-1)
            xj = x[3*j+1:3*j+3]
            U += prob.MASS[i+1] * prob.MASS[j+1] / norm(xi-xj)
        end
    end
    U *= -prob.G
    return U
end
