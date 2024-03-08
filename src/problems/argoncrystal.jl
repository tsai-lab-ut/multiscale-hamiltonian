"""
Argon crystal problem
"""

Base.@kwdef struct ArgonCrystal <: SeparableHamiltonianSystem 
    Natoms::Int64 = 7
    d::Int64 = 2
    MASS::Float64 = 66.34e-27               # [kg]
    SIGMA::Float64 = 0.341                  # [nm]
    kB::Float64 = 1.380658e-23              # [J / K] = [kg * nm^2 / (ns^2 * K)]
    EPSILON_div_kB::Float64 = 119.8         # [K]
    EPSILON::Float64 = EPSILON_div_kB * kB  # [J]
    MASS_div_kB::Float64 = MASS/kB          # [K * ns^2 / nm^2]
    C0::Float64 = sqrt(EPSILON/MASS)        # [nm / ns]
    C1::Float64 = 24EPSILON/(MASS*SIGMA)    # [nm / ns^2]
end


# const twoSIGMA12 = 2SIGMA^(12);
# const SIGMA6 = SIGMA^(6);
# const twentyfourEPSILONdivbyMASS = 24EPSILON/MASS;
# const halfMASSdivbyNatomsdivbykB = 0.5MASS/(Natoms*kB);

# const Lengthz = Int(2Natoms + Natoms*(Natoms-1)/2);
# const twoSqrtEPSILON = 2 * sqrt(EPSILON);
# const SqrtHalfMASS = sqrt(MASS/2);

mass(prob::ArgonCrystal) = ones(prob.Natoms*prob.d) * prob.MASS_div_kB 

function initial_condition(prob::ArgonCrystal, T::Type)
    x0 = vec([0.0 0.0 0.02 0.39 0.34 0.17 0.36 -0.21 -0.02 -0.4 -0.35 -0.16 -0.31 0.21]);
    v0 = vec([-30.0 -20.0 50.0 -90.0 -70.0 -60.0 90.0 40.0 80.0 90.0 -40.0 100.0 -80.0 -60.0]);  # H0 = -1260 kB
    # v0 = vec([-130.0 -20.0 150.0 -90.0 -70.0 -60.0 90.0 40.0 80.0 90.0 -40.0 100.0 -80.0 -60.0]);  # H0 = -1174 kB
    # v0 = vec([0.0 -20.0 20.0 -90.0 -50.0 -60.0 70.0 40.0 80.0 90.0 -40.0 20.0 -80.0 20.0]); # H0 = -1312 kB

    x0 = convert.(T, x0)
    v0 = convert.(T, v0)

    return v0, x0
end 

function nondimensionalize(prob::ArgonCrystal, v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return v / prob.C0, x / prob.SIGMA
end

function dimensionalize(prob::ArgonCrystal, v_nd::AbstractArray{T, 1}, x_nd::AbstractArray{T, 1}) where T<:AbstractFloat
    return v_nd * prob.C0, x_nd * prob.SIGMA
end

"""Compute distances between x1, x2, ..., xn."""
function distance_matrix(X::AbstractArray{T, 2}) where T<:AbstractFloat
    n = size(X, 2)         
    return [norm(X[:, i]-X[:, j]) for i in 1:n, j in 1:n]
end

"""Compute Lennard-Jones potential energy between 2 particles r distance apart (divided by kB)."""
LJ_potential(prob, r) = 4*prob.EPSILON_div_kB * ((prob.SIGMA/r)^12 - (prob.SIGMA/r)^6)

function compute_ddx!(prob::ArgonCrystal, ddx, dx, x)   
    d = prob.d  
    for i in 1:prob.Natoms
        xi = @view x[d*i-d+1:d*i]
        ddx[d*i-d+1:d*i] .= 0.
        for j in 1:prob.Natoms
            if j != i
                xj = @view x[d*j-d+1:d*j]
                diff = (xi-xj)/prob.SIGMA
                rij = norm(diff)
                ddx[d*i-d+1:d*i] += (2. / rij^14 - 1. / rij^8) * diff
            end
        end
    end
    ddx[:] *= prob.C1
    nothing 
end

function compute_ddx(prob::ArgonCrystal, x::AbstractArray{T, 1}) where T<:AbstractFloat
    x = reshape(x, (prob.d, prob.Natoms))
    x = x / prob.SIGMA
    dist = distance_matrix(x)
    ddx = zero(x)
    
    for i in 1:prob.Natoms
        for j in 1:prob.Natoms
            if j != i
                rij = dist[i, j]
                ddx[:, i] += (2. / rij^14 - 1. / rij^8) * (x[:, i] - x[:, j])
            end
        end
    end
    ddx *= prob.C1
    
    return vec(ddx)
end

function compute_K(prob::ArgonCrystal, v::AbstractArray{T, 1}) where T<:AbstractFloat
    return 0.5 * prob.MASS_div_kB * v' * v
end

function compute_U(prob::ArgonCrystal, x::AbstractArray{T, 1}) where T<:AbstractFloat
    x = reshape(x, (prob.d, prob.Natoms))
    dist = distance_matrix(x)
    U = sum([LJ_potential(prob, dist[i, j]) for i in 1:prob.Natoms for j in 1:i-1])
    return U
end

# w(r::T) where T<:AbstractFloat = (SIGMA / r)^6


# function construct_z(v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
#     z = zeros(T, Lengthz)
#     z[1:2Natoms] = v * SqrtHalfMASS
#     x_reshaped = reshape(x, (d, Natoms))
#     dist = distance_matrix(x_reshaped)
#     z[2Natoms+1:end] = [twoSqrtEPSILON * (w(dist[i, j])-0.5) for i in 1:Natoms for j in 1:i-1]
#     return z
# end


# function vec2symmat(vec::Array{T, 1}, n::Integer) where T<:AbstractFloat
#     n*(n-1)/2 == length(vec) || error("length of vector is not valid")
    
#     mat = zeros(T, n, n)
#     k = 1 
    
#     for i in 1:n
#         for j in 1:i-1
#             mat[i, j] = vec[k]
#             mat[j, i] = vec[k]
#             k += 1
#         end
#     end
    
#     return mat
# end


# function classicalMDS(D::Array{T, 2}, d::Integer, tol::T=1e-8) where T<:AbstractFloat
#     n = size(D, 1)
#     J = Matrix(1.0I, n, n) - ones(n) * ones(n)' / n
#     G = -0.5*J*D*J
#     G = (G + G')/2
#     sol = eigen(G, sortby = x -> -abs(x))
# #     print(sol.values)
#     if ~all(sol.values .> -tol)  # check if G is psd
#        error("G is not positive semidefinite.") 
#     end
#     X = (sol.vectors[:, 1:d] * Diagonal(sqrt.(sol.values[1:d])))'
#     return X
# end


# function align(X::AbstractArray{T, 2}, Y::AbstractArray{T, 2}) where T<:AbstractFloat
#     d = size(X, 1)
#     n = size(X, 2)
#     xc = sum(X, dims=2) / n
#     yc = sum(Y, dims=2) / n
#     Xc = X .- xc
#     Yc = Y .- yc
#     sol = svd(Xc * Yc')
#     R = (sol.U * sol.Vt)'
#     return R * Xc .+ yc
# end


# function recover_canonical_vars(z::Array{T, 1}) where T<:AbstractFloat

#     v = z[1:2Natoms] ./ SqrtHalfMASS
#     W = z[2Natoms+1:end] ./ twoSqrtEPSILON .+ 0.5    
#     dist_sq_vec = sign.(W).*(W./sign.(W)).^(-1/3) * SIGMA^2
#     D = vec2symmat(dist_sq_vec, Natoms)
#     x = classicalMDS(D, d)
# #     xhat = align(classicalMDS(D, d), reshape(x, (d, Natoms)))
#     x = vec(x)
#     return v, x
# end
