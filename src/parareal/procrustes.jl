using GenericLinearAlgebra, Statistics

"""Procrustes analysis"""

struct PA{T<:AbstractFloat}
    Omega::AbstractArray{T, 2}  # orthogonal matrix for rotation
    scale::T  # scaling factor 
end

function PA(A::AbstractArray{T, 2}, B::AbstractArray{T, 2}, use_scaling::Bool) where T<:AbstractFloat

    # estimate rotation 
    M = A * B'
    sol = GenericLinearAlgebra.svd(M)
    Omega = sol.U * sol.Vt

    # estimate scaling 
    scale = use_scaling ? sum(A .* (Omega * B)) / sum(B .* B) : T(1.0)

    @show Omega
    @show scale

    return PA(Omega, scale)
end

(pa::PA)(z) = pa.Omega * z * pa.scale


"""Procrustes analysis in hyperbolic space (Tabaghi and Dokmanic, 2021)"""

struct PAH{T<:AbstractFloat}
    R_V::AbstractArray{T, 2}  # hyperbolic rotation matrix
    R_mA::AbstractArray{T, 2}  # hyperbolic translation matrix for counter centering A
    R_neg_mB::AbstractArray{T, 2}  # hyperbolic translation matrix for centering B
end

average(data::AbstractArray{T, 2}) where T<:AbstractFloat = vec(mean(data, dims=2))

function lorentzian_inner_product(x::AbstractArray{T, 1}, y::AbstractArray{T, 1}) where T<:AbstractFloat
    return - x[1]*y[1] + sum(x[2:end].*y[2:end])
end

function hyperbolic_translation_matrix(b::AbstractArray{T, 1}) where T<:AbstractFloat 
    return [sqrt(1. + b'*b) b'; b sqrt(GenericLinearAlgebra.I + b*b')]
end

function hyperbolic_rotation_matrix(U::AbstractArray{T, 2}) where T<:AbstractFloat
    return cat(1., U, dims=(1, 2))
end

function geodesic_distance(x::AbstractArray{T, 1}, y::AbstractArray{T, 1}) where T<:AbstractFloat
    return acosh(-lorentzian_inner_product(x, y))
end

function PAH(A::AbstractArray{T, 2}, B::AbstractArray{T, 2}) where T<:AbstractFloat

    # check whether data lie on unit hyperboloid
    A_inner = mapslices(z -> lorentzian_inner_product(z, z), A, dims=1)
    B_inner = mapslices(z -> lorentzian_inner_product(z, z), B, dims=1)
    println("min:", minimum(A_inner), " max:", maximum(A_inner), " mean:", mean(A_inner))
    println("min:", minimum(B_inner), " max:", maximum(B_inner), " mean:", mean(B_inner))
    
    # centering data
    mA = average(A[2:end, :])
    mA /= sqrt(-lorentzian_inner_product(average(A), average(A)))

    mB = average(B[2:end, :])
    mB /= sqrt(-lorentzian_inner_product(average(B), average(B)))

    A_centered = hyperbolic_translation_matrix(-mA) * A
    B_centered = hyperbolic_translation_matrix(-mB) * B

    # estimate rotation
    M = A_centered[2:end, :] * B_centered[2:end, :]'
    sol = GenericLinearAlgebra.svd(M)
    V = sol.U * sol.Vt

    println("V:", V)
    println("mA:", mA)
    println("mB:", mB)

    B_aligned = hyperbolic_translation_matrix(mA) * hyperbolic_rotation_matrix(V) * B_centered
    # discrepancy = mean(geodesic_distance(A[:, i], B_aligned[:, i])^2 for i in 1:size(A, 2))
    discrepancy = mean((-lorentzian_inner_product(A[:, i], B_aligned[:, i]))^2 for i in 1:size(A, 2))
    println("discrepancy:", discrepancy)

    return PAH(hyperbolic_rotation_matrix(V), hyperbolic_translation_matrix(mA), hyperbolic_translation_matrix(-mB))
end

(pah::PAH)(z) = pah.R_mA * pah.R_V * pah.R_neg_mB * z


"""Hyperbolic Procrustes analysis (Lin et al., 2021)"""

struct HPA{T<:AbstractFloat}
    mA_frechet::AbstractArray{T, 1}
    mB_frechet::AbstractArray{T, 1}
    s::T
    Omega::AbstractArray{T, 2}
    A_proj_mean::AbstractArray{T, 1}
    B_proj_mean::AbstractArray{T, 1}
end

function exp_map(v::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    v_L_norm = sqrt(lorentzian_inner_product(v, v))
    return cosh(v_L_norm) * x + sinh(v_L_norm) * v / v_L_norm
end

function log_map(y::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    lambda = -lorentzian_inner_product(y, x);
    return acosh(lambda) / sqrt(lambda^2 - 1.) * (y - lambda * x)
end

function PT_x_to_y(v::AbstractArray{T, 1}, x::AbstractArray{T, 1}, y::AbstractArray{T, 1}) where T<:AbstractFloat
    lambda = -lorentzian_inner_product(y, x);
    return v + lorentzian_inner_product(y - lambda * x, v) / (lambda + 1.) * (x + y)
end

function geodesic_path(t::T, x::AbstractArray{T, 1}, v::AbstractArray{T, 1}) where T<:AbstractFloat
    v_L_norm = sqrt(lorentzian_inner_product(v, v))
    return cosh(v_L_norm * t) * x + sinh(v_L_norm * t) * v / v_L_norm
end

function Riemannian_translation_x_to_y(z::AbstractArray{T, 1}, x::AbstractArray{T, 1}, y::AbstractArray{T, 1}) where T<:AbstractFloat
    return exp_map(PT_x_to_y(log_map(z, x), x, y), y)
end

function Riemannian_scaling(s::T, y::AbstractArray{T, 1}, x::AbstractArray{T, 1}) where T<:AbstractFloat
    return geodesic_path(s, x, log_map(y, x))
end

function HPA(A::AbstractArray{T, 2}, B::AbstractArray{T, 2}) where T<:AbstractFloat
    
    # check whether data lie on unit hyperboloid
    A_inner = mapslices(z -> lorentzian_inner_product(z, z), A, dims=1)
    B_inner = mapslices(z -> lorentzian_inner_product(z, z), B, dims=1)
    println("min:", minimum(A_inner), " max:", maximum(A_inner), " mean:", mean(A_inner))
    println("min:", minimum(B_inner), " max:", maximum(B_inner), " mean:", mean(B_inner))

    # compute Riemannian mean and dispersion for each set
    # mA_frechet, dA_frechet = compute_frechet_mean(A)
    # mB_frechet, dB_frechet = compute_frechet_mean(B)
    mA_frechet = [1; zeros(size(A, 1)-1)]
    dA_frechet = 1.0
    mB_frechet = [1; zeros(size(A, 1)-1)]
    dB_frechet = 1.0
    s = sqrt(dA_frechet / dB_frechet)

    # apply Riemannian translation: exp_map(PT_x_to_y(log_map(z, mB_frechet), mB_frechet, mA_frechet), mA_frechet)
    # B = mapslices(z -> Riemannian_translation_x_to_y(z, mB_frechet, mA_frechet), B, dims=1)
    
    # apply Riemannian scaling: geodesic_path(s, mA_frechet, log_map(z, mA_frechet))
    # B = mapslices(z -> Riemannian_scaling(s, z, mA_frechet), B, dims=1)
    
    # estimate wrapped rotation 
    A_tan = mapslices(z -> log_map(z, mA_frechet), A, dims=1)
    B_tan = mapslices(z -> s * PT_x_to_y(log_map(z, mB_frechet), mB_frechet, mA_frechet), B, dims=1)
    
    A_proj = A_tan[2:end, :]
    B_proj = B_tan[2:end, :]
    
    A_proj_mean = average(A_proj)
    B_proj_mean = average(B_proj)
    
    A_proj_centered = A_proj .- A_proj_mean
    B_proj_centered = B_proj .- B_proj_mean
    
    M = A_proj_centered * B_proj_centered'
    sol = GenericLinearAlgebra.svd(M)
    Omega = sol.U * sol.Vt
    
    println("mA_frechet:", mA_frechet)
    println("mB_frechet:", mB_frechet)
    println("dA_frechet:", dA_frechet)
    println("dB_frechet:", dB_frechet)
    println("Omega:", Omega)
    println("A_proj_mean:", A_proj_mean)
    println("B_proj_mean:", B_proj_mean)

    B_aligned_proj = Omega * B_proj_centered .+ B_proj_mean
    B_aligned = mapslices(x -> exp_map([x' * mA_frechet[2:end] / mA_frechet[1]; x], mA_frechet), B_aligned_proj, dims=1)
    # discrepancy = mean(geodesic_distance(A[:, i], B_aligned[:, i])^2 for i in 1:size(A, 2))
    discrepancy = mean((-lorentzian_inner_product(A[:, i], B_aligned[:, i]))^2 for i in 1:size(A, 2))
    println("discrepancy:", discrepancy)

    HPA(mA_frechet, mB_frechet, s, Omega, A_proj_mean, B_proj_mean)
end

function (hpa::HPA)(z)

    # apply translation + scaling + log + projection
    z_tan = hpa.s * PT_x_to_y(log_map(z, hpa.mB_frechet), hpa.mB_frechet, hpa.mA_frechet)
    z_proj = z_tan[2:end]

    # apply wrapped rotation 
    znew_proj = hpa.Omega * (z_proj - hpa.B_proj_mean) + hpa.B_proj_mean
    
    # apply inverse projection + exp
    znew_tan = [znew_proj' * hpa.mA_frechet[2:end] / hpa.mA_frechet[1]; znew_proj]
    znew = exp_map(znew_tan, hpa.mA_frechet)

    return znew
end

function compute_frechet_mean(X::AbstractArray{T, 2}; niters::Integer=7) where T<:AbstractFloat
    
    # initialization
    mX = X[:, 1]
    k = 0

    while k < niters
        k += 1

        # update mean 
        neg_products = - X' * [-mX[1]; mX[2:end]]

        # println(neg_products[neg_products .< 1.])
        # println(X[:, neg_products .< 1.])
        neg_products[(neg_products .< 1.) .&  (1-1e-14 .< neg_products)] .= 1.  # to avoid domain error of acosh
        
        weights = 2. * acosh.(neg_products) ./ (sqrt.(neg_products.^2 .- 1.) .+ 1e-14)
        mX = vec(sum(X .* weights', dims=2)) 
        mX /= sqrt(-lorentzian_inner_product(mX, mX))

        println("mX:", mX)
    end

    dX = mean(acosh.(- X' * [-mX[1]; mX[2:end]]) .^2)

    return mX, dX
end


"""Procrustes analysis in p-space and q-space separately"""

struct DoublePA{T<:AbstractFloat}
    Omega_p::AbstractArray{T, 2}  
    Omega_q::AbstractArray{T, 2}
    dim::Integer
end

function DoublePA(A::AbstractArray{T, 2}, B::AbstractArray{T, 2}) where T<:AbstractFloat

    dim = Int(size(A, 1) // 2)
    A_p, A_q = A[1:dim, :], A[dim+1:end, :]
    B_p, B_q = B[1:dim, :], B[dim+1:end, :]

    # estimate rotation 
    M_p = A_p * B_p'
    sol = GenericLinearAlgebra.svd(M_p)
    Omega_p = sol.U * sol.Vt

    M_q = A_q * B_q'
    sol = GenericLinearAlgebra.svd(M_q)
    Omega_q = sol.U * sol.Vt
    
    println("Omega_p:", Omega_p)
    println("Omega_q:", Omega_q)

    return DoublePA(Omega_p, Omega_q, dim)
end

(dpa::DoublePA)(z) = [dpa.Omega_p * z[1:dpa.dim]; dpa.Omega_q * z[dpa.dim+1:end]]

