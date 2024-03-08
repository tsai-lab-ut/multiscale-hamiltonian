"""
Parareal algorithms.
"""

module Parareal

using Distributed
using Logging
using LinearAlgebra
using Optim
using ProgressMeter 
include("procrustes.jl")
include("interpolation.jl")
include("../utils/saving_utils.jl")


function save(output_dir, k, p, q)
    if ~isempty(output_dir)
        filepath = joinpath(output_dir, "k=$k/u.csv")
        @info "Saving iter $k solution at $filepath ..."
        save_csv(filepath, p, q)
    end
end


function save(output_dir, k, p, q, diagnostics)
    if ~isempty(output_dir)
        filepath = joinpath(output_dir, "k=$k/u.csv")
        filepath2 = joinpath(output_dir, "k=$k/diagnostics.csv")
        @info "Saving iter $k solution at $filepath ..."
        save_csv(filepath, p, q)
        @info "Saving iter $k diagnostics at $filepath2 ..."
        save_csv(filepath2, diagnostics)
    end
end


"Plain parareal algorithm."
function plain(
        p0::AbstractArray{T, 1},
        q0::AbstractArray{T, 1},
        fine_solve::Function,
        coarse_solve::Function,
        N::Integer,
        niters::Integer;
        output_dir::String="") where T<:AbstractFloat
    
    # get dimension d
    @assert length(p0) == length(q0)
    d = length(p0)
    
    # initialize arrays 
    p = zeros(T, d, N+1)
    q = zeros(T, d, N+1)
    pnew = zero(p)
    qnew = zero(q)
    p_all = zeros(T, d, N+1, niters+1)
    q_all = zeros(T, d, N+1, niters+1)

    # solve for solutions at iteration 0 
    @info "Starting iter 0 ..."
    elapsed_time = @elapsed begin
        p[:, 1] = p0
        q[:, 1] = q0
        @showprogress for n in 1:N
            p[:, n+1], q[:, n+1] = coarse_solve(p[:, n], q[:, n])
        end
    end
    @info "Done. Elapsed time = $elapsed_time seconds."
    
    p_all[:, :, 1] = p
    q_all[:, :, 1] = q

    save(output_dir, 0, p, q)

    # begin parareal iterations 
    for k in 1:niters
        @info "Starting iter $k ..."
        elapsed_time = @elapsed begin
            pnew[:, 1] = p0
            qnew[:, 1] = q0
            
            F = @showprogress pmap(fine_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))
            G = @showprogress pmap(coarse_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))
            
            @showprogress for n in 1:N
                pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n]) .- G[n] .+ F[n]
            end
        end
        @info "Done. Elapsed time = $elapsed_time seconds."

        p[:, :] = pnew[:, :]
        q[:, :] = qnew[:, :]
        p_all[:, :, k+1] = p
        q_all[:, :, k+1] = q

        save(output_dir, k, p, q)
    end
    return p_all, q_all
end


"Procrustes parareal algorithm."
function procrustes(
        p0::AbstractArray{T, 1},
        q0::AbstractArray{T, 1},
        fine_solve::Function,
        coarse_solve::Function,
        N::Integer,
        niters::Integer,
        Lambda::Function,
        Theta::Function;
        output_dir::String="",
        with_additive::Bool=true,
        use_scaling::Bool=false) where T<:AbstractFloat
    
    # get dimension d
    @assert length(p0) == length(q0)
    d = length(p0)
    
    # initialize arrays 
    p = zeros(T, d, N+1)
    q = zeros(T, d, N+1)
    pnew = zero(p)
    qnew = zero(q)
    p_all = zeros(T, d, N+1, niters+1)
    q_all = zeros(T, d, N+1, niters+1)
    
    # solve for solutions at iteration 0 
    @info "Starting iter 0 ..."
    elapsed_time = @elapsed begin
        p[:, 1] = p0
        q[:, 1] = q0
        @showprogress for n in 1:N
            p[:, n+1], q[:, n+1] = coarse_solve(p[:, n], q[:, n])
        end
    end
    @info "Done. Elapsed time = $elapsed_time seconds."
    
    p_all[:, :, 1] = p
    q_all[:, :, 1] = q

    save(output_dir, 0, p, q)
    
    # begin parareal iterations 
    for k in 1:niters
        @info "Starting iter $k ..."
        elapsed_time = @elapsed begin
            pnew[:, 1] = p0
            qnew[:, 1] = q0
            
            F = @showprogress pmap(fine_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))
            G = @showprogress pmap(coarse_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))
            
            Fh = hcat([Lambda(p, q) for (p, q) in F]...)
            Gh = hcat([Lambda(p, q) for (p, q) in G]...)
            
            # solve procrustes problem 
            Omega = PA(Fh, Gh, use_scaling)
            # Omega = PAH(Fh, Gh)
            # Omega = HPA(Fh, Gh)
            # Omega = DoublePA(Fh, Gh)

            if with_additive
                @showprogress for n in 1:N
                    pnew[:, n+1], qnew[:, n+1] = Theta(coarse_solve(pnew[:, n], qnew[:, n])..., Omega) .- Theta(G[n]..., Omega) .+ F[n] 
                end
            else
                @showprogress for n in 1:N
                    pnew[:, n+1], qnew[:, n+1] = Theta(coarse_solve(pnew[:, n], qnew[:, n])..., Omega)
                end
            end
        end
        @info "Done. Elapsed time = $elapsed_time seconds."

        p[:, :] = pnew[:, :]
        q[:, :] = qnew[:, :]
        p_all[:, :, k+1] = p
        q_all[:, :, k+1] = q

        save(output_dir, k, p, q)
    end
    return p_all, q_all
end


"Interpolation based theta parareal algorithm."
function interpolative(
        p0::AbstractArray{T, 1},
        q0::AbstractArray{T, 1},
        fine_solve::Function,
        coarse_solve::Function,
        N::Integer,
        niters::Integer;
        output_dir::String="",
        tol::T=1e-14,
        use_bias::Bool=true,
        centering::Bool=true,
        k_most_recent::Integer=-1,
        k_nearest::Integer=-1) where T<:AbstractFloat
    
    # get dimension d
    @assert length(p0) == length(q0)
    d = length(p0)
    
    # initialize arrays 
    p = zeros(T, d, N+1)
    q = zeros(T, d, N+1)
    pnew = zero(p)
    qnew = zero(q)
    p_all = zeros(T, d, N+1, niters+1)
    q_all = zeros(T, d, N+1, niters+1)

    # initialize diagnostics vector
    diagnostics = [Dict(
        "num_singular"=>zeros(Integer, N), 
        "condition_number"=>zeros(T, N),
        "interp_err"=>zeros(T, N),
        "is_exception"=>zeros(Bool, N),
        "range_space_projection_ratio"=>zeros(T, N)) for k in 1:niters]

    # solve for solutions at iteration 0 
    @info "Starting iter 0 ..."
    elapsed_time = @elapsed begin
        p[:, 1] = p0
        q[:, 1] = q0
        @showprogress for n in 1:N
            p[:, n+1], q[:, n+1] = coarse_solve(p[:, n], q[:, n])
        end
    end
    @info "Done. Elapsed time = $elapsed_time seconds."
    
    p_all[:, :, 1] = p
    q_all[:, :, 1] = q
    save(output_dir, 0, p, q)

    # initialize W_n for n = 1, ..., N
    W = zeros(T, 2*d, 2*d+1, N)
    for i in 1:(2*d+1)
        W[:, i, :] = [p[:, 1:N]; q[:, 1:N]]
    end
    
    if niters == 0
        return p_all, q_all, diagnostics
    end 

    ### for k = 1
    @info "Starting iter 1 ..."
    elapsed_time = @elapsed begin
        pnew[:, 1] = p0
        qnew[:, 1] = q0

        F = @showprogress pmap(fine_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))
        G = @showprogress pmap(coarse_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))

        @showprogress for n in 1:N
            pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n]) .- G[n] .+ F[n] 
        end
    end
    @info "Done. Elapsed time = $elapsed_time seconds."
    
    p[:, :] = pnew[:, :]
    q[:, :] = qnew[:, :]
    p_all[:, :, 2] = p
    q_all[:, :, 2] = q

    # update W 
    W[:, 2:end, :] = W[:, 1:end-1, :]
    W[:, 1, :] = [p[:, 1:N]; q[:, 1:N]]

    # initialize K_n for n = 1, ..., N
    K = zeros(T, 2*d, 2*d+1, N)
    for n in 1:N
        dp, dq = F[n] .- G[n]
        for i in 1:(2*d+1)
            K[:, i, n] = [dp; dq]
        end
    end

    # record num_singular_vals(W_n) and condition_number(W_n) for n = 1, ..., N
    diagnostics[1]["num_singular"][:] .= 1
    diagnostics[1]["condition_number"][:] .= 1.
    diagnostics[1]["interp_err"][:] .= 0.
    diagnostics[1]["is_exception"][:] .= false
    diagnostics[1]["range_space_projection_ratio"][:] .= 0.

    save(output_dir, 1, p, q, diagnostics[1])

    ### for k >= 2 
    for k in 2:niters
        @info "Starting iter $k ..."
        elapsed_time = @elapsed begin
            pnew[:, 1] = p0
            qnew[:, 1] = q0
            
            F = @showprogress pmap(fine_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))
            G = @showprogress pmap(coarse_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))
            
            # update K 
            K[:, 2:end, :] = K[:, 1:end-1, :]
            for n in 1:N
                dp, dq = F[n] .- G[n]
                K[:, 1, n] = [dp; dq]
            end
            
            # linear_maps = @showprogress pmap((X, Y) -> Linear(X, Y, true, tol), eachslice(W, dims=3), eachslice(K, dims=3))
            @showprogress for n in 1:N
                # perform linear interpolation
                if k_most_recent > 0  # TODO: parallelize 
                    X = W[:, 1:k_most_recent, n]
                    Y = K[:, 1:k_most_recent, n]
                elseif k_nearest > 0
                    dist = [norm(W[:, i, n] - [pnew[:, n]; qnew[:, n]]) for i in 1:(2*d+1)]
                    idx = sortperm(dist)[1:k_nearest]
                    @show idx
                    X = W[:, idx, n]
                    Y = K[:, idx, n]
                else  # TODO: parallelize 
                    X = W[:, :, n]
                    Y = K[:, :, n]
                end 
                linear = centering ? Linear(X, Y, W[:, 1, n], K[:, 1, n], use_bias, tol) : Linear(X, Y, use_bias, tol)

                # record num_singular_vals(W_n) and condition_number(W_n) 
                diagnostics[k]["num_singular"][n] = linear.rank
                diagnostics[k]["condition_number"][n] = linear.condition_number

                if linear.rank == 1 
                    pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n]) .- G[n] .+ F[n]
                else
                    corr = linear.A * ([pnew[:, n]; qnew[:, n]] - [p[:, n]; q[:, n]])
                    # temp1 = linear([pnew[:, n]; qnew[:, n]]) - linear([p[:, n]; q[:, n]])
                    # temp2 = linear.A * ([pnew[:, n]; qnew[:, n]] - [p[:, n]; q[:, n]])
                    # println("norm:", norm(temp1 - temp2))
                    diagnostics[k]["interp_err"][n] = norm(K[:, 1, n] - linear([p[:, n]; q[:, n]]))
                    diagnostics[k]["range_space_projection_ratio"][n] = range_space_projection_ratio([pnew[:, n]; qnew[:, n]], linear)

                    if norm(corr) > 2 * maximum(norm.(eachcol(K[:, :, n])))  # TODO: modify condition for exception 
                        diagnostics[k]["is_exception"][n] = true
                        pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n]) .- G[n] .+ F[n]
                    else
                        pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n]) .- G[n] .+ F[n]
                        pnew[:, n+1] += corr[1:d]
                        qnew[:, n+1] += corr[d+1:2*d]
                    end
                end 
            end
        end
        @info "Done. Elapsed time = $elapsed_time seconds."

        p[:, :] = pnew[:, :]
        q[:, :] = qnew[:, :]
        p_all[:, :, k+1] = p
        q_all[:, :, k+1] = q

        # update W
        W[:, 2:end, :] = W[:, 1:end-1, :]
        W[:, 1, :] = [p[:, 1:N]; q[:, 1:N]]
        
        save(output_dir, k, p, q, diagnostics[k])
    end
    
    return p_all, q_all, diagnostics
end


# "Interpolation based theta parareal algorithm"
# function interpolative2(
#         p0::AbstractArray{T, 1},
#         q0::AbstractArray{T, 1},
#         t_grid::AbstractArray{Float64, 1},
#         fine_solve::Function,
#         coarse_solve::Function;
#         niters::Integer=3,
#         tol::T=1e-14) where T<:AbstractFloat
    
#     # get dimension d
#     @assert length(p0) == length(q0)
#     d = length(p0)
    
#     # initialize arrays 
#     p = zeros(T, d, N+1)
#     q = zeros(T, d, N+1)
#     pnew = zero(p)
#     qnew = zero(q)
#     p_all = zeros(T, d, N+1, niters+1)
#     q_all = zeros(T, d, N+1, niters+1)

#     # solve for solutions at iteration 0 
#     p[:, 1] = p0
#     q[:, 1] = q0
#     @showprogress for n in 1:N
#         p[:, n+1], q[:, n+1] = coarse_solve(p[:, n], q[:, n])
#     end
#     p_all[:, :, 1] = p
#     q_all[:, :, 1] = q

#     # initialize diagnostics vector
#     diagnostics = [Dict(
#         "num_singular"=>zeros(Integer, N), 
#         "condition_number"=>zeros(T, N),
#         "interp_err"=>zeros(T, N),
#         "is_exception"=>zeros(Bool, N),
#         "range_space_projection_ratio"=>zeros(T, N)) for k in 1:niters]

#     if niters == 0
#         return p_all, q_all
#     end 
    
#     ### for k = 1
#     println("iter 1")

#     # record num_singular_vals(W_n) and condition_number(W_n) for n = 1, ..., N
#     diagnostics[1]["num_singular"][:] .= 1
#     diagnostics[1]["condition_number"][:] .= 1.
#     diagnostics[1]["interp_err"][:] .= 0.
#     diagnostics[1]["is_exception"][:] .= false
#     diagnostics[1]["range_space_projection_ratio"][:] .= 0.

#     pnew[:, 1] = p0
#     qnew[:, 1] = q0

#     F = @showprogress pmap(fine_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))
#     G = @showprogress pmap(coarse_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))

#     # initialize X_n, Y_n for n = 1, ..., N
#     X = zeros(T, 2*d, 2*d+1, N)
#     Y = zeros(T, 2*d, 2*d+1, N)
#     for n in 1:N
#         Fu_p, Fu_q = F[n]
#         Cu_p, Cu_q = G[n]
#         for i in 1:(2*d+1)
#             X[:, i, n] = [Cu_p; Cu_q]
#             Y[:, i, n] = [Fu_p; Fu_q]
#         end
#     end

#     @showprogress for n in 1:N
#         pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n]) .- G[n] .+ F[n] 
#     end

#     p[:, :] = pnew[:, :]
#     q[:, :] = qnew[:, :]

#     p_all[:, :, 2] = p
#     q_all[:, :, 2] = q

#     ### for k >= 2 
#     for k in 2:niters
        
#         println("iter ", k)
        
#         pnew[:, 1] = p0
#         qnew[:, 1] = q0
         
#         F = @showprogress pmap(fine_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))
#         G = pmap(coarse_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2))
        
#         # update X, Y
#         X[:, 2:end, :] = X[:, 1:end-1, :]
#         Y[:, 2:end, :] = Y[:, 1:end-1, :]
#         for n in 1:N
#             Fu_p, Fu_q = F[n]
#             Cu_p, Cu_q = G[n]
#             X[:, 1, n] = [Cu_p; Cu_q]
#             Y[:, 1, n] = [Fu_p; Fu_q]
#         end
        
#         # linear_maps = @showprogress pmap((X, Y) -> Linear(X, Y, true, tol), eachslice(W, dims=3), eachslice(K, dims=3))

#         @showprogress for n in 1:N            
#             # dist = [norm(W[:, i, n] - [pnew[:, n]; qnew[:, n]]) for i in 1:(2*d+1)]
#             # idx = sortperm(dist)[1:5]
#             # println(idx)

#             # linear = linear_maps[n]
#             linear = Linear(X[:, :, n], Y[:, :, n], X[:, 1, n], Y[:, 1, n], false, tol)
#             # linear = Linear(X[:, 1:5, n], Y[:, 1:5, n], X[:, 1, n], Y[:, 1, n], true, tol)
#             # linear = Linear(W[:, idx, n], K[:, idx, n], W[:, 1, n], K[:, 1, n], true, tol)

#             # record num_singular_vals(W_n) and condition_number(W_n) 
#             diagnostics[k]["num_singular"][n] = linear.rank
#             println("m:", linear.rank)
#             diagnostics[k]["condition_number"][n] = linear.condition_number
#             interp_err = norm(Y[:, 1, n] - linear(X[:, 1, n]))
#             diagnostics[k]["interp_err"][n] = interp_err
#             # println("interp_err:", interp_err)
#             # println("A_norm:", norm(linear.A))
#             # println("A_det:", det(linear.A))

#             if linear.rank <= 1 
#                 pnew[:, n+1], qnew[:, n+1] = coarse_solve(pnew[:, n], qnew[:, n]) .- G[n] .+ F[n]
#             else
#                 Cunew_p, Cunew_q = coarse_solve(pnew[:, n], qnew[:, n])

#                 ratio = range_space_projection_ratio([Cunew_p; Cunew_q], linear)
#                 println("ratio:", ratio)
#                 diagnostics[k]["range_space_projection_ratio"][n] = ratio

#                 # if norm(corr) > 2 * maximum(norm.(eachcol(K[:, :, n])))
#                 if ratio < 0.98
#                     println("is exception")
#                     diagnostics[k]["is_exception"][n] = true
#                     pnew[:, n+1], qnew[:, n+1] = (Cunew_p, Cunew_q) .- G[n] .+ F[n]
#                 else
#                     diagnostics[k]["is_exception"][n] = false
#                     dp, dq = (Cunew_p, Cunew_q) .- G[n]
#                     corr = linear.A * [dp; dq]    
#                     pnew[:, n+1], qnew[:, n+1] = F[n]
#                     pnew[:, n+1] += corr[1:d]
#                     qnew[:, n+1] += corr[d+1:2*d]
#                 end
#             end 
#         end

#         p[:, :] = pnew[:, :]
#         q[:, :] = qnew[:, :]
#         p_all[:, :, k+1] = p
#         q_all[:, :, k+1] = q
#     end
    
#     return p_all, q_all, diagnostics
# end


# "Parareal algorithm with symplectic correction"
# function sympcorr(
#         p0::AbstractArray{T, 1},
#         q0::AbstractArray{T, 1},
#         t_grid::AbstractArray{Float64, 1},
#         fine_solve::Function,
#         coarse_solve::Function,
#         phi::Function;
#         objective::Function,
#         niters::Integer=3,
#         with_additive::Bool=true) where T<:AbstractFloat
    
#     # get dimension d
#     @assert length(p0) == length(q0)
#     d = length(p0)
    
#     # initialize arrays 
#     p = zeros(T, d, N+1)
#     q = zeros(T, d, N+1)
#     pnew = zero(p)
#     qnew = zero(q)
#     p_all = zeros(T, d, N+1, niters+1)
#     q_all = zeros(T, d, N+1, niters+1)
    
#     # solve for solutions at iteration 0 
#     p[:, 1] = p0
#     q[:, 1] = q0
#     @showprogress for n in 1:N
#         p[:, n+1], q[:, n+1] = coarse_solve(p[:, n], q[:, n], t_grid[n], dt[n])
#     end
#     p_all[:, :, 1] = p
#     q_all[:, :, 1] = q

#     # begin parareal iterations 
#     for k in 1:niters
        
#         println("iter ", k)
        
#         pnew[:, 1] = p0
#         qnew[:, 1] = q0
        
#         F = @showprogress pmap(fine_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2), t_grid[1:end-1], dt)
# #         G = pmap(coarse_solve, eachslice(p[:, 1:end-1], dims=2), eachslice(q[:, 1:end-1], dims=2), t_grid[1:end-1], dt)
#         G = [coarse_solve(p[:, n], q[:, n], t_grid[n], dt[n]) for n in 1:N]
        
#         H = sum(dt)/N
#         res = optimize(h -> objective(h, F, G), -H, H)
#         h = Optim.minimizer(res)
#         println(h)
        
#         if with_additive
#             @showprogress for n in 1:N
#                 pnew[:, n+1], qnew[:, n+1] = phi(coarse_solve(pnew[:, n], qnew[:, n], t_grid[n], dt[n])..., h) .- phi(G[n]..., h) .+ F[n] 
#             end
#         else
#             @showprogress for n in 1:N
#                 pnew[:, n+1], qnew[:, n+1] = phi(coarse_solve(pnew[:, n], qnew[:, n], t_grid[n], dt[n])..., h)
#             end
#         end

#         p = pnew
#         q = qnew

#         p_all[:, :, k+1] = p
#         q_all[:, :, k+1] = q
#     end
    
#     return p_all, q_all
# end

export plain, procrustes, interpolative

end
