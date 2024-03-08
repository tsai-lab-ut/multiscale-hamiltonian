using DifferentialEquations
using MultiFloats 


METHODS = Dict(
    "VelocityVerlet"=>VelocityVerlet(), 
    "CalvoSanz4"=>CalvoSanz4(),
    "McAte5"=>McAte5(),
    "KahanLi6"=>KahanLi6(),
    "KahanLi8"=>KahanLi8(),
    "DPRKN12"=>DPRKN12(),
)


function ode_solve(
        A::Function, 
        method::OrdinaryDiffEqAlgorithm, 
        v0::AbstractArray{T, 1}, 
        x0::AbstractArray{T, 1}, 
        t0::Float64, 
        H::Float64,
        nsteps::Integer,
        retfull::Bool;
        param::Any=nothing) where T<:AbstractFloat
        
    t0 = convert(T, t0)
    H = convert(T, H)
        
    h = H/nsteps 
    prob = SecondOrderODEProblem(A, v0, x0, (t0, t0+H), param);
    if retfull 
        sol = solve(prob, method, tstops=t0:h:(t0+H), adaptive=false, save_everystep=true);
        V = hcat([u.x[1] for u in sol.u]...)   
        X = hcat([u.x[2] for u in sol.u]...)
        return V, X
    else 
        sol = solve(prob, method, tstops=t0:h:(t0+H), adaptive=false, save_everystep=false);
        v = sol[end].x[1]
        x = sol[end].x[2]
        return v, x
    end
end

    
Base.round(x::MultiFloat{Float64, N}, y::RoundingMode) where {N} = MultiFloat{Float64, N}(Base.round(BigFloat(x),y))
Base.trunc(x::Type{Integer}, y::MultiFloat{Float64, N}) where {N} = Base.trunc(x::Type{Integer}, BigFloat(y))
