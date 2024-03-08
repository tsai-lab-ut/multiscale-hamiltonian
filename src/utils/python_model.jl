using PyCall

py"""
import sys
sys.path.insert(0, "/workspace/projects_rui/learnsolnmap/deep_learning")
"""
torch = pyimport("torch")
model = pyimport("model")

function load_nn(checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    # nn_func = model.SolutionMap(checkpoint["hyper_parameters"]...).double()
    nn_func = model.SolutionMap.load_from_checkpoint(checkpoint_path, strict=false).double()
    nn_func.load_state_dict(checkpoint["state_dict"], strict=false)
    nn_func.to("cpu")
    return nn_func
end 

function nn_solve(
    p0::AbstractArray{T, 1}, 
    q0::AbstractArray{T, 1}, 
    nn_func::PyObject) where T<:AbstractFloat
    
    if T != Float64
        p0 = convert.(Float64, p0)
        q0 = convert.(Float64, q0)
    end

    dim = length(p0)
    u0 = torch.tensor(vcat(p0, q0))
    u = nn_func(u0)
    u = u.detach().numpy()
    p = u[1:dim]
    q = u[dim+1:end]

    if T != Float64
        p = convert.(T, p)
        q = convert.(T, q)
    end
    return p, q
end
