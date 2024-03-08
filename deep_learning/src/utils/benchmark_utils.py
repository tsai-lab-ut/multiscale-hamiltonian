import torch
import torch.utils.benchmark as benchmark
import pandas as pd

from models import BaseSolutionMap
from typing import List


def time_forward(model: BaseSolutionMap, nsteps_list: List[int] = [0, 1,]):
    """Benchmark forward pass time of a SolutionMap model."""

    # compare runtime for different batch sizes and devices 
    batch_sizes = [1, 64, 128, 512]
    devices = [torch.device("cpu")] 
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    
    # use default initial states as inputs
    u0 = model.problem.default_initial_states().to(model.dtype)[0, :]

    results = []
    for b in batch_sizes:
        for device in devices:
            x = u0.repeat(b, 1).to(device)
            model.to(device)
            label = "forward time"
            sub_label=f"{x.shape}"

            for nsteps in nsteps_list:
                results.append(benchmark.Timer(
                    stmt="model(x, sequence_len=nsteps+1)",
                    globals={"x": x, "model": model, "nsteps": nsteps},
                    label=label,
                    sub_label=sub_label,
                    description=f"{device}: nsteps={nsteps}",
                ).blocked_autorange(min_run_time=1))

    return benchmark.Compare(results)


def time_backward(model: BaseSolutionMap, nsteps_list: List[int] = [0, 1,]):
    """Benchmark backward pass time of a SolutionMap model."""

    # compare runtime for different batch sizes
    batch_sizes = [1, 64, 128, 512]

    # use default initial states as inputs
    u0 = model.problem.default_initial_states().to(model.dtype)[0, :]

    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    model.to(device)
    u0 = u0.to(device)

    def backward(model, batch):
        out = model.model_step(batch, batch_idx=0)
        loss = out["loss"]
        loss.backward()
        return

    results = []
    for b in batch_sizes:
        x = u0.repeat(b, 1)
        label = f"backward time ({device})"
        sub_label=f"{x.shape}"

        for nsteps in nsteps_list:
            batch = [x for _ in range(nsteps+1)]
            seq_weights = torch.ones(nsteps+1)
            model.set_seq_weights(seq_weights)
            results.append(benchmark.Timer(
                stmt="backward(model, batch)",
                globals={"backward": backward, "model": model, "batch": batch},
                label=label,
                sub_label=sub_label,
                description=f"nsteps={nsteps}",
            ).blocked_autorange(min_run_time=1))

    return benchmark.Compare(results)


def outputs_stats(model: BaseSolutionMap, nsteps: int = 5):
    """Benchmark outputs mean and variance for a SolutionMap model."""
    
    data = {}
    
    # use default initial states as inputs
    u0 = model.problem.default_initial_states().to(model.dtype)

    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    model.to(device)
    u0 = u0.to(device)

    v0, x0 = u0.chunk(2, dim=-1)
    col = pd.Series(
        torch.stack([torch.mean(v0), torch.mean(x0), torch.var(v0), torch.var(x0)]).detach().cpu().numpy(),
        index=["v_mean", "x_mean", "v_var", "x_var"])
    data["input"] = col

    pred_seq = model.predict_step(u0, batch_idx=0, sequence_len=nsteps+1)
    for i, u in enumerate(pred_seq):
        v, x = u.chunk(2, dim=-1)
        col = pd.Series(
            torch.stack([torch.mean(v), torch.mean(x), torch.var(v), torch.var(x)]).detach().cpu().numpy(),
            index=["v_mean", "x_mean", "v_var", "x_var"])
        data[f"output_{i}"] = col
    
    return pd.DataFrame(data=data)


# def time_backward(model: BaseSolutionMap, max_nsteps: int = 5):
#     """Benchmark backward pass time of a SolutionMap model."""

#     # compare runtime for different batch sizes
#     batch_sizes = [1, 64, 128, 512]

#     # use default initial states as inputs
#     u0 = model.problem.default_initial_states().to(model.dtype)[0, :]

#     device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
#     model.to(device)
#     u0 = u0.to(device)

#     def backward(model, batch):
#         out = model.model_step(batch, batch_idx=0)
#         loss = out["loss"]
#         loss.backward()
#         return

#     results = []
#     for b in batch_sizes:
#         x = u0.repeat(b, 1)
#         batch = [x for _ in range(max_nsteps+1)]
#         label = f"backward time ({device})"
#         sub_label=f"{x.shape}"

#         for i in range(max_nsteps):
#             seq_weights = torch.zeros(max_nsteps)
#             seq_weights[i] = 1.
#             model.set_seq_weights(seq_weights)
#             results.append(benchmark.Timer(
#                 stmt="backward(model, batch)",
#                 globals={"backward": backward, "model": model, "batch": batch},
#                 label=label,
#                 sub_label=sub_label,
#                 description=f"step {i+1}",
#             ).blocked_autorange(min_run_time=1))

#         seq_weights = torch.ones(max_nsteps)
#         model.set_seq_weights(seq_weights)
#         results.append(benchmark.Timer(
#             stmt="backward(model, batch)",
#             globals={"backward": backward, "model": model, "batch": batch},
#             label=label,
#             sub_label=sub_label,
#             description=f"all steps",
#         ).blocked_autorange(min_run_time=1))

#     return benchmark.Compare(results)