import torch
import torch.nn.utils.prune as prune
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.loader import NeighborLoader
import matplotlib.pyplot as plt
import time
import numpy as np
import torch.nn.functional as F

import os


@torch.no_grad()
def test_model_loader(model, data, loader, device):
    model.eval()
    total_correct = 0
    total_examples = 0
    y_cpu = data.y

    for batch in loader:
        batch = batch.to(device)
        out = model(
            batch.x, batch.edge_index, return_hidden=False, return_attention=False
        )
        if isinstance(out, tuple):
            out = out[0]
        out = out[: batch.batch_size]
        pred = out.argmax(dim=-1)
        batch_target_indices_cpu = batch.n_id[: batch.batch_size].to("cpu")
        batch_y_true = y_cpu[batch_target_indices_cpu].to(device).squeeze()
        correct = pred == batch_y_true
        total_correct += int(correct.sum())
        total_examples += batch.batch_size

    return total_correct / total_examples if total_examples > 0 else 0


def calculate_params_sparsity(model):
    """Calculates total parameters and sparsity."""
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param.data == 0).sum().item()
    sparsity = zero_params / total_params if total_params > 0 else 0
    return total_params, sparsity


def fine_tune_model_loader(
    model,
    data,
    train_loader,
    val_loader,
    device,
    epochs=50,
    lr=0.001,
    weight_decay=5e-4,
    patience=10,
):
    print("\n--- Fine-tuning function called (but disabled in this run) ---")
    return []


@torch.no_grad()
def measure_inference_time_loader(
    model, loader, device, num_batches=100, warmup_batches=10
):
    model.eval()
    model.to(device)
    latencies = []
    batch_iterator = iter(loader)

    print(f"Performing {warmup_batches} warmup batches...")
    for _ in range(warmup_batches):
        try:
            batch = next(batch_iterator).to(device)
            out = model(
                batch.x, batch.edge_index, return_hidden=False, return_attention=False
            )
            if isinstance(out, tuple):
                out = out[0]
        except StopIteration:
            print("Warning: Ran out of batches during warmup.")
            batch_iterator = iter(loader)
            break
    print("Warmup complete.")

    print(f"Performing {num_batches} measurement batches...")
    actual_runs = 0
    if device.type == "cuda":
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        for i in range(num_batches):
            try:
                batch = next(batch_iterator).to(device)
                starter.record()
                out = model(
                    batch.x,
                    batch.edge_index,
                    return_hidden=False,
                    return_attention=False,
                )
                if isinstance(out, tuple):
                    out = out[0]
                ender.record()
                torch.cuda.synchronize()
                latencies.append(starter.elapsed_time(ender))
                actual_runs += 1
            except StopIteration:
                print(f"Warning: Ran out of batches after {i} measurements.")
                break
    else:
        for i in range(num_batches):
            try:
                batch = next(batch_iterator).to(device)
                start_time = time.time()
                out = model(
                    batch.x,
                    batch.edge_index,
                    return_hidden=False,
                    return_attention=False,
                )
                if isinstance(out, tuple):
                    out = out[0]
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)
                actual_runs += 1
            except StopIteration:
                print(f"Warning: Ran out of batches after {i} measurements.")
                break
    print("Measurement complete.")

    if not latencies:
        print("Warning: No latency measurements recorded.")
        return 0.0

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    print(
        f"Avg Inference Latency ({actual_runs} batches): {avg_latency:.3f} +/- {std_latency:.3f} ms per batch"
    )
    return avg_latency


def attention_loss_mse(att_s, att_t):
    if att_s is None or att_t is None:
        print("Warning: Attention tuple is None.")
        return torch.tensor(0.0, device="cpu")

    edge_index_s, scores_s = att_s
    edge_index_t, scores_t = att_t

    if scores_s is None or scores_t is None:
        print("Warning: Attention scores are None.")
        device = (
            edge_index_s.device
            if edge_index_s is not None
            else (edge_index_t.device if edge_index_t is not None else "cpu")
        )
        return torch.tensor(0.0, device=device)

    if torch.isnan(scores_s).any() or torch.isinf(scores_s).any():
        print(
            "Warning: NaN/Inf detected in student attention scores. Returning 0 loss."
        )
        return torch.tensor(0.0, device=scores_s.device)
    if torch.isnan(scores_t).any() or torch.isinf(scores_t).any():
        print(
            "Warning: NaN/Inf detected in teacher attention scores. Returning 0 loss."
        )
        return torch.tensor(0.0, device=scores_t.device)

    if scores_s.shape[0] != scores_t.shape[0]:
        print(
            f"Warning: Attention score shape mismatch (num_edges). S: {scores_s.shape}, T: {scores_t.shape}"
        )
        return torch.tensor(0.0, device=scores_s.device)

    if scores_s.dim() != scores_t.dim() or (
        scores_s.dim() > 1 and scores_s.shape[1] != scores_t.shape[1]
    ):
        if (
            scores_s.dim() == scores_t.dim()
            and scores_t.dim() > 1
            and scores_t.shape[1] > 1
            and scores_t.shape[1] > scores_s.shape[1]
        ):
            print("Warning: Averaging teacher attention heads for loss calculation.")
            scores_t = scores_t.mean(dim=1, keepdim=True)
            if scores_s.shape[1] != scores_t.shape[1]:
                print(
                    "Warning: Cannot align attention head dimensions after averaging."
                )
                return torch.tensor(0.0, device=scores_s.device)
        else:
            print(
                f"Warning: Incompatible attention score dimensions. S: {scores_s.shape}, T: {scores_t.shape}"
            )
            return torch.tensor(0.0, device=scores_s.device)

    loss = F.mse_loss(scores_s, scores_t)

    if torch.isnan(loss):
        print("Warning: MSE attention loss resulted in NaN. Returning 0 loss.")
        return torch.tensor(0.0, device=scores_s.device)

    return loss


def _setup_plot_dir(plot_dir="plots"):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    return plot_dir


def plot_losses(losses_dict, title, filename="losses.png", plot_dir="plots"):
    plot_dir = _setup_plot_dir(plot_dir)
    filepath = os.path.join(plot_dir, filename)
    plt.figure(figsize=(12, 8))
    for label, values in losses_dict.items():
        alpha = 0.7 if "Loss" in label and "Total" not in label else 1.0
        plt.plot(values, label=label, alpha=alpha)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved loss plot to {filepath}")
    plt.close()


def plot_accuracy(acc_list, title, filename="accuracy.png", plot_dir="plots"):
    plot_dir = _setup_plot_dir(plot_dir)
    filepath = os.path.join(plot_dir, filename)
    plt.figure(figsize=(12, 5))
    plt.plot(acc_list, label="Validation Accuracy")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved accuracy plot to {filepath}")
    plt.close()


def plot_comparison_bar(
    values, labels, title, ylabel, filename="comparison.png", plot_dir="plots"
):
    plot_dir = _setup_plot_dir(plot_dir)
    filepath = os.path.join(plot_dir, filename)
    plt.figure(figsize=(10, 7))
    colors = ["#f9f", "#baffc9"]
    bars = plt.bar(labels, values, color=colors[: len(values)])
    plt.ylabel(ylabel)
    plt.title(title)
    for bar in bars:
        yval = bar.get_height()
        format_str = "{:,.0f}" if yval >= 1000 else "{:.3f}" if yval < 10 else "{:.2f}"
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            format_str.format(yval),
            va="bottom",
            ha="center",
            fontsize=9,
        )
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved comparison plot to {filepath}")
    plt.close()
