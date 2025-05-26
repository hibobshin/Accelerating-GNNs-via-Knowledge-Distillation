import torch
import torch.nn.utils.prune as prune
from torch_geometric.nn import GCNConv, GATConv
import matplotlib.pyplot as plt
import time
import numpy as np
import os


def test_model(model, data, device):
    model.eval()
    with torch.no_grad():
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.to(device)
        val_mask = data.val_mask.to(device)
        test_mask = data.test_mask.to(device)

        out = model(x, edge_index, return_hidden=False)
        pred = out.argmax(dim=1)

        val_correct = pred[val_mask] == y[val_mask]
        val_acc = (
            int(val_correct.sum()) / int(val_mask.sum())
            if int(val_mask.sum()) > 0
            else 0
        )

        test_correct = pred[test_mask] == y[test_mask]
        test_acc = (
            int(test_correct.sum()) / int(test_mask.sum())
            if int(test_mask.sum()) > 0
            else 0
        )
    return val_acc, test_acc


def calculate_params_sparsity(model):
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param.data == 0).sum().item()
    sparsity = zero_params / total_params if total_params > 0 else 0
    return total_params, sparsity


def make_pruning_permanent(model):
    for module_name, module in model.named_modules():
        if isinstance(module, GCNConv):
            if prune.is_pruned(module.lin):
                prune.remove(module.lin, "weight")
        elif isinstance(module, GATConv):
            possible_lin_layers = ["lin", "lin_src", "lin_dst", "lin_l", "lin_r"]
            for layer_name in possible_lin_layers:
                if hasattr(module, layer_name):
                    lin_layer = getattr(module, layer_name)
                    if isinstance(lin_layer, torch.nn.Linear) and prune.is_pruned(
                        lin_layer
                    ):
                        try:
                            prune.remove(lin_layer, "weight")
                        except ValueError:
                            pass
                        try:
                            prune.remove(lin_layer, "bias")
                        except ValueError:
                            pass
        elif isinstance(module, torch.nn.Linear):
            if prune.is_pruned(module):
                try:
                    prune.remove(module, "weight")
                except ValueError:
                    pass
                try:
                    prune.remove(module, "bias")
                except ValueError:
                    pass


def fine_tune_model(
    model, data, device, epochs=50, lr=0.001, weight_decay=5e-4, patience=10
):
    print(f"\n--- Fine-tuning Model for {epochs} epochs with lr={lr} ---")
    for param in model.parameters():
        param.requires_grad = True
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    val_accs = []
    best_val_acc = 0
    patience_counter = 0

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        val_acc, test_acc = test_model(model, data, device)
        val_accs.append(val_acc)

        if epoch % 10 == 0:
            print(
                f"Fine-tune Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Fine-tuning early stopping at epoch {epoch}")
                break

    print("--- Fine-tuning Complete ---")
    return val_accs


def measure_inference_time(model, data, device, num_runs=100, warmup_runs=10):
    model.eval()
    model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    latencies = []

    with torch.no_grad():
        print(f"Performing {warmup_runs} warmup runs...")
        for _ in range(warmup_runs):
            _ = model(x, edge_index)
        print("Warmup complete.")

        print(f"Performing {num_runs} measurement runs...")
        if device.type == "cuda":
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            for _ in range(num_runs):
                starter.record()
                _ = model(x, edge_index)
                ender.record()
                torch.cuda.synchronize()
                latencies.append(starter.elapsed_time(ender))
        else:
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(x, edge_index)
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)
        print("Measurement complete.")

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    print(
        f"Avg Inference Latency ({num_runs} runs): {avg_latency:.3f} +/- {std_latency:.3f} ms"
    )
    return avg_latency


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
    colors = ["#f9f", "#ccf", "#cfc", "#ffdfba"]
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
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"Saved comparison plot to {filepath}")
    plt.close()
