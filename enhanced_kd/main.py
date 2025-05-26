import os
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
import time
import copy
import matplotlib.pyplot as plt

from model import GraphTransformer
from utils import (
    test_model_loader,
    calculate_params_sparsity,
    measure_inference_time_loader,
    plot_losses,
    plot_accuracy,
    plot_comparison_bar,
)


def feature_distillation_loss(student_features, teacher_features, scale=0.01):
    # Normalize features to make learning easier
    s_norm = F.normalize(student_features, p=2, dim=1)
    t_norm = F.normalize(teacher_features, p=2, dim=1)
    return scale * F.mse_loss(s_norm, t_norm)


def attention_transfer_loss(student_attn, teacher_attn, scale=0.01):
    """Transfer attention knowledge from teacher to student"""
    if student_attn is None or teacher_attn is None:
        return torch.tensor(0.0, device="cuda" if torch.cuda.is_available() else "cpu")

    edge_index_s, student_weights = student_attn
    edge_index_t, teacher_weights = teacher_attn

    if student_weights.shape != teacher_weights.shape:
        if teacher_weights.dim() > 1 and teacher_weights.shape[1] > 1:
            teacher_weights = teacher_weights.mean(dim=1, keepdim=True)
        if student_weights.dim() > 1 and student_weights.shape[1] > 1:
            student_weights = student_weights.mean(dim=1, keepdim=True)

    student_attn_log = F.log_softmax(student_weights, dim=0)
    teacher_attn = F.softmax(teacher_weights, dim=0)
    return scale * F.kl_div(student_attn_log, teacher_attn, reduction="batchmean")


def relational_distillation_loss(student_outs, teacher_outs, scale=0.01):
    """Transfer structural relations between nodes with scaling"""
    student_norm = F.normalize(student_outs, p=2, dim=1)
    teacher_norm = F.normalize(teacher_outs, p=2, dim=1)

    s_sim = torch.mm(student_norm, student_norm.t())
    t_sim = torch.mm(teacher_norm, teacher_norm.t())

    return scale * F.mse_loss(s_sim, t_sim)


def contrastive_distillation_loss(
    student_feat, teacher_feat, temperature=0.1, scale=0.01
):
    s_norm = F.normalize(student_feat, p=2, dim=1)
    t_norm = F.normalize(teacher_feat, p=2, dim=1)

    sim_matrix = torch.matmul(s_norm, t_norm.T) / temperature
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

    loss = F.cross_entropy(sim_matrix, labels)
    return scale * loss


TEACHER_HIDDEN = 128
STUDENT_HIDDEN = 48
TEACHER_HEADS = 8
STUDENT_HEADS = 4

LR = 0.005
STUDENT_LR = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 2048
NUM_NEIGHBORS = [15, 10]
GRADIENT_CLIP = 1.0

TASK_LOSS_W = 0.5
OUTPUT_KD_LOSS_W = 0.3
FEATURE_KD_LOSS_W = 0.1
RELATION_KD_LOSS_W = 0.05
ATTN_KD_LOSS_W = 0.025
CONTRASTIVE_KD_LOSS_W = 0.025
TEMPERATURE = 2.0

USE_PROGRESSIVE_KD = True
WARMUP_EPOCHS = 5

INFERENCE_RUNS = 100
INFERENCE_WARMUP = 10

PLOT_DIR = "plots_enhanced_kd"
NUM_WORKERS = 4


def main():
    print("--- Loading Data (ogbn-arxiv) ---")
    dataset_name = "ogbn-arxiv"
    dataset = PygNodePropPredDataset(
        name=dataset_name, root="data/OGB", transform=T.ToUndirected()
    )
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[split_idx["train"]] = True
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask[split_idx["valid"]] = True
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask[split_idx["test"]] = True

    num_features = data.num_node_features
    num_classes = dataset.num_classes

    print(f"Dataset: {dataset_name}:")
    print("======================")
    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.edge_index.shape[1]}")
    print(f"Number of training nodes: {data.train_mask.sum()}")
    print(f"Number of validation nodes: {data.val_mask.sum()}")
    print(f"Number of test nodes: {data.test_mask.sum()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    y_cpu = data.y

    print("--- Creating DataLoaders ---")
    train_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        input_nodes=data.train_mask,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE * 2,
        input_nodes=data.val_mask,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE * 2,
        input_nodes=data.test_mask,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("\n--- Initializing Teacher Model (GraphTransformer) ---")
    teacher_model = GraphTransformer(
        num_features, TEACHER_HIDDEN, num_classes, heads=TEACHER_HEADS
    ).to(device)
    print(teacher_model)

    print("\n--- Training Teacher Model ---")
    optimizer_teacher = torch.optim.Adam(
        teacher_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    criterion_teacher = torch.nn.CrossEntropyLoss()
    best_teacher_val_acc = 0
    teacher_patience_counter = 0

    for epoch in range(1, EPOCHS):
        teacher_model.train()
        total_loss = 0
        total_nodes = 0
        start_time = time.time()

        for batch in train_loader:
            batch = batch.to(device)
            optimizer_teacher.zero_grad()

            out = teacher_model(
                batch.x, batch.edge_index, return_hidden=False, return_attention=False
            )
            if isinstance(out, tuple):
                out = out[0]

            out = out[: batch.batch_size]
            batch_target_indices_cpu = batch.n_id[: batch.batch_size].to("cpu")
            batch_y = y_cpu[batch_target_indices_cpu].to(device).squeeze()

            loss = criterion_teacher(out, batch_y)
            loss.backward()
            optimizer_teacher.step()

            total_loss += loss.item() * batch.batch_size
            total_nodes += batch.batch_size

        avg_loss = total_loss / total_nodes if total_nodes > 0 else 0
        epoch_time = time.time() - start_time

        val_acc = test_model_loader(teacher_model, data, val_loader, device)
        test_acc = -1.0
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            test_acc = test_model_loader(teacher_model, data, test_loader, device)

        print(
            f"Teacher Epoch: {epoch:03d}, Time: {epoch_time:.2f}s, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}"
        )

        if val_acc > best_teacher_val_acc:
            best_teacher_val_acc = val_acc
            teacher_patience_counter = 0
            torch.save(teacher_model.state_dict(), f"best_teacher_{dataset_name}.pth")
        else:
            teacher_patience_counter += 1
            if teacher_patience_counter >= PATIENCE:
                print(f"Teacher early stopping at epoch {epoch}")
                break

    teacher_model.load_state_dict(torch.load(f"best_teacher_{dataset_name}.pth"))
    teacher_model.eval()

    for param in teacher_model.parameters():
        param.requires_grad = False

    print("\n--- Teacher Model Trained ---")
    final_teacher_test_acc = test_model_loader(teacher_model, data, test_loader, device)
    print(f"Final Teacher Test Acc: {final_teacher_test_acc:.4f}")

    print("\n--- Initializing Enhanced Distilled Student Model ---")
    student_model = GraphTransformer(
        num_features, STUDENT_HIDDEN, num_classes, heads=STUDENT_HEADS
    ).to(device)
    print(student_model)

    print("Determining feature dimensions for projection layer...")
    sample_batch = next(iter(train_loader)).to(device)

    with torch.no_grad():
        teacher_out = teacher_model(
            sample_batch.x,
            sample_batch.edge_index,
            return_hidden=True,
            return_attention=False,
        )
        teacher_hidden = teacher_out[1][: sample_batch.batch_size]

        student_out = student_model(
            sample_batch.x,
            sample_batch.edge_index,
            return_hidden=True,
            return_attention=False,
        )
        student_hidden = student_out[1][: sample_batch.batch_size]

        teacher_dim = teacher_hidden.shape[1]
        student_dim = student_hidden.shape[1]

    print(f"Actual feature dimensions - Student: {student_dim}, Teacher: {teacher_dim}")

    projection_layer = Sequential(
        Linear(student_dim, student_dim * 2),
        ReLU(),
        Linear(student_dim * 2, teacher_dim),
    ).to(device)

    print(
        f"Created projection MLP: {student_dim} -> {student_dim * 2} -> {teacher_dim}"
    )

    criterion_task = torch.nn.CrossEntropyLoss()
    criterion_distill = torch.nn.KLDivLoss(reduction="batchmean")

    optimizer_student = torch.optim.Adam(
        [
            {"params": student_model.parameters()},
            {"params": projection_layer.parameters()},
        ],
        lr=STUDENT_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler_student = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_student,
        mode="max",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=True,
    )

    print("\n--- Training Enhanced Distilled Student Model ---")
    student_losses_log = {
        "Total Loss": [],
        "Task Loss": [],
        "Output KD Loss": [],
        "Feature KD Loss": [],
        "Attention KD Loss": [],
        "Relational KD Loss": [],
        "Contrastive KD Loss": [],
    }

    student_val_accs_log = []
    best_student_val_acc = 0
    student_patience_counter = 0
    best_epoch = 0

    for epoch in range(1, EPOCHS):
        student_model.train()
        projection_layer.train()

        if USE_PROGRESSIVE_KD:
            # Warm-up phase: focus more on mimicking teacher
            if epoch <= WARMUP_EPOCHS:
                curr_task_w = 0.3
                curr_output_kd_w = 0.5
                curr_feature_kd_w = 0.1
                curr_relation_kd_w = 0.05
                curr_attn_kd_w = 0.025
                curr_contrastive_kd_w = 0.025
            else:
                # Gradually transition to task loss
                progress = min(1.0, (epoch - WARMUP_EPOCHS) / (EPOCHS // 2))
                curr_task_w = 0.3 + progress * 0.4
                remaining = 1.0 - curr_task_w

                total_kd = (
                    OUTPUT_KD_LOSS_W
                    + FEATURE_KD_LOSS_W
                    + RELATION_KD_LOSS_W
                    + ATTN_KD_LOSS_W
                    + CONTRASTIVE_KD_LOSS_W
                )
                curr_output_kd_w = OUTPUT_KD_LOSS_W * remaining / total_kd
                curr_feature_kd_w = FEATURE_KD_LOSS_W * remaining / total_kd
                curr_relation_kd_w = RELATION_KD_LOSS_W * remaining / total_kd
                curr_attn_kd_w = ATTN_KD_LOSS_W * remaining / total_kd
                curr_contrastive_kd_w = CONTRASTIVE_KD_LOSS_W * remaining / total_kd
        else:
            curr_task_w = TASK_LOSS_W
            curr_output_kd_w = OUTPUT_KD_LOSS_W
            curr_feature_kd_w = FEATURE_KD_LOSS_W
            curr_relation_kd_w = RELATION_KD_LOSS_W
            curr_attn_kd_w = ATTN_KD_LOSS_W
            curr_contrastive_kd_w = CONTRASTIVE_KD_LOSS_W

        total_loss, total_task, total_output_kd = 0, 0, 0
        total_feature_kd, total_relation_kd, total_attn_kd, total_contrastive_kd = (
            0,
            0,
            0,
            0,
        )
        total_nodes = 0
        start_time = time.time()

        for batch in train_loader:
            batch = batch.to(device)
            optimizer_student.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher_model(
                    batch.x, batch.edge_index, return_hidden=True, return_attention=True
                )

                if len(teacher_outputs) == 3:
                    out_t, hidden_t, attn_t = teacher_outputs
                else:
                    out_t, hidden_t = teacher_outputs[:2]
                    attn_t = None

            student_outputs = student_model(
                batch.x, batch.edge_index, return_hidden=True, return_attention=True
            )

            if len(student_outputs) == 3:
                out_s, hidden_s, attn_s = student_outputs
            else:
                out_s, hidden_s = student_outputs[:2]
                attn_s = None

            out_s_target = out_s[: batch.batch_size]
            out_t_target = out_t[: batch.batch_size]

            hidden_s_target = hidden_s[: batch.batch_size]
            hidden_t_target = hidden_t[: batch.batch_size]

            hidden_s_projected = projection_layer(hidden_s_target)

            batch_target_indices_cpu = batch.n_id[: batch.batch_size].to("cpu")
            batch_y = y_cpu[batch_target_indices_cpu].to(device).squeeze()

            task_loss = criterion_task(out_s_target, batch_y)

            student_softmax = F.log_softmax(out_s_target / TEMPERATURE, dim=1)
            teacher_softmax = F.softmax(out_t_target / TEMPERATURE, dim=1)
            output_kd_loss = criterion_distill(student_softmax, teacher_softmax) * (
                TEMPERATURE**2
            )

            feature_kd_loss = feature_distillation_loss(
                hidden_s_projected, hidden_t_target
            )

            if batch.batch_size <= 2048:
                relation_kd_loss = relational_distillation_loss(
                    out_s_target, out_t_target
                )
            else:
                relation_kd_loss = torch.tensor(0.0, device=device)

            if attn_s is not None and attn_t is not None:
                attn_kd_loss = attention_transfer_loss(attn_s, attn_t)
            else:
                attn_kd_loss = torch.tensor(0.0, device=device)

            if batch.batch_size <= 1024:
                contrastive_kd_loss = contrastive_distillation_loss(
                    hidden_s_projected, hidden_t_target
                )
            else:
                contrastive_kd_loss = torch.tensor(0.0, device=device)

            loss = (
                curr_task_w * task_loss
                + curr_output_kd_w * output_kd_loss
                + curr_feature_kd_w * feature_kd_loss
                + curr_relation_kd_w * relation_kd_loss
                + curr_attn_kd_w * attn_kd_loss
                + curr_contrastive_kd_w * contrastive_kd_loss
            )

            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(student_model.parameters())
                    + list(projection_layer.parameters()),
                    max_norm=GRADIENT_CLIP,
                )

                optimizer_student.step()

                total_loss += loss.item() * batch.batch_size
                total_task += task_loss.item() * batch.batch_size
                total_output_kd += output_kd_loss.item() * batch.batch_size
                total_feature_kd += feature_kd_loss.item() * batch.batch_size
                total_relation_kd += relation_kd_loss.item() * batch.batch_size
                total_attn_kd += attn_kd_loss.item() * batch.batch_size
                total_contrastive_kd += contrastive_kd_loss.item() * batch.batch_size
                total_nodes += batch.batch_size
            else:
                print(f"Warning: NaN/Inf loss detected in batch. Skipping batch.")

        avg_loss = total_loss / total_nodes if total_nodes > 0 else float("nan")
        avg_task = total_task / total_nodes if total_nodes > 0 else float("nan")
        avg_output_kd = (
            total_output_kd / total_nodes if total_nodes > 0 else float("nan")
        )
        avg_feature_kd = (
            total_feature_kd / total_nodes if total_nodes > 0 else float("nan")
        )
        avg_relation_kd = (
            total_relation_kd / total_nodes if total_nodes > 0 else float("nan")
        )
        avg_attn_kd = total_attn_kd / total_nodes if total_nodes > 0 else float("nan")
        avg_contrastive_kd = (
            total_contrastive_kd / total_nodes if total_nodes > 0 else float("nan")
        )

        epoch_time = time.time() - start_time

        student_losses_log["Total Loss"].append(avg_loss)
        student_losses_log["Task Loss"].append(avg_task)
        student_losses_log["Output KD Loss"].append(avg_output_kd)
        student_losses_log["Feature KD Loss"].append(avg_feature_kd)
        student_losses_log["Relational KD Loss"].append(avg_relation_kd)
        student_losses_log["Attention KD Loss"].append(avg_attn_kd)
        student_losses_log["Contrastive KD Loss"].append(avg_contrastive_kd)

        student_val_acc = test_model_loader(student_model, data, val_loader, device)
        student_val_accs_log.append(student_val_acc)

        scheduler_student.step(student_val_acc)

        student_test_acc = -1.0
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            student_test_acc = test_model_loader(
                student_model, data, test_loader, device
            )

        print(
            f"Enhanced KD Epoch: {epoch:03d}, Time: {epoch_time:.2f}s, "
            f'LR: {optimizer_student.param_groups[0]["lr"]:.6f}, Loss: {avg_loss:.4f}, '
            f"Val Acc: {student_val_acc:.4f}, Test Acc: {student_test_acc:.4f}"
        )

        if not np.isnan(student_val_acc):
            if student_val_acc > best_student_val_acc:
                best_student_val_acc = student_val_acc
                student_patience_counter = 0
                best_epoch = epoch
                torch.save(
                    {
                        "student_model": student_model.state_dict(),
                        "projection_layer": projection_layer.state_dict(),
                    },
                    f"best_student_{dataset_name}.pth",
                )
            else:
                student_patience_counter += 1
                if student_patience_counter >= PATIENCE:
                    print(f"Enhanced KD Student early stopping at epoch {epoch}")
                    break
        else:
            print(f"Warning: NaN validation accuracy at epoch {epoch}.")

    checkpoint = torch.load(f"best_student_{dataset_name}.pth")
    student_model.load_state_dict(checkpoint["student_model"])
    projection_layer.load_state_dict(checkpoint["projection_layer"])
    student_model.eval()

    print(
        f"\n--- Enhanced Distilled Student Model Trained (Best at epoch {best_epoch}) ---"
    )
    final_student_test_acc = test_model_loader(student_model, data, test_loader, device)
    print(f"Final Enhanced Distilled Student Test Acc: {final_student_test_acc:.4f}")

    print("\n--- Model Parameter Counts & Sparsity ---")
    teacher_params, teacher_sparsity = calculate_params_sparsity(teacher_model)
    student_params, student_sparsity = calculate_params_sparsity(student_model)

    print(
        f"Teacher (Transformer): Params={teacher_params: <10} Sparsity={teacher_sparsity:.4f}"
    )
    print(
        f"Student (Transformer): Params={student_params: <10} Sparsity={student_sparsity:.4f}"
    )
    print(f"Compression Ratio: {teacher_params/student_params:.2f}x")

    print("\n--- Measuring Inference Latency (Time per Batch) ---")
    teacher_latency = measure_inference_time_loader(
        teacher_model,
        test_loader,
        device,
        num_batches=INFERENCE_RUNS,
        warmup_batches=INFERENCE_WARMUP,
    )
    student_latency = measure_inference_time_loader(
        student_model,
        test_loader,
        device,
        num_batches=INFERENCE_RUNS,
        warmup_batches=INFERENCE_WARMUP,
    )
    print(f"Speedup: {teacher_latency/student_latency:.2f}x")

    print(f"\n--- Saving Training and Comparison Plots to '{PLOT_DIR}/' directory ---")

    plot_losses(
        student_losses_log,
        "Enhanced Distilled Student Training Losses",
        filename=f"{dataset_name}_enhanced_distilled_student_losses.png",
        plot_dir=PLOT_DIR,
    )

    plot_accuracy(
        student_val_accs_log,
        "Enhanced Distilled Student Validation Accuracy",
        filename=f"{dataset_name}_enhanced_distilled_student_accuracy.png",
        plot_dir=PLOT_DIR,
    )

    model_labels = ["Teacher\n(Transformer)", "Enhanced Distilled\nStudent"]
    accuracies = [final_teacher_test_acc, final_student_test_acc]
    latencies = [teacher_latency, student_latency]
    params = [teacher_params, student_params]

    plot_comparison_bar(
        accuracies,
        model_labels,
        f"{dataset_name} Test Accuracy Comparison",
        "Test Accuracy",
        filename=f"{dataset_name}_comparison_accuracy.png",
        plot_dir=PLOT_DIR,
    )

    plot_comparison_bar(
        latencies,
        model_labels,
        f"{dataset_name} Inference Latency Comparison",
        "Avg Latency per Batch (ms)",
        filename=f"{dataset_name}_comparison_latency.png",
        plot_dir=PLOT_DIR,
    )

    plot_comparison_bar(
        params,
        model_labels,
        f"{dataset_name} Parameter Count Comparison",
        "# Parameters",
        filename=f"{dataset_name}_comparison_params.png",
        plot_dir=PLOT_DIR,
    )

    print("\n--- Enhanced Knowledge Distillation Complete ---")


if __name__ == "__main__":
    main()
