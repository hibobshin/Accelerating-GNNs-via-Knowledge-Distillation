import os
import torch

os.environ["TORCH"] = torch.__version__
print(f"PyTorch Version: {torch.__version__}")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.utils.prune as prune
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import numpy as np
import time
import copy


from model import GCN, GATStudent
from utils import (
    test_model,
    calculate_params_sparsity,
    make_pruning_permanent,
    fine_tune_model,
    measure_inference_time,
    plot_losses,
    plot_accuracy,
    plot_comparison_bar,
)


TEACHER_HIDDEN = 64
STUDENT_HIDDEN = 16
GAT_HEADS = 8

LR = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 201
PATIENCE = 20

FT_EPOCHS = 51
FT_LR = 0.001
FT_PATIENCE = 10

TASK_LOSS_W = 0.3
DISTILL_LOSS_W = 0.5
FEATURE_LOSS_W = 0.2
TEMPERATURE = 4.0

PRUNING_AMOUNT = 0.3

INFERENCE_RUNS = 100
INFERENCE_WARMUP = 10

PLOT_DIR = "plots_cora_final"


print("--- Loading Data ---")
dataset = Planetoid(root="data/Planetoid", name="Cora", transform=NormalizeFeatures())
data = dataset[0]

print(f"Dataset: {dataset}:")
print("======================")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of features: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Number of training nodes: {data.train_mask.sum()}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
data = data.to(device)

print("\n--- Initializing Teacher Model ---")
teacher_model = GCN(dataset.num_features, TEACHER_HIDDEN, dataset.num_classes).to(
    device
)
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
    optimizer_teacher.zero_grad()
    out = teacher_model(data.x, data.edge_index)
    loss = criterion_teacher(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer_teacher.step()

    val_acc, test_acc = test_model(teacher_model, data, device)
    if epoch % 10 == 0:
        print(
            f"Teacher Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}"
        )
    if val_acc > best_teacher_val_acc:
        best_teacher_val_acc = val_acc
        teacher_patience_counter = 0
    else:
        teacher_patience_counter += 1
        if teacher_patience_counter >= PATIENCE:
            print(f"Teacher early stopping at epoch {epoch}")
            break
teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False
print("\n--- Teacher Model Trained ---")
final_teacher_val_acc, final_teacher_test_acc = test_model(teacher_model, data, device)
print(
    f"Final Teacher Val Acc: {final_teacher_val_acc:.4f}, Final Teacher Test Acc: {final_teacher_test_acc:.4f}"
)

print("\n--- Initializing Distilled GCN Student Model ---")
gcn_student_model = GCN(dataset.num_features, STUDENT_HIDDEN, dataset.num_classes).to(
    device
)
print(gcn_student_model)

assert np.isclose(
    TASK_LOSS_W + DISTILL_LOSS_W + FEATURE_LOSS_W, 1.0
), "Loss weights should sum to 1"

criterion_task = torch.nn.CrossEntropyLoss()
criterion_distill = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
criterion_feature = torch.nn.MSELoss()
feature_projection = torch.nn.Linear(STUDENT_HIDDEN, TEACHER_HIDDEN).to(device)

optimizer_gcn_student = torch.optim.Adam(
    list(gcn_student_model.parameters()) + list(feature_projection.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)
scheduler_gcn_student = torch.optim.lr_scheduler.StepLR(
    optimizer_gcn_student, step_size=50, gamma=0.5
)

print("\n--- Training Distilled GCN Student Model ---")
with torch.no_grad():
    teacher_outputs, teacher_hidden = teacher_model(
        data.x, data.edge_index, return_hidden=True
    )

gcn_student_losses_log = {
    "Total Loss": [],
    "Task Loss": [],
    "Distill Loss": [],
    "Feature Loss": [],
}
gcn_student_val_accs_log = []
best_gcn_student_val_acc = 0
gcn_student_patience_counter = 0

for epoch in range(1, EPOCHS):
    gcn_student_model.train()
    feature_projection.train()
    optimizer_gcn_student.zero_grad()

    student_outputs, student_hidden = gcn_student_model(
        data.x, data.edge_index, return_hidden=True
    )

    task_loss = criterion_task(
        student_outputs[data.train_mask], data.y[data.train_mask]
    )
    student_log_softmax = F.log_softmax(student_outputs / TEMPERATURE, dim=1)
    teacher_log_softmax = F.log_softmax(teacher_outputs / TEMPERATURE, dim=1)
    distill_loss = criterion_distill(student_log_softmax, teacher_log_softmax) * (
        TEMPERATURE**2
    )
    student_hidden_proj = feature_projection(student_hidden)
    feature_loss = criterion_feature(student_hidden_proj, teacher_hidden)
    loss = (
        TASK_LOSS_W * task_loss
        + DISTILL_LOSS_W * distill_loss
        + FEATURE_LOSS_W * feature_loss
    )

    loss.backward()
    optimizer_gcn_student.step()
    scheduler_gcn_student.step()

    gcn_student_losses_log["Total Loss"].append(loss.item())
    gcn_student_losses_log["Task Loss"].append(task_loss.item())
    gcn_student_losses_log["Distill Loss"].append(distill_loss.item())
    gcn_student_losses_log["Feature Loss"].append(feature_loss.item())

    student_val_acc, student_test_acc = test_model(gcn_student_model, data, device)
    gcn_student_val_accs_log.append(student_val_acc)
    if epoch % 10 == 0:
        print(
            f"GCN Distill Epoch: {epoch:03d}, Lr: {scheduler_gcn_student.get_last_lr()[0]:.6f}, Total Loss: {loss:.4f}, Task: {task_loss:.4f}, Distill: {distill_loss:.4f}, Feat: {feature_loss:.4f}, Val Acc: {student_val_acc:.4f}, Test Acc: {student_test_acc:.4f}"
        )

    if student_val_acc > best_gcn_student_val_acc:
        best_gcn_student_val_acc = student_val_acc
        gcn_student_patience_counter = 0
    else:
        gcn_student_patience_counter += 1
        if gcn_student_patience_counter >= PATIENCE:
            print(f"Distilled GCN Student early stopping at epoch {epoch}")
            break

print("\n--- Distilled GCN Student Model Trained (Before Pruning) ---")
final_gcn_student_val_acc, final_gcn_student_test_acc = test_model(
    gcn_student_model, data, device
)
print(
    f"Final Distilled GCN Student Val Acc: {final_gcn_student_val_acc:.4f}, Final Distilled GCN Student Test Acc: {final_gcn_student_test_acc:.4f}"
)


pruned_gcn_student_model = copy.deepcopy(gcn_student_model)

print(f"\n--- Applying Pruning ({PRUNING_AMOUNT*100:.1f}%) to GCN Student Model ---")
parameters_to_prune = (
    (pruned_gcn_student_model.conv1.lin, "weight"),
    (pruned_gcn_student_model.conv2.lin, "weight"),
)
prune.global_unstructured(
    parameters_to_prune, pruning_method=prune.L1Unstructured, amount=PRUNING_AMOUNT
)
make_pruning_permanent(pruned_gcn_student_model)
print("Pruning made permanent.")
pruned_val_acc_before_ft, pruned_test_acc_before_ft = test_model(
    pruned_gcn_student_model, data, device
)
print(
    f"Pruned GCN Student Acc (Before FT): Val={pruned_val_acc_before_ft:.4f}, Test={pruned_test_acc_before_ft:.4f}"
)


ft_val_accs_log = fine_tune_model(
    pruned_gcn_student_model,
    data,
    device,
    epochs=FT_EPOCHS,
    lr=FT_LR,
    weight_decay=WEIGHT_DECAY,
    patience=FT_PATIENCE,
)
print("\n--- Evaluating Pruned GCN Student Model (After Fine-tuning) ---")
final_pruned_gcn_val_acc, final_pruned_gcn_test_acc = test_model(
    pruned_gcn_student_model, data, device
)
print(
    f"Final Pruned GCN Student Val Acc: {final_pruned_gcn_val_acc:.4f}, Final Pruned GCN Student Test Acc: {final_pruned_gcn_test_acc:.4f}"
)


print("\n--- Initializing Distilled GAT Student Model ---")
gat_student_model = GATStudent(
    in_channels=dataset.num_features,
    hidden_channels=STUDENT_HIDDEN,
    out_channels=dataset.num_classes,
    heads=GAT_HEADS,
).to(device)
print(gat_student_model)


feature_projection_gat = torch.nn.Linear(STUDENT_HIDDEN, TEACHER_HIDDEN).to(device)
optimizer_gat_student = torch.optim.Adam(
    list(gat_student_model.parameters()) + list(feature_projection_gat.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler_gat_student = torch.optim.lr_scheduler.StepLR(
    optimizer_gat_student, step_size=50, gamma=0.5
)

print("\n--- Training Distilled GAT Student Model ---")


gat_student_losses_log = {
    "Total Loss": [],
    "Task Loss": [],
    "Distill Loss": [],
    "Feature Loss": [],
}
gat_student_val_accs_log = []
best_gat_student_val_acc = 0
gat_student_patience_counter = 0


for epoch in range(1, EPOCHS):
    gat_student_model.train()
    feature_projection_gat.train()
    optimizer_gat_student.zero_grad()

    student_outputs, student_hidden = gat_student_model(
        data.x, data.edge_index, return_hidden=True
    )

    task_loss = criterion_task(
        student_outputs[data.train_mask], data.y[data.train_mask]
    )
    student_log_softmax = F.log_softmax(student_outputs / TEMPERATURE, dim=1)
    teacher_log_softmax = F.log_softmax(teacher_outputs / TEMPERATURE, dim=1)
    distill_loss = criterion_distill(student_log_softmax, teacher_log_softmax) * (
        TEMPERATURE**2
    )
    student_hidden_proj = feature_projection_gat(student_hidden)
    feature_loss = criterion_feature(student_hidden_proj, teacher_hidden)

    loss = (
        TASK_LOSS_W * task_loss
        + DISTILL_LOSS_W * distill_loss
        + FEATURE_LOSS_W * feature_loss
    )

    loss.backward()
    optimizer_gat_student.step()
    scheduler_gat_student.step()

    gat_student_losses_log["Total Loss"].append(loss.item())
    gat_student_losses_log["Task Loss"].append(task_loss.item())
    gat_student_losses_log["Distill Loss"].append(distill_loss.item())
    gat_student_losses_log["Feature Loss"].append(feature_loss.item())

    student_val_acc, student_test_acc = test_model(gat_student_model, data, device)
    gat_student_val_accs_log.append(student_val_acc)
    if epoch % 10 == 0:
        print(
            f"GAT Distill Epoch: {epoch:03d}, Lr: {scheduler_gat_student.get_last_lr()[0]:.6f}, Total Loss: {loss:.4f}, Task: {task_loss:.4f}, Distill: {distill_loss:.4f}, Feat: {feature_loss:.4f}, Val Acc: {student_val_acc:.4f}, Test Acc: {student_test_acc:.4f}"
        )

    if student_val_acc > best_gat_student_val_acc:
        best_gat_student_val_acc = student_val_acc
        gat_student_patience_counter = 0
    else:
        gat_student_patience_counter += 1
        if gat_student_patience_counter >= PATIENCE:
            print(f"Distilled GAT Student early stopping at epoch {epoch}")
            break

print("\n--- Distilled GAT Student Model Trained ---")
final_gat_student_val_acc, final_gat_student_test_acc = test_model(
    gat_student_model, data, device
)
print(
    f"Final Distilled GAT Student Val Acc: {final_gat_student_val_acc:.4f}, Final Distilled GAT Student Test Acc: {final_gat_student_test_acc:.4f}"
)


print("\n--- Model Parameter Counts & Sparsity ---")
teacher_params, teacher_sparsity = calculate_params_sparsity(teacher_model)
gcn_distilled_params, gcn_distilled_sparsity = calculate_params_sparsity(
    gcn_student_model
)
gcn_pruned_params, gcn_pruned_sparsity = calculate_params_sparsity(
    pruned_gcn_student_model
)
gat_distilled_params, gat_distilled_sparsity = calculate_params_sparsity(
    gat_student_model
)

print(
    f"Teacher (GCN):         Params={teacher_params: <8} Sparsity={teacher_sparsity:.4f}"
)
print(
    f"Distilled GCN Student: Params={gcn_distilled_params: <8} Sparsity={gcn_distilled_sparsity:.4f}"
)
print(
    f"Pruned GCN Student:    Params={gcn_pruned_params: <8} Sparsity={gcn_pruned_sparsity:.4f}"
)
print(
    f"Distilled GAT Student: Params={gat_distilled_params: <8} Sparsity={gat_distilled_sparsity:.4f}"
)

print("\n--- Measuring Inference Latency ---")
teacher_latency = measure_inference_time(
    teacher_model, data, device, num_runs=INFERENCE_RUNS, warmup_runs=INFERENCE_WARMUP
)
gcn_distilled_latency = measure_inference_time(
    gcn_student_model,
    data,
    device,
    num_runs=INFERENCE_RUNS,
    warmup_runs=INFERENCE_WARMUP,
)
gcn_pruned_latency = measure_inference_time(
    pruned_gcn_student_model,
    data,
    device,
    num_runs=INFERENCE_RUNS,
    warmup_runs=INFERENCE_WARMUP,
)
gat_distilled_latency = measure_inference_time(
    gat_student_model,
    data,
    device,
    num_runs=INFERENCE_RUNS,
    warmup_runs=INFERENCE_WARMUP,
)

print(f"\n--- Saving Training and Comparison Plots to '{PLOT_DIR}/' directory ---")

plot_losses(
    gcn_student_losses_log,
    "Distilled GCN Student Model Training Losses",
    filename="distilled_gcn_student_losses.png",
    plot_dir=PLOT_DIR,
)
plot_accuracy(
    gcn_student_val_accs_log,
    "Distilled GCN Student Model Validation Accuracy",
    filename="distilled_gcn_student_accuracy.png",
    plot_dir=PLOT_DIR,
)
if ft_val_accs_log:
    plot_accuracy(
        ft_val_accs_log,
        "Pruned GCN Student Fine-tuning Validation Accuracy",
        filename="pruned_gcn_student_finetune_accuracy.png",
        plot_dir=PLOT_DIR,
    )
plot_losses(
    gat_student_losses_log,
    "Distilled GAT Student Model Training Losses",
    filename="distilled_gat_student_losses.png",
    plot_dir=PLOT_DIR,
)
plot_accuracy(
    gat_student_val_accs_log,
    "Distilled GAT Student Model Validation Accuracy",
    filename="distilled_gat_student_accuracy.png",
    plot_dir=PLOT_DIR,
)

model_labels = [
    "Teacher\n(GCN)",
    "Distilled\nGCN",
    "Pruned\nGCN (FT)",
    "Distilled\nGAT",
]
accuracies = [
    final_teacher_test_acc,
    final_gcn_student_test_acc,
    final_pruned_gcn_test_acc,
    final_gat_student_test_acc,
]
latencies = [
    teacher_latency,
    gcn_distilled_latency,
    gcn_pruned_latency,
    gat_distilled_latency,
]
params = [teacher_params, gcn_distilled_params, gcn_pruned_params, gat_distilled_params]

plot_comparison_bar(
    accuracies,
    model_labels,
    "Model Test Accuracy Comparison",
    "Test Accuracy",
    filename="comparison_accuracy_with_gat.png",
    plot_dir=PLOT_DIR,
)
plot_comparison_bar(
    latencies,
    model_labels,
    "Model Inference Latency Comparison",
    "Avg Latency (ms)",
    filename="comparison_latency_with_gat.png",
    plot_dir=PLOT_DIR,
)
plot_comparison_bar(
    params,
    model_labels,
    "Model Parameter Count Comparison",
    "# Parameters",
    filename="comparison_params_with_gat.png",
    plot_dir=PLOT_DIR,
)

print("\n--- End of Script ---")
