mamba create -n gnn python=3.9 -y
source activate gnn
module load cuda-11.8.0-gcc-12.1.0
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_cluster-1.6.3%2Bpt21cu118-cp39-cp39-linux_x86_64.whl
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_scatter-2.1.2%2Bpt21cu118-cp39-cp39-linux_x86_64.whl
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_sparse-0.6.18%2Bpt21cu118-cp39-cp39-linux_x86_64.whl
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_spline_conv-1.2.2%2Bpt21cu118-cp39-cp39-linux_x86_64.whl
pip install torch_geometric


# To run naive KD
cd naive_kd
python main.py

# To run enhanced KD
cd enhanced_kd
python main.py

Naive Knowledge Distillation (in naive_kd directory):
Targets the Cora dataset.
Uses GCN for the teacher model, and smaller GCN and GAT models for students.
Distillation involves matching task performance, output logits, and hidden features (with linear projection). Also includes model pruning.

Enhanced Knowledge Distillation (in enhanced_kd directory):
Targets the large-scale ogbn-arxiv dataset.
Uses Graph Transformer models for both teacher and student.
Employs advanced distillation: normalized feature matching (with MLP projection), relational knowledge (similarity matrices), attention pattern transfer, and contrastive learning.
Features progressive loss weighting and robust training techniques for large graphs (neighbor sampling).