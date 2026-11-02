# GCNMCG — Path-Attentive Spatio-Temporal GNN for MCG

## 目录结构

```
D:\pycharmproject\GCNMCG\
├── mcg_bad_channel_pipeline_revised.py   # 已有：坏道修复
├── node_edge.py                          # 已有：ROI 投影 + 5 阶段 + 静/动态边（不改）
├── all_data_labels.xlsx                  # ID / Label / CAG
├── data\
│   ├── health, NSTEMI, CAGNSTEMI, noNSTEMI  # 原始
│   ├── 坏道处理后数据\修复后数据\           # 修复后 NPZ
│   └── node_edge\节点边数据\               # 节点边 NPZ（node_edge.py 输出）
├── gnn_mcg\                              # 新增：模型和训练
│   ├── configs.py                        # 所有超参
│   ├── utils.py                          # 文件名解析、标签查表、CV 划分
│   ├── dataset.py                        # PyG Data + 时空图模板
│   ├── model.py                          # 路径枚举 + Path-Attentive GAT
│   ├── train.py                          # 分层 CV + 迁移训练
│   └── interpret.py                      # 路径原型 + 类别差异路径
├── run_e1.py                             # E1: 健康 vs NSTEMI（36ch）
└── run_e2.py                             # E2: NSTEMI vs 非 NSTEMI（64ch）
```

## 依赖

```
pip install torch torch-geometric numpy pandas scikit-learn scipy openpyxl matplotlib
```

## 三步端到端流程

### 1. 坏道修复
```
python mcg_bad_channel_pipeline_revised.py
```
输出到 `data\坏道处理后数据\修复后数据\`。**目前 noNSTEMI 只处理了 5 例，E2 需要先补跑所有 130 例。**

### 2. 节点边数据（node_edge.py，不改）
```
python node_edge.py
```
默认输出到 `data\node_edge\节点边数据\`。每份 NPZ 包含：
- `node_feature_blocks` [16, 5, 128, 3]：16 ROI × 5 阶段 × 128 采样 × 3 特征
- `edge_index`, `dynamic_edge_index`, `dynamic_edge_active/weight`
- `landmarks_samples`, `phase_bounds_samples`, `phase_duration_ms`

### 3. E1 主实验
```
python run_e1.py --device cuda
```
输出到 `results/e1_health_vs_nstemi/`：
- `fold_{i}/model.pt` `history.json` `val_preds.npz`
- `summary.json`
- `interpret/path_prototypes.json` `path_weight_matrix.npz`

### 4. E2 子实验
```
python run_e2.py --mode primary        # NSTEMI vs 非 NSTEMI 混合
python run_e2.py --mode nstemi-stemi   # 头对头
python run_e2.py --mode multiclass --subcats STEMI 心肌病  # 多分类
```

## 关键设计对应

| 方案里的东西 | 代码位置 |
|---|---|
| 16 ROI × 5 阶段 = 80 节点时空图 | `dataset.build_template` |
| 空间/时序/前向传递/动态四类边 | `dataset.NodeEdgeGraphDataset.__getitem__` |
| 候选路径枚举 | `model.enumerate_candidate_paths` |
| Node encoder (1D-CNN + AdaptivePool) | `model.NodeSignalEncoder` |
| GATv2 骨干（含 edge-type embedding） | `model.PathAttentiveGAT.gat_layers` |
| GRU 编码 + 稀疏路径 attention + L1 | `model.PathAttention` |
| 患者级分层 CV | `utils.stratified_patient_folds` |
| E1→E2 迁移（低 lr 骨干 + 头替换） | `train._load_pretrained` + `_split_optim_params` |
| Focal loss + 类别加权 | `train.FocalLoss` |
| 路径原型（top-K + 激活率） | `interpret._class_prototypes` |
| 类别差异路径（Welch-t + BH-FDR） | `interpret._class_differentiating_paths` |
| 路径稳定性（跨折 Jaccard） | `interpret.path_stability_across_folds` |
