# 毕业论文大纲

## 题目建议

基于多模态生理信号的心理工作负荷识别方法研究

可选题目：

- 基于 EDA 与 BVP 多模态融合的心理工作负荷识别研究
- 基于 NASA-TLX 聚类标签与深度学习的工作负荷识别方法研究
- 面向可穿戴生理信号的心理工作负荷识别与跨数据集验证

## 摘要

### 中文摘要

需要包含：

- 研究背景：心理工作负荷评估的重要性，可穿戴生理信号的优势。
- 研究问题：主观评分标签构造、多模态生理信号融合、跨被试泛化困难。
- 方法：基于 NASA-TLX 的 KMeans 标签构造，EDA/BVP 窗口化，LOSO 验证，CNN/ResNet/融合模型。
- 实验：UNIVERSE 主实验，传统机器学习与 RNN baseline，CNN 消融，窗口长度消融，MAUS 外部验证。
- 主要结果：30s/15s LateFusion ResNet binary 表现最好；MAUS 迁移验证显示跨数据集泛化仍具有挑战。
- 结论：多模态融合与短窗口对二分类工作负荷识别有效，但三分类和跨数据集泛化仍困难。

### 英文摘要

中文摘要完成后翻译。

## 第 1 章 绪论

### 1.1 研究背景

- 心理工作负荷的定义与应用场景。
- 传统主观量表与行为指标的局限。
- 可穿戴设备采集生理信号用于实时负荷识别的意义。
- EDA/GSR、BVP/PPG、ECG/HRV 等信号与认知负荷的关系。

### 1.2 研究意义

- 理论意义：探索主观负荷评分与生理信号之间的映射关系。
- 应用意义：为人机交互、驾驶、医疗监测、学习状态评估等场景提供参考。

### 1.3 研究难点

- 主观标签噪声较大。
- 不同被试生理差异明显。
- 多模态信号采样率和动态特征不同。
- 小样本条件下深度模型容易过拟合。
- 跨数据集泛化存在设备和任务范式差异。

### 1.4 本文主要工作

本文主要工作可以概括为：

1. 基于 NASA-TLX 主观评分构建工作负荷标签，并比较 1D 与 6D KMeans 标签质量。
2. 构建 EDA 与 BVP 生理信号窗口数据，并采用 LOSO 进行跨被试评估。
3. 比较传统机器学习、RNN、CNN、Early Fusion、Late Fusion 和 ResNet 模型。
4. 对窗口长度、融合方式、网络结构和训练策略进行消融实验。
5. 将模型流程迁移至 MAUS 数据集，进行外部数据集验证。

### 1.5 论文结构

简述各章节安排。

## 第 2 章 相关工作

### 2.1 心理工作负荷评估方法

- 主观量表：NASA-TLX。
- 行为指标。
- 生理信号指标。

### 2.2 生理信号与工作负荷识别

- EDA/GSR。
- BVP/PPG。
- ECG/HRV。
- 多模态融合。

### 2.3 机器学习与深度学习方法

- 传统机器学习：RF、SVM、XGBoost、MLP。
- 时序模型：LSTM、GRU。
- CNN/FCN/ResNet。
- 注意力机制。

### 2.4 跨被试与跨数据集泛化

- LOSO 验证。
- 个体差异问题。
- 跨数据集迁移验证的挑战。

### 2.5 本章小结

## 第 3 章 数据集与预处理方法

### 3.1 UNIVERSE 数据集

说明：

- 被试数量。
- 实验任务。
- 采集模态：EDA、BVP 等。
- NASA-TLX 标签。
- 本文使用 Lab1/Lab2 和相关任务。

### 3.2 MAUS 数据集

说明：

- 22 名被试。
- N-back 任务。
- 信号：ECG、GSR、指尖 PPG、腕式 PPG。
- NASA-TLX 与 PSQI。
- 本文使用 GSR 与 PPG，作为 EDA/BVP 的对应模态。

### 3.3 NASA-TLX 标签构造

#### 3.3.1 UNIVERSE 标签构造

- 基于 Weighted NASA Score 的 1D KMeans。
- 基于 NASA-TLX 六维特征的 6D KMeans。
- low/mid/high 三分类标签。
- binary 标签：保留 low/high 或 KMeans-2。

#### 3.3.2 MAUS 标签构造

- 使用 NASA_TLX.csv 最后一行 Adjusted rate 作为连续工作负荷分数。
- 对 trial-level NASA score 做 KMeans-2。
- 低中心为 low，高中心为 high。

### 3.4 信号窗口化

#### 3.4.1 UNIVERSE 窗口化

- EDA：4Hz。
- BVP：64Hz。
- 窗口设置：30s/15s、60s/30s、90s/30s、120s/60s。
- 主要模型使用 30s/15s。

#### 3.4.2 MAUS 窗口化

- 原始 GSR/PPG 读取。
- 30s window、15s hop。
- GSR 重采样到 120 点。
- PPG 重采样到 1920 点。
- 与 UNIVERSE 30s/15s 输入长度对齐。

### 3.5 数据标准化

- fold 内标准化，避免测试数据泄漏。
- 对不同模态分别标准化。
- 异常窗口过滤。

### 3.6 评价指标

- Macro-F1。
- Balanced Accuracy。
- Confusion Matrix。
- kept/skipped LOSO folds。

### 3.7 本章小结

## 第 4 章 方法设计

### 4.1 整体流程

建议画 pipeline 图：

```text
原始/预处理生理信号
    ↓
NASA-TLX 标签构造
    ↓
滑动窗口切分
    ↓
LOSO 训练与测试
    ↓
模型预测
    ↓
Macro-F1 / BalAcc / Confusion Matrix
```

### 4.2 传统机器学习模型

- RF。
- XGBoost。
- MLP。
- EDA/HRV/EDA+HRV 特征。

### 4.3 RNN 模型

- LSTM。
- GRU。
- EDA、BVP、EDA+BVP。

### 4.4 CNN 单模态模型

- EDA CNN。
- BVP CNN。
- 1D convolution block。
- class weight。

### 4.5 多模态融合模型

#### 4.5.1 Early Fusion CNN

- EDA 重采样到 BVP 长度。
- 通道拼接。
- 统一 CNN 编码。

#### 4.5.2 Late Fusion CNN

- EDA 分支。
- BVP 分支。
- 特征 concat。
- 分类头。

#### 4.5.3 Late Fusion ResNet

- EDA ResNet branch。
- BVP ResNet branch。
- 残差连接对长序列训练的作用。

### 4.6 注意力池化拓展模型

- 用 Attention Pooling 替代 Adaptive Average Pooling。
- 目的：学习关键时间片段权重。
- 实验结果用于拓展分析，不作为最终主模型。

### 4.7 训练策略

- LOSO。
- train/validation split 使用 GroupShuffleSplit。
- Cross Entropy / Focal Loss。
- Learning rate scheduler。
- Early stopping。
- Data augmentation。
- Subject-balanced sampler。

### 4.8 本章小结

## 第 5 章 实验设计与结果分析

### 5.1 实验设置

- 硬件与软件环境。
- 随机种子。
- 训练 epoch、batch size、learning rate。
- 路径与结果文件说明。

### 5.2 标签聚类质量分析

需要展示：

- 1D KMeans vs 6D KMeans。
- silhouette、ARI、NMI 等。
- 聚类可视化图。
- 结论：1D Weighted NASA Score 聚类更稳定，或说明两者差异。

### 5.3 传统机器学习 baseline

表格：

| 模型 | 模态 | 任务 | Macro-F1 | BalAcc |
|---|---|---|---:|---:|

分析：

- binary 优于 3-class。
- EDA/HRV/EDA+HRV 哪个更好。

### 5.4 RNN baseline

表格：

| 模型 | 模态 | 任务 | Macro-F1 | BalAcc |
|---|---|---|---:|---:|

分析：

- LSTM/GRU 表现。
- 与传统机器学习比较。

### 5.5 CNN 主实验

使用 `results/cnn_collected/cnn_summary_all.csv`。

表格：

| 模型 | 融合方式 | 模态 | 任务 | Macro-F1 | BalAcc |
|---|---|---|---|---:|---:|

需要包含：

- EDA CNN。
- BVP CNN。
- Early Fusion CNN。
- Late Fusion CNN。
- Late Fusion ResNet。

主要结论：

- binary 明显优于 3-class。
- 多模态融合在 binary 上有效。
- ResNet 对 binary 略有收益。

### 5.6 窗口长度消融实验

使用 `results/cnn_window_ablation/summary_all.csv`。

窗口：

- 30s/15s。
- 60s/30s。
- 90s/30s。
- 120s/60s。

结论：

- 30s/15s + LateFusion ResNet binary 表现最好。
- 长窗口减少样本数，120s/60s 导致更多 skipped folds。

### 5.7 训练策略优化实验

使用 `results/cnn_resnet_binary_optimization/summary_all.csv`。

比较：

- baseline。
- lr_3e4。
- dropout_035。
- focal_g1。
- augment。
- balanced_sampler。
- final_combo。

当前结论：

- augmentation 的 Macro-F1 最高。
- 组合策略未必带来进一步提升。
- 训练技巧提升有限，说明跨被试差异和标签噪声是主要瓶颈。

### 5.8 注意力机制拓展实验

使用 `results/cnn_attention_binary/summary_all.csv`。

比较：

- avg pooling baseline。
- attention pooling。

当前结论：

- attention pooling 没有提升，反而低于平均池化。
- 可能原因：30s 短窗口下全局平均池化已经足够稳定，注意力引入额外参数导致过拟合。

### 5.9 MAUS 外部数据集验证

使用：

```text
results/maus_latefusion_resnet_binary_loso/
```

需要说明：

- MAUS 标签构造。
- GSR/PPG 与 EDA/BVP 对齐。
- LOSO 结果。

当前结果：

- Macro-F1：0.420 ± 0.102。
- BalAcc：0.520 ± 0.087。
- kept folds：16/22。
- skipped folds：6/22。

分析：

- 模型在 MAUS 上略高于随机，但效果有限。
- 跨数据集/跨被试泛化困难。
- 原因包括设备差异、任务范式差异、标签分布差异、部分 subject 单类别。

### 5.10 错误分析

建议包含：

- confusion matrix。
- high workload recall 较弱。
- 3-class 中 mid 类容易混淆。
- subject-level 差异明显。
- skipped folds 原因。

### 5.11 本章小结

## 第 6 章 总结与展望

### 6.1 工作总结

总结本文完成：

- NASA-TLX 标签构造。
- 多模态生理信号窗口化。
- 传统 ML、RNN、CNN 对比。
- LateFusion ResNet 模型。
- 窗口长度与训练策略消融。
- 注意力机制拓展。
- MAUS 外部验证。

### 6.2 主要结论

建议写：

1. 基于 NASA-TLX 的二分类工作负荷识别比三分类更稳定。
2. EDA+BVP 多模态融合优于部分单模态设置。
3. 30s/15s 短窗口更适合当前 CNN/ResNet 模型。
4. ResNet 在 binary 任务上有一定优势。
5. 训练策略优化带来有限提升，数据增强略有帮助。
6. 注意力池化未显著改善结果。
7. MAUS 外部验证表明跨数据集泛化仍然困难。

### 6.3 不足

- 数据规模有限。
- 主观标签存在噪声。
- 部分 subject 类别不完整。
- 跨数据集设备与任务差异较大。
- 当前模型没有充分解决个体差异。

### 6.4 未来工作

- 个体自适应/领域自适应。
- 更多数据集验证。
- 更稳健的标签构造。
- 自监督预训练。
- 更细粒度的时间注意力或生理特征解释。

## 参考文献

初步建议引用方向：

- NASA-TLX 原始论文。
- UNIVERSE 数据集论文/说明。
- MAUS 数据集论文。
- 生理信号 workload/stress recognition 相关综述。
- CNN/ResNet/TCN/RNN 用于 time-series classification 的论文。
- LOSO/cross-subject validation 相关工作。

## 附录

### 附录 A 主要实验结果路径

```text
results/cnn_collected/cnn_summary_all.csv
results/cnn_collected/cnn_per_subject_all.csv
results/cnn_window_ablation/summary_all.csv
results/cnn_resnet_binary_optimization/summary_all.csv
results/cnn_attention_binary/summary_all.csv
results/maus_latefusion_resnet_binary_loso/loso_results_kept.csv
```

### 附录 B 主要脚本路径

```text
universe/preprocess/eda_build_windows.py
universe/preprocess/bvp_build_windows.py
universe/train/cnn_eda_bvp_latefusion_loso.py
universe/train/cnn_eda_bvp_latefusion_resnet_loso.py
universe/train/cnn_eda_bvp_latefusion_resnet_binary_optimized.py
universe/train/cnn_eda_bvp_latefusion_resnet_attention_binary.py
maus/build_windows_kmeans_binary.py
maus/train_latefusion_resnet_binary_loso.py
```
