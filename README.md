# 物理信息引导的回旋管电子枪等效热偏移代理模型

## Physics-Informed Surrogate Model for Equivalent Thermal Drift in Gyrotron Electron Guns

---

### 项目简介 / Project Abstract

本项目提出了等效热偏移(Equivalent Thermal Drift, ETD)概念框架及物理信息引导的两段式代理模型PI-ETD-Net,用于回旋管磁控注入枪(MIG)电子束参数的快速预测与设计优化。模型通过嵌入Busch定理、能量守恒等物理约束,在速度比、速度零散、导引中心半径和注入厚度四个关键输出上达到R^2 > 0.97的预测精度。

This project proposes the Equivalent Thermal Drift (ETD) framework and a physics-informed two-stage surrogate model (PI-ETD-Net) for rapid prediction and design optimization of gyrotron Magnetron Injection Gun (MIG) electron beam parameters. By embedding physical constraints (Busch's theorem, energy conservation), the model achieves R^2 > 0.97 across all four key outputs.

---

### 目录结构 / Directory Structure

```
MIG-Prediction/
├── README.md                  # 本文件 / This file
├── paper/
│   └── paper.md               # 完整中文学术论文 / Full Chinese academic paper
├── src/
│   ├── generate_dataset.py    # 数据集生成脚本 / Dataset generation (LHS sampling + physics response)
│   └── train_and_plot.py      # 模型训练与绘图 / Model training, evaluation, and plotting
├── data/
│   ├── cst_cold_state_dataset.csv  # 1500组CST仿真数据 / Synthetic CST simulation dataset
│   └── metrics_table.csv           # 模型对比指标 / Model comparison metrics
├── figures/
│   ├── fig05_input_distribution.png   # 输入参数LHS采样分布
│   ├── fig06_output_correlation.png   # 输出参数相关矩阵
│   ├── fig07_training_curves.png      # 训练曲线对比
│   ├── fig08_pred_vs_true.png         # 预测-真值散点图
│   ├── fig09_model_comparison.png     # 基线模型对比
│   ├── fig10_sobol.png                # Sobol灵敏度热力图
│   ├── fig11_etd_contour.png          # ETD偏移等高线
│   └── fig12_design_inversion.png     # 设计空间反演
└── framework/
    ├── fig01_gyrotron_mig.drawio      # 回旋管与MIG结构示意图
    ├── fig02_etd_concept.drawio       # ETD概念图
    ├── fig03_pi_etd_net_framework.drawio  # PI-ETD-Net框架图
    └── fig04_cold_surrogate_arch.drawio   # 冷态代理网络结构图
```

---

### 运行方法 / How to Run

#### 环境要求 / Requirements

- Python 3.8+
- NumPy, Pandas, Matplotlib, SciPy, scikit-learn

#### 生成数据 / Generate Dataset

```bash
python src/generate_dataset.py
```

生成 `data/cst_cold_state_dataset.csv`(1500组样本,11输入+4输出)。

#### 训练模型并生成图表 / Train Models and Generate Figures

```bash
python src/train_and_plot.py
```

输出所有8张数据图(fig05-fig12)至 `figures/` 目录,并生成 `data/metrics_table.csv` 模型对比表。

---

### 图表说明 / Figure Descriptions

| 图号 | 文件 | 内容描述 |
|------|------|----------|
| Fig.1 | framework/fig01_gyrotron_mig.drawio | 回旋管系统总体布局与MIG电子枪截面结构 |
| Fig.2 | framework/fig02_etd_concept.drawio | 冷态-真实态偏离来源与ETD统一概念 |
| Fig.3 | framework/fig03_pi_etd_net_framework.drawio | PI-ETD-Net两段式架构总览 |
| Fig.4 | framework/fig04_cold_surrogate_arch.drawio | 冷态代理网络详细结构(残差MLP+多任务头) |
| Fig.5 | figures/fig05_input_distribution.png | 11维输入空间LHS采样的边缘分布 |
| Fig.6 | figures/fig06_output_correlation.png | 4输出参数的相关矩阵与边缘分布 |
| Fig.7 | figures/fig07_training_curves.png | 含/不含物理损失的训练曲线对比 |
| Fig.8 | figures/fig08_pred_vs_true.png | 四输出的预测值vs真实值散点图 |
| Fig.9 | figures/fig09_model_comparison.png | 五种模型的R^2/MAPE对比柱状图 |
| Fig.10 | figures/fig10_sobol.png | Sobol一阶/全阶灵敏度指数热力图 |
| Fig.11 | figures/fig11_etd_contour.png | ETD偏移量随工况(V_a, I_b)的变化 |
| Fig.12 | figures/fig12_design_inversion.png | 目标alpha=1.4的可行设计域投影 |

---

### 引用 / Citation

如引用本工作,请使用以下格式:

```bibtex
@article{pi-etd-net-2024,
  title={Physics-Informed Surrogate Model for Equivalent Thermal Drift in Gyrotron Electron Guns},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024},
  note={Manuscript in preparation}
}
```

---

### 许可证 / License

本项目仅供学术研究使用。/ For academic research purposes only.
