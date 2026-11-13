# FinFET 漏极电流代理模型（finfet-ids-surrogate）

[English](README.md) · **简体中文**

一个小型 PyTorch 研究：基于 BSIM-CMG 仿真数据学习 15 nm 级 n 型 FinFET 的漏极电流，并检验普通神经网络代理模型在哪些区域可靠、哪些区域不可靠。

输入为鳍数、沟道长度、工艺角、温度、V<sub>ds</sub> 与 V<sub>gs</sub>，输出为 I<sub>ds</sub>。

## 背景

本项目源自 2025 年 7 月上海交通大学暑期项目 *Emerging Semiconductor Devices and Their Sustainable Innovations* 中的双人课题（Shun Hao Pang 与 Jiayi Zhu）。同组还研读了基于深度学习的 BSIM-CMG 参数提取工作（Kao et al., *IEEE Trans. Electron Devices* 69, 4765, 2022）。

| 文件夹 | 内容 |
| --- | --- |
| [`original/`](original/) | 2025 年 7 月的训练脚本及其三张输出图，保持原样 |
| [`src/finfet_ids_mlp.py`](src/finfet_ids_mlp.py) | 整理后的版本（2026）：网络结构与训练设置不变，增加固定随机种子、多种训练/测试划分及留出集指标 |
| [`results/`](results/) | 各划分与各种子的指标、汇总及留出集图像 |
| [`data/`](data/README.md) | 数据格式说明；仿真数据文件本身不再分发 |

## 模型

- MLP：6 → 100 → 50 → 20 → 1，ReLU 激活，前两个隐藏层使用批归一化与 dropout（0.1）
- 对标准化后的 I<sub>ds</sub> 使用 Huber（smooth L1）损失，Adam 优化器（lr 1e-3，权重衰减 1e-5），余弦退火学习率，早停（patience 50）

## 结果

测试集指标为三个种子（0、1、2）的平均值。MAPE 仅对 |I<sub>ds</sub>| > 1 µA 的点计算。

| 划分 | 留出内容 | 测试点数 | R²（均值，范围） | RMSE (µA) | 1 µA 以上 MAPE |
| --- | --- | --- | --- | --- | --- |
| `ordered` | 文件末尾 10 %，与 2025 年脚本一致 | 113 | 0.996 (0.996–0.998) | 32 | 54 % |
| `random-points` | 随机 10 % 偏置点 | 113 | 0.997 (0.996–0.998) | 23 | 78 % |
| `random-curves` | 随机 10 % 完整 Id–Vg 扫描曲线 | 108 | 0.997 (0.995–0.999) | 22 | 75 % |
| `holdout-temp` | 125 °C 下的全部扫描（温度外推） | 225 | 0.988 (0.982–0.991) | 46 | 87 % |

<p align="center">
<img src="results/figures/idvg_random-curves.png" width="720" alt="线性与对数坐标下的留出集 Id-Vg 曲线">
</p>

**结论**

1. **阈值以上代理模型准确。** 线性坐标下预测曲线与 BSIM-CMG 高度吻合，未见过的扫描曲线 R² 仍高于 0.99。
2. **阈值以下失效。** 对数坐标下预测值偏差可达数个数量级，甚至出现负值。损失函数基于线性电流计算，接近零的电流几乎没有权重；高 R² 掩盖了这一问题，而较大的相对误差将其暴露出来。
3. **温度外推会损失精度。** 留出全部 125 °C 数据后 R² 下降，误差约翻倍。

第一点解释了 2025 年原始结果图为何看起来令人信服；第二点是主要局限。对亚阈值区而言，改用对数电流或 arcsinh 目标，或引入基于物理的输出层，是自然的下一步。

**注意。** 与原始脚本相同，早停监控的是留出集，因此测试指标略偏乐观。

## 复现

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# 将 N15A-nmos1.json 放入 data/（见 data/README.md）
for s in ordered random-points random-curves holdout-temp; do
  for seed in 0 1 2; do python src/finfet_ids_mlp.py --data data/N15A-nmos1.json --split $s --seed $seed; done
done
```

在 CPU 上每次运行仅需数秒。
