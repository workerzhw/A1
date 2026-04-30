# A1 软件架构说明文档

> A1: 全透明、开源、自适应且高效的截断式视觉-语言-动作模型（Vision-Language-Action, VLA）

---

## 1. 项目概述

A1 是一个面向机器人操作任务的研究级 **视觉-语言-动作（VLA）** 模型。项目基于 Allen AI 的 Molmo 多模态大语言模型扩展而来，增加了机器人动作预测能力。系统接收视觉（RGB 图像）和语言（文本指令）输入，输出机器人可执行的动作序列。

**核心能力：**
- 支持多种 LLM 骨干网络（Qwen2、Qwen3、OLMoE）
- 支持多种视觉编码器（ViT-L/14、SigLIP、DINOv2、MetaCLIP）
- 多种动作预测头（L1 回归、Diffusion、Flow Matching）
- 仿真与真实机器人评估（LIBERO、VLABench、RoboChallenge）
- 分布式训练（FSDP）与 API 推理服务部署

---

## 2. 顶层目录结构

```
A1/
├── a1/                        # 核心 Python 包（可安装）
│   ├── config.py              # 全局配置数据类
│   ├── model.py               # 基础 Molmo VLM 模型
│   ├── train.py               # 训练基础设施（VLATrainer）
│   ├── tokenizer.py           # 分词器封装
│   ├── torch_util.py          # PyTorch 工具函数
│   ├── util.py                # 通用工具
│   ├── optim.py               # 优化器
│   ├── version.py             # 版本号（0.1.1）
│   ├── vla/                   # VLA 模型组件
│   │   ├── affordvla.py       # 核心 AffordVLA 模型
│   │   ├── affordvla_early_exit.py  # 早退推理变体
│   │   ├── action_heads.py    # 动作预测头实现
│   │   ├── projectors.py      # 本体感觉/噪声动作投影器
│   │   ├── dit/               # DiT（扩散变换器）模块
│   │   ├── constants.py       # YAML 配置懒加载常量
│   │   ├── config_loader.py   # YAML 配置解析器
│   │   ├── value_net.py       # 早退价值网络
│   │   └── model_selector.py  # 模型选择器
│   ├── data/                  # 数据集加载与预处理
│   │   ├── A1_datasets.py     # 自定义预训练数据集
│   │   ├── dataset.py         # 通用数据集接口
│   │   ├── collator.py        # 数据批处理
│   │   ├── data_formatter.py  # 数据格式化
│   │   └── iterable_dataset_mixture.py  # 可迭代数据混合
│   ├── hf_datasets/           # HuggingFace 数据集适配器
│   └── eval/                  # 评估工具
├── configs/                   # YAML 配置文件
│   ├── experiments/           # 实验配置（组合模型+数据集）
│   ├── models/                # 模型配置（动作维度等）
│   └── datasets/              # 数据集配置（路径、归一化、增强）
├── deploy/                    # 部署/推理服务
│   ├── api_server.py          # FastAPI 推理服务器
│   ├── api_client.py          # API 推理客户端
│   ├── infer_vla.py           # 推理逻辑（模型加载、归一化）
│   └── deploy.sh              # 部署启动脚本
├── launch_scripts/            # 训练/评估入口脚本
│   ├── train_vla.py           # 主训练启动脚本
│   ├── utils.py               # 模型配置定义
│   └── eval_*.py              # 评估启动脚本
├── scripts/                   # 工具脚本
│   ├── train_for_action.py    # 核心训练循环
│   └── slurms/                # Slurm 集群提交脚本
├── robot_experiments/         # 机器人评估脚本
│   ├── libero/                # LIBERO 仿真评估
│   ├── vlabench/              # VLABench 仿真评估
│   └── RoboChallengeInference/ # 真实机器人评估
├── aex/                       # 架构文档
├── pyproject.toml             # Python 包定义
├── requirements.txt           # 额外依赖
├── .env.example               # 环境变量模板
├── train_*.sh / eval_*.sh     # 训练/评估 Shell 脚本
└── README.md                  # 项目文档
```

---

## 3. 系统架构

### 3.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                       应用层（Application）                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ 训练脚本  │  │ 评估脚本  │  │ API 服务  │  │ 集群提交  │    │
│  │train_*.sh │  │ eval_*.sh │  │api_server│  │  slurms  │    │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └──────────┘    │
│        │             │             │                          │
│  ┌─────┴─────────────┴─────────────┴───────────────────┐    │
│  │              launch_scripts / deploy                 │    │
│  │         （入口脚本层：参数解析、配置构建）               │    │
│  └─────────────────────┬───────────────────────────────┘    │
├────────────────────────┼────────────────────────────────────┤
│                        ▼                                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 核心库层（a1 包）                      │   │
│  │                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  │   │
│  │  │  VLATrainer   │  │   AffordVLA   │  │  数据管线  │  │   │
│  │  │  (train.py)   │  │ (affordvla.py)│  │  (data/)  │  │   │
│  │  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  │   │
│  │         │                 │                  │        │   │
│  │  ┌──────┴─────────────────┴──────────────────┴─────┐  │   │
│  │  │             配置管理（config.py + YAML）          │  │   │
│  │  └─────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      基础设施层                               │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐  │
│  │  PyTorch  │  │ HF Trans- │  │  FastAPI   │  │  FSDP    │  │
│  │  + CUDA   │  │  formers  │  │  Uvicorn   │  │ 分布式   │  │
│  └──────────┘  └───────────┘  └───────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 模型推理流水线

```
输入                          编码                          融合                        输出
─────                        ──────                        ──────                      ──────

┌──────────┐          ┌──────────────┐
│ RGB 图像  │ ───────▶ │  视觉编码器   │
│ (多视角)  │          │  ViT-L/14    │
└──────────┘          │  SigLIP      │
                      │  DINOv2      │        ┌───────────────────┐
                      └──────┬───────┘        │                   │
                             │                │  LLM 骨干网络      │
                      ┌──────▼───────┐        │  (Qwen2/Qwen3/    │
                      │ 视觉-语言     │ ─────▶ │   OLMoE)          │
                      │ 连接器       │        │                   │
                      │ (注意力池化   │        │  文本+视觉+本体    │
                      │  + MLP投影)  │        │  感觉Token融合     │
                      └──────────────┘        └────────┬──────────┘
                                                       │
┌──────────┐          ┌──────────────┐                  │
│ 文本指令  │ ───────▶ │  分词器       │ ───────────────▶│
└──────────┘          │  Tokenizer   │                  │
                      └──────────────┘                  │
                                                       │
┌──────────┐          ┌──────────────┐                  │
│ 机器人本体│ ───────▶ │ 本体感觉     │ ───────────────▶│
│ 感觉状态  │          │ 投影器       │                  │
│ (关节位置)│          │ (MLP)        │                  │
└──────────┘          └──────────────┘                  │
                                                       ▼
                                              ┌──────────────────┐
                                              │   动作预测头       │
                                              │ ┌──────────────┐ │
                                              │ │ L1 回归头     │ │
                                              │ ├──────────────┤ │
                                              │ │ Diffusion头  │ │
                                              │ ├──────────────┤ │
                                              │ │ Flow Matching│ │──▶ 动作序列
                                              │ │ (推荐/默认)  │ │    (7-DoF)
                                              │ └──────────────┘ │
                                              └──────────────────┘
```

---

## 4. 核心模块详细说明

### 4.1 模型模块（a1/vla/）

#### 4.1.1 AffordVLA（affordvla.py）

核心 VLA 模型类，继承自 Molmo 基类。负责：
- 整合视觉编码器、LLM 骨干和动作预测头
- 管理本体感觉信息注入（通过 `ProprioProjector`）
- 提供 FSDP 包装策略（按模块/按参数大小分片）
- 支持早退推理变体（`AffordVLAEarlyExit`）

#### 4.1.2 动作预测头（action_heads.py）

| 动作头 | 类名 | 机制 |
|--------|------|------|
| L1 回归 | `L1RegressionActionHead` | 从 LLM 隐藏状态直接回归动作值 |
| Diffusion (DiT) | `DiffusionTransformerActionHead` | 使用 DiT 迭代去噪生成动作 |
| Diffusion (OpenVLA) | `DiffusionActionHead` | 将噪声动作和时间步嵌入注入 LLM 序列 |
| Flow Matching | `FlowMatchingActionHead` | 独立专家 Transformer 交叉注意力到 LLM KV-Cache，通过 Euler 积分预测向量场（推荐） |

#### 4.1.3 投影器（projectors.py）

- **ProprioProjector**：将机器人本体感觉状态（关节位置）通过 MLP 投影到 LLM 嵌入空间
- **NoisyActionProjector**：将噪声动作投影到嵌入空间，用于 Diffusion 动作头

#### 4.1.4 DiT 模块（dit/）

Diffusion Transformer 实现模块，包含：
- DiT Block（自注意力 + 交叉注意力 + FFN）
- DiT Model（多层 DiT Block 堆叠）

### 4.2 基础模型（a1/model.py）

`Molmo` 基类，实现完整的视觉-语言模型：
- **视觉编码器集成**：支持多种 ViT 变体，多裁剪策略
- **注意力池化**：将视觉特征池化为固定长度序列
- **LLM Transformer**：完整的 Transformer 解码器（RoPE 位置编码、GQA 注意力）
- **KV-Cache 支持**：支持增量推理的 KV-Cache 管理
- **非因果注意力**：支持双向注意力用于动作预测

### 4.3 训练模块（a1/train.py）

`VLATrainer` 类，提供完整训练基础设施：

```
VLATrainer
├── FSDP 分布式训练管理
│   ├── 按模块分片策略（LLM、ViT、Connector）
│   ├── 按参数大小分片策略
│   └── 混合精度训练（BF16）
├── 训练循环
│   ├── 梯度累积
│   ├── 梯度裁剪
│   └── 学习率调度
├── 损失计算
│   ├── 语言建模损失
│   ├── 动作预测损失（独立权重）
│   └── 状态掩码（本体感觉鲁棒性）
├── 检查点管理
│   ├── 周期性保存
│   ├── 最佳模型追踪
│   └── 断点续训
└── 评估循环
    ├── 周期性评估
    └── WandB 指标记录
```

### 4.4 数据模块（a1/data/）

| 文件 | 功能 |
|------|------|
| `A1_datasets.py` | 自定义预训练数据集（ShareRobot、DROID、RoboVQA、AGD20k 等） |
| `dataset.py` | 通用数据集接口 |
| `collator.py` | 数据批处理（padding、attention mask 生成） |
| `data_formatter.py` | 数据格式标准化 |
| `iterable_dataset_mixture.py` | 多数据集混合采样 |

数据流：
```
原始数据（TFRecord/RLDS/图像目录）
    │
    ▼
Dataset（数据集类，解析+预处理）
    │
    ▼
Collator（批处理+padding）
    │
    ▼
DataLoader（分布式采样）
    │
    ▼
VLATrainer（训练循环）
```

### 4.5 配置模块

#### 4.5.1 配置数据类（a1/config.py）

提供全面的配置数据类体系：

```
配置体系
├── ModelConfig         # 模型架构配置（LLM、ViT、动作头）
├── TrainConfig         # 训练超参数
├── DataConfig          # 数据集配置
├── OptimizerConfig     # 优化器参数
├── SchedulerConfig     # 学习率调度
├── FSDPConfig          # FSDP 分布式配置
├── EvalConfig          # 评估配置
└── DiTActionConfig     # DiT/Flow Matching 动作头配置
```

#### 4.5.2 YAML 配置系统（configs/）

```
configs/
├── experiments/              # 实验配置（引用模型+数据集配置）
│   ├── pretrain.yaml         # 预训练
│   ├── libero_simulation.yaml
│   ├── vlabench.yaml
│   └── dobot.yaml
├── models/                   # 模型配置
│   ├── pretrain.yaml
│   ├── libero.yaml           # 固定动作维度、动作块大小
│   └── vlabench.yaml
└── datasets/                 # 数据集配置
    ├── pretrain.yaml
    ├── libero_4_tasks.yaml   # 路径、归一化方式、数据增强
    └── vlabench.yaml
```

配置加载流程：`实验YAML → 引用模型YAML + 数据集YAML → 合并为运行时配置`

### 4.6 部署模块（deploy/）

#### 4.6.1 API 服务（api_server.py）

基于 FastAPI 的推理服务：

```
┌──────────────────────────────────────────┐
│            FastAPI Server (port 6789)    │
│                                          │
│  GET  /health         → 健康检查          │
│  POST /inference      → 动作预测          │
│  GET  /model_info     → 模型信息          │
│                                          │
│  请求格式：                               │
│  {                                       │
│    "image": "base64编码图像",             │
│    "instruction": "文本指令",             │
│    "proprio": [关节位置数组]              │
│  }                                       │
│                                          │
│  响应格式：                               │
│  {                                       │
│    "actions": [[动作序列]],               │
│    "status": "success"                   │
│  }                                       │
└──────────────────────────────────────────┘
```

#### 4.6.2 推理逻辑（infer_vla.py）

- 模型加载（支持 BF16/FP32）
- 输入预处理（图像 resize、归一化）
- 动作归一化/反归一化
- 本体感觉归一化

---

## 5. 支持的模型与硬件

### 5.1 LLM 骨干网络

| 模型 | 参数量 | 配置名称 |
|------|--------|----------|
| Qwen2 | 1.5B / 7B / 72B | `QWEN2_7B` 等 |
| Qwen3 | 1.7B / 4B / 8B | `QWEN3_1_7B`, `QWEN3_4B`, `QWEN3_8B` |
| OLMoE | 1B-7B (MoE) | `OLMOE` |

### 5.2 视觉编码器

| 编码器 | 来源 | 特点 |
|--------|------|------|
| ViT-L/14 | OpenAI CLIP | 通用视觉特征 |
| SigLIP | Google | Sigmoid 注意力 |
| DINOv2-Large | Meta | 自监督视觉特征 |
| MetaCLIP-L14 | Meta | 元学习 CLIP |

### 5.3 训练基础设施

- **分布式策略**：PyTorch FSDP（Fully Sharded Data Parallel）
- **多节点多 GPU**：通过 torchrun 启动
- **集群支持**：Slurm 作业调度（`scripts/slurms/`）
- **混合精度**：BF16（默认），可选 FP32（AMD GPU 兼容）
- **实验追踪**：WandB 集成

---

## 6. 评估体系

### 6.1 仿真评估

| 评估环境 | 脚本位置 | 说明 |
|----------|----------|------|
| LIBERO | `robot_experiments/libero/` | 4 个任务套件（spatial、object、goal、libero_10） |
| VLABench | `robot_experiments/vlabench/` | 通用仿真评估 |

### 6.2 真实机器人评估

| 机器人平台 | 脚本位置 | 说明 |
|------------|----------|------|
| ALOHA | `robot_experiments/RoboChallengeInference/` | 双臂操作 |
| ARX5 | 同上 | 机械臂 |
| UR5 | 同上 | 工业协作臂 |
| FRANKA | 同上 | Franka Emika |

### 6.3 评估方式

所有评估均采用 **Client-Server** 架构：
- 服务端：加载模型，启动 API Server
- 客户端：连接仿真/真实环境，发送观测数据，接收动作指令

---

## 7. 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 动作头默认选择 | Flow Matching | 更好的动作生成质量，独立专家 Transformer 不增加 LLM 计算负担 |
| 分布式训练 | FSDP（非 DeepSpeed） | 更好的内存效率和与 PyTorch 生态兼容性 |
| 配置系统 | YAML + Dataclass | 兼顾可读性和类型安全 |
| 推理部署 | FastAPI REST API | 轻量、标准化、易于与仿真/真实环境集成 |
| 早退机制 | Value Network | 通过价值网络判断 LLM 何时可以提前退出，平衡速度与精度 |
| 非因果注意力 | 支持双向注意力 | 动作预测任务中，双向注意力比因果注意力效果更好 |

---

## 8. 技术栈总结

```
┌─────────────────────────────────────────────┐
│                编程语言                       │
│  Python >= 3.10                              │
├─────────────────────────────────────────────┤
│                深度学习框架                    │
│  PyTorch >= 2.3.1 + CUDA 12.4               │
│  HuggingFace Transformers >= 4.37.1          │
│  FSDP (内置)                                 │
├─────────────────────────────────────────────┤
│                配置管理                       │
│  OmegaConf + YAML + Python Dataclass         │
├─────────────────────────────────────────────┤
│                推理服务                       │
│  FastAPI + Uvicorn                           │
├─────────────────────────────────────────────┤
│                实验管理                       │
│  WandB + TensorBoard                         │
├─────────────────────────────────────────────┤
│                数据处理                       │
│  TensorFlow Datasets (RLDS) + LeRobot 0.3.3  │
├─────────────────────────────────────────────┤
│                仿真环境                       │
│  LIBERO + VLABench (Git Submodules)          │
├─────────────────────────────────────────────┤
│                集群调度                       │
│  Slurm + torchrun                            │
└─────────────────────────────────────────────┘
```

---

## 9. 模块依赖关系

```
                     ┌──────────┐
                     │ pyproject│
                     │  .toml   │
                     └────┬─────┘
                          │ pip install -e .
                          ▼
┌─────────────────────────────────────────────────┐
│                    a1 包                          │
│                                                  │
│  config.py ◄──── vla/constants.py                │
│      │                  │                        │
│      ▼                  ▼                        │
│  model.py ◄──── vla/affordvla.py                 │
│      │                  │                        │
│      │           vla/action_heads.py              │
│      │                  │                        │
│      │           vla/projectors.py                │
│      │                  │                        │
│      │              vla/dit/                      │
│      │                                           │
│      ▼                                           │
│  train.py (VLATrainer)                           │
│      ▲                                           │
│      │                                           │
│  data/ (Dataset, Collator)                       │
│                                                  │
│  eval/ (评估工具)                                 │
└─────────────────────────────────────────────────┘
         │              │              │
         ▼              ▼              ▼
  launch_scripts/   deploy/     robot_experiments/
  (训练入口)        (推理服务)    (评估脚本)
```

---

*文档生成日期：2026-04-30*
*A1 版本：0.1.1*
