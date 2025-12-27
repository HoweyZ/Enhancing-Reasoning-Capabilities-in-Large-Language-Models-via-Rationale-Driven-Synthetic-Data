# MiniPromptCoT

## 简介

MiniPromptCoT 是一个 LLM reasoning 框架，通过合成高质量 prompts 来提升模型在数学和编程任务上的推理能力。

## 核心功能

### 数据生成模块
- **概念编码** (`concept_encoder.py`): 将数学概念转换为向量表示
- **概念采样** (`concept_sampler.py`): 基于向量相似度采样概念组合
- **问题生成** (`problem_generator.py`): 使用 LLM 生成数学/编程问题
- **测试用例生成** (`test_case_generator.py`): 为编程问题生成测试用例
- **轨迹收集** (`trajectory_collector.py`): 收集模型的解答轨迹
- **结果评估** (`evaluator.py`): 评估解答的正确性

### 模型训练模块
- **SFT 训练** (`sft_trainer.py`): 使用合成数据进行监督微调
- **Self-Play 训练** (`selfplay_trainer.py`): 基于自生成数据进行 DPO 训练

## 安装

```bash
# 克隆项目
git clone <repo_url>
cd MiniPromptCoT

# 创建虚拟环境
conda create -n minipromptcot python=3.10
conda activate minipromptcot

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 数据生成

```bash
# 运行完整数据生成流程
bash scripts/run_data_gen.sh

# 或者分步运行
python -m MiniPromptCoT.data_generation.concept_encoder \
    --input data/concepts.jsonl \
    --output output/concept_embeddings.jsonl

python -m MiniPromptCoT.data_generation.concept_sampler \
    --input data/concepts.jsonl \
    --embed output/concept_embeddings.jsonl \
    --output output/concept_pairs.jsonl

python -m MiniPromptCoT.data_generation.problem_generator \
    --input output/concept_pairs.jsonl \
    --output output/problems.jsonl
```

### 2. SFT 训练

```bash
# 运行 SFT 训练
bash scripts/run_sft.sh data/sft_training.jsonl

# 或使用自定义数据
python -m MiniPromptCoT.training.sft_trainer \
    --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
    --train_data_path "data/sft_training.jsonl" \
    --output_dir "sft_output" \
    --use_lora true
```

### 3. Self-Play 训练

```bash
# 运行 Self-Play 训练
bash scripts/run_selfplay.sh data/selfplay_training.jsonl

# 或使用自定义数据
python -m MiniPromptCoT.training.selfplay_trainer \
    --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
    --train_data_path "data/selfplay_training.jsonl" \
    --output_dir "selfplay_output"
```

## 项目结构

```
MiniPromptCoT/
├── MiniPromptCoT/           # 主包
│   ├── __init__.py
│   ├── data_generation/     # 数据生成模块
│   │   ├── __init__.py
│   │   ├── concept_encoder.py
│   │   ├── concept_sampler.py
│   │   ├── problem_generator.py
│   │   ├── test_case_generator.py
│   │   ├── trajectory_collector.py
│   │   └── evaluator.py
│   └── training/            # 模型训练模块
│       ├── __init__.py
│       ├── sft_trainer.py
│       └── selfplay_trainer.py
├── configs/                  # 配置文件
│   └── default.yaml
├── scripts/                  # 运行脚本
│   ├── run_data_gen.sh
│   ├── run_sft.sh
│   └── run_selfplay.sh
├── data/                     # 数据目录
├── output/                   # 输出目录
├── requirements.txt          # 依赖列表
└── README.md                 # 本文档
```

## 核心创新点

1. **概念驱动的数据合成**: 通过概念组合生成多样化问题
2. **EM 风格的迭代改进**: 概念 → 原理 → 问题的合成循环
3. **自训练提升**: Self-Play 范式实现模型自主进化

## 评估指标

- **数学**: 准确率 (Accuracy)
- **编程**: Pass@k, 代码测试通过率

## 依赖

- torch >= 2.0
- transformers >= 4.30
- datasets >= 2.14
- accelerate >= 0.20
- vllm >= 0.3
- sentence-transformers >= 2.2


## License

MIT License
