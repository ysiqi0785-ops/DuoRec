# DuoRec 主实验复现分析报告

---

## Step A. 主实验信息盘点与证据地图

### A1. 主实验清单

| Experiment ID | 数据集/场景 | 任务 | 论文证据位置 |
|---------------|-------------|------|--------------|
| E1 | Amazon Beauty | 序列推荐（下一项预测） | Table 2, Section 5.2 |
| E2 | Amazon Clothing | 序列推荐（下一项预测） | Table 2, Section 5.2 |
| E3 | Amazon Sports | 序列推荐（下一项预测） | Table 2, Section 5.2 |
| E4 | MovieLens-1M (ML-1M) | 序列推荐（下一项预测） | Table 2, Section 5.2 |
| E5 | Yelp | 序列推荐（下一项预测） | Table 2, Section 5.2 |

**被排除实验清单（不展开）：**
- Section 5.3: Ablation Study of Contrastive Learning（消融实验：UCL, SCL, UCL+SCL变体对比）
- Section 5.4: Contrastive Regularization in Training（可视化分析、alignment/uniformity分析）
- Section 5.5: Parameter Sensitivity（Dropout比率敏感性、λ参数敏感性）
- Figure 1: 表征退化问题可视化（案例分析）
- Figure 3-5: 嵌入可视化与训练损失分析

### A2. 代码仓库关键入口文件清单

| 文件路径 | 功能 |
|----------|------|
| `run_seq.py` | 主训练/评测入口脚本 |
| `duorec.sh` | 运行示例shell脚本 |
| `seq.yaml` | 主配置文件（数据、训练、评测参数） |
| `recbole/properties/model/DuoRec.yaml` | DuoRec模型默认参数配置 |
| `recbole/properties/overall.yaml` | 全局默认配置 |
| `recbole/model/sequential_recommender/duorec.py` | DuoRec模型实现 |
| `recbole/data/dataloader/sequential_dataloader.py` | 数据加载器（含DuoRec语义增强） |
| `recbole/data/dataset/sequential_dataset.py` | 数据集处理（含same_target_index生成） |
| `recbole/quick_start/quick_start.py` | 训练/评测流程封装 |
| `recbole/evaluator/metrics.py` | 评测指标实现（Hit, NDCG, MRR等） |
| `requirements.txt` | Python依赖 |

### A3. 主实验映射表

| Experiment ID | 论文证据 | 代码证据 | 映射状态 |
|---------------|----------|----------|----------|
| E1 (Beauty) | Table 2, Section 5.1.1 | `run_seq.py --dataset='Amazon_Beauty'`, `seq.yaml` | ✓ 可映射 |
| E2 (Clothing) | Table 2, Section 5.1.1 | `run_seq.py --dataset='Amazon_Clothing_Shoes_and_Jewelry'`, `seq.yaml` | ✓ 可映射 |
| E3 (Sports) | Table 2, Section 5.1.1 | `run_seq.py --dataset='Amazon_Sports_and_Outdoors'`, `seq.yaml` | ✓ 可映射 |
| E4 (ML-1M) | Table 2, Section 5.1.1 | `run_seq.py --dataset='ml-1m'`, `seq.yaml`, `duorec.sh` | ✓ 可映射 |
| E5 (Yelp) | Table 2, Section 5.1.1 | `run_seq.py --dataset='yelp'`, `seq.yaml` | ✓ 可映射 |

**缺失项说明：**
- 数据集需从外部下载（RecSysDatasets），仓库内无数据
- 论文中Dropout和λ的最优值未明确给出每个数据集的具体设置【需调参】
- `init_seed`在`quick_start.py`中被注释，可能影响可复现性 [Repo: recbole/quick_start/quick_start.py 第31行]

---

## 0. 主实验复现结论总览

| Experiment ID | 场景/数据集 | 任务 | 论文主指标与数值 | 代码入口 | 复现难度 | 可复现性判断 | 主要风险点 |
|---------------|-------------|------|------------------|----------|----------|--------------|------------|
| E1 | Amazon Beauty | 序列推荐 | HR@5=0.0546±0.0013, HR@10=0.0845±0.0010, NDCG@5=0.0352±0.0006, NDCG@10=0.0443±0.0006 [Table 2] | `python run_seq.py --dataset='Amazon_Beauty' --model='DuoRec'` | 中 | 部分可复现 | 数据需外部下载；Dropout/λ最优值未明确；随机种子初始化被注释 |
| E2 | Amazon Clothing | 序列推荐 | HR@5=0.0193±0.0012, HR@10=0.0302±0.0009, NDCG@5=0.0113±0.0011, NDCG@10=0.0148±0.0008 [Table 2] | `python run_seq.py --dataset='Amazon_Clothing_Shoes_and_Jewelry' --model='DuoRec'` | 中 | 部分可复现 | 同E1 |
| E3 | Amazon Sports | 序列推荐 | HR@5=0.0326±0.0007, HR@10=0.0498±0.0009, NDCG@5=0.0208±0.0010, NDCG@10=0.0262±0.0008 [Table 2] | `python run_seq.py --dataset='Amazon_Sports_and_Outdoors' --model='DuoRec'` | 中 | 部分可复现 | 同E1 |
| E4 | MovieLens-1M | 序列推荐 | HR@5=0.2038±0.0021, HR@10=0.2946±0.0018, NDCG@5=0.1390±0.0030, NDCG@10=0.1680±0.0032 [Table 2] | `python run_seq.py --dataset='ml-1m' --model='DuoRec'` | 低 | 可复现 | 有示例脚本；数据需外部下载 |
| E5 | Yelp | 序列推荐 | HR@5=0.0441±0.0006, HR@10=0.0631±0.0010, NDCG@5=0.0325±0.0004, NDCG@10=0.0386±0.0005 [Table 2] | `python run_seq.py --dataset='yelp' --model='DuoRec'` | 中 | 部分可复现 | 同E1；Yelp需特定时间过滤 |

**可复现性判断依据：**
- 代码完整：模型、数据加载、评测脚本均存在
- 数据可得：需从RecSysDatasets下载，公开可用
- 配置基本完整：默认参数与论文描述一致
- 风险：随机种子初始化被注释可能导致结果波动；超参数调优范围已知但最优值未明确

---

## 1. 论文概述

### 1.1 标题
Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation [Paper: 标题]

### 1.2 方法一句话总结
DuoRec是一个序列推荐模型，**输入**为用户历史交互序列，**输出**为下一个物品的预测概率分布，**核心机制**是通过Transformer编码器生成序列表示，并使用双重对比学习正则化（基于Dropout的无监督增强 + 基于相同目标物品的有监督正样本采样）来缓解物品嵌入的表征退化问题。

### 1.3 核心贡献
1. **识别并分析表征退化问题**：发现序列推荐模型中物品嵌入分布退化为各向异性的窄锥形，导致语义相似度失真 [Paper: Section 2.2, Figure 1]
2. **提出对比正则化**：设计对比学习目标作为序列表示的正则化，隐式改善物品嵌入分布的均匀性 [Paper: Section 4.3]
3. **模型级增强（Dropout）**：提出基于Dropout的无监督增强方法，对同一序列应用不同Dropout掩码生成语义一致的正样本对 [Paper: Section 4.2]
4. **有监督正样本采样**：开发基于目标物品的正样本采样策略，将具有相同目标物品的序列视为语义相似的硬正样本 [Paper: Section 4.2]
5. **【归纳】端到端训练**：将推荐损失与对比损失联合优化，无需预训练阶段

---

## 2. 主实验复现详解

---

### 【E1 主实验标题：Amazon Beauty数据集序列推荐】

#### A. 这个主实验在回答什么问题
- **实验目的**：验证DuoRec在Amazon Beauty数据集上的序列推荐性能，与7个基线方法对比
- **核心结论对应点**：DuoRec在所有指标上显著优于基线，HR@5提升35.91%，NDCG@5提升57.85%
- **论文证据位置**：Table 2 (Beauty行), Section 5.2 Overall Performance

#### B. 实验任务与工作原理

**任务定义：**
- **输入**：用户历史交互物品序列 $s = [v_1, v_2, ..., v_t]$，最大长度50
- **输出**：下一个物品 $v_{t+1}$ 的预测概率分布
- **预测目标**：在全物品集上排序，预测用户下一个交互的物品
- **约束条件**：留一法评测，最后一个交互作为测试，倒数第二个作为验证

**方法关键流程：**
1. **数据处理**：物品序列 → 物品嵌入 + 位置嵌入 → LayerNorm → Dropout
2. **序列编码**：Transformer编码器（2层，2头）→ 取最后位置输出作为序列表示
3. **推荐损失**：序列表示与物品嵌入矩阵点积 → CrossEntropy损失
4. **对比损失**：
   - 无监督：同一序列两次前向传播（不同Dropout掩码）→ InfoNCE
   - 有监督：采样相同目标物品的其他序列 → InfoNCE
5. **总损失**：$L = L_{rec} + \lambda \cdot L_{un} + \lambda_{sem} \cdot L_{su}$ [Paper: Equation 15]

**最终设置**：
- 完整DuoRec模型（`contrast='us_x'`）：无监督增强序列与有监督正样本计算对比损失
- 标准训练策略：Adam优化器，lr=0.001，batch=256
- 官方评测协议：留一法划分，全物品排序

**实例说明**：
用户交互序列 `[item_1, item_2, item_3, item_4]`：
- 训练样本：`[item_1, item_2, item_3]` → 预测 `item_4`
- 无监督增强：对 `[item_1, item_2, item_3]` 两次编码（不同Dropout）得到 $z_i, z_j$
- 有监督增强：找到其他以 `item_4` 为目标的序列，编码得到 $z_{sem}$

#### C. 数据

**数据集名称与来源：**
- 名称：Amazon Beauty [Paper: Section 5.1.1]
- 来源：RecSysDatasets (https://github.com/RUCAIBox/RecSysDatasets) [Repo: README.md]
- 下载链接：https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI [Repo: README.md]

**数据许可/访问限制：**
- 【未知】README未明确说明许可证，Amazon数据集通常用于学术研究

**数据结构示例：**
```
# 文件：dataset/Amazon_Beauty/Amazon_Beauty.inter
user_id:token    item_id:token    rating:float    timestamp:float
A1234            B5678            5.0             1234567890
```
[Repo: recbole/dataset_example/ml-100k/ml-100k.inter 格式参考]

**Dataset类__getitem__返回内容：**
```python
{
    'user_id': tensor,           # 用户ID
    'item_id_list': tensor,      # 历史物品序列 [max_len]
    'item_length': tensor,       # 序列实际长度
    'item_id': tensor,           # 目标物品ID
    'sem_aug': tensor,           # 有监督增强序列 [max_len]（仅训练）
    'sem_aug_lengths': tensor    # 增强序列长度（仅训练）
}
```
[Repo: recbole/data/dataloader/sequential_dataloader.py duorec_aug函数]

**数据量：**
| 指标 | 论文值 [Table 1] |
|------|------------------|
| # Users | 22,363 |
| # Items | 12,101 |
| # Avg. Length | 8.9 |
| # Actions | 198,502 |
| Sparsity | 99.93% |

**训练集构建：**
- 过滤：用户/物品交互少于5次的被移除 [Paper: Section 5.1.1]
- 序列构建：按时间排序，滑动窗口生成训练样本 [Repo: recbole/data/dataset/sequential_dataset.py prepare_data_augmentation]
- 最大长度：50 [Paper: Section 5.1.1, Repo: seq.yaml MAX_ITEM_LIST_LENGTH]
- 语义增强索引：预计算相同目标物品的序列索引 [Repo: recbole/data/dataset/sequential_dataset.py semantic_augmentation]

**测试集构建：**
- 留一法：每用户最后一个交互作为测试 [Repo: seq.yaml eval_setting: TO_LS]
- 验证集：倒数第二个交互

**预处理与缓存：**
- 语义增强索引缓存：`dataset/Amazon_Beauty/semantic_augmentation.npy` [Repo: recbole/data/dataset/sequential_dataset.py 第91-104行]
- 首次运行自动生成

#### D. 模型与依赖

**基础模型/Backbone：**
- Transformer编码器（自实现）[Repo: recbole/model/layers.py TransformerEncoder]
- 无需预训练权重下载

**关键模块与参数：**
| 模块 | 参数 | 值 | 来源 |
|------|------|-----|------|
| Item Embedding | hidden_size | 64 | [Paper: Section 5.1.4, Repo: DuoRec.yaml] |
| Position Embedding | max_seq_length | 50 | [Repo: seq.yaml] |
| Transformer | n_layers | 2 | [Paper: Section 5.1.4, Repo: DuoRec.yaml] |
| Transformer | n_heads | 2 | [Paper: Section 5.1.4, Repo: DuoRec.yaml] |
| Transformer | inner_size | 256 | [Repo: DuoRec.yaml] |
| Dropout | hidden_dropout_prob | 0.5 | [Repo: seq.yaml, DuoRec.yaml] |
| Dropout | attn_dropout_prob | 0.5 | [Repo: seq.yaml, DuoRec.yaml] |
| Contrastive | lmd (λ) | 0.1 | [Repo: seq.yaml, DuoRec.yaml] |
| Contrastive | lmd_sem (λ_sem) | 0.1 | [Repo: seq.yaml, DuoRec.yaml] |
| Contrastive | tau (温度) | 1 | [Repo: seq.yaml] |
| Contrastive | contrast | 'us_x' | [Repo: seq.yaml] |
| Contrastive | sim | 'dot' | [Repo: seq.yaml] |

**训练策略：**
- 优化器：Adam [Paper: Section 5.1.4, Repo: seq.yaml learner: adam]
- 学习率：0.001 [Paper: Section 5.1.4, Repo: seq.yaml]
- Batch size：256 [Paper: Section 5.1.4, Repo: seq.yaml]
- Epochs：50 [Repo: seq.yaml]
- 早停：10步无提升 [Repo: seq.yaml stopping_step: 10]
- 验证指标：MRR@10 [Repo: seq.yaml valid_metric]
- 损失类型：CrossEntropy [Repo: DuoRec.yaml loss_type: 'CE']
- Weight decay：0 [Repo: seq.yaml]

**随机性控制：**
- seed：2020 [Repo: overall.yaml]
- reproducibility：True [Repo: overall.yaml]
- **注意**：`init_seed`在`quick_start.py`第31行被注释，需手动取消注释以确保可复现性 [Repo: recbole/quick_start/quick_start.py]

#### E. 评价指标与论文主表预期结果

**指标定义：**
| 指标 | 含义 | 计算方式 |
|------|------|----------|
| HR@K (Hit Ratio) | Top-K推荐列表中包含正确物品的用户比例 | $\frac{\text{命中用户数}}{\text{总用户数}}$ [Repo: recbole/evaluator/metrics.py hit函数] |
| NDCG@K | 归一化折损累积增益，考虑排序位置 | $\frac{DCG@K}{IDCG@K}$ [Repo: recbole/evaluator/metrics.py ndcg函数] |

**论文与代码指标对应：**
- 论文使用HR@K，代码中对应Hit@K [Repo: recbole/evaluator/evaluators.py 第25行]
- 计算方式一致

**论文主结果数值 [Table 2]：**
| 指标 | DuoRec | 标准差 |
|------|--------|--------|
| HR@5 | 0.0546 | ±0.0013 |
| HR@10 | 0.0845 | ±0.0010 |
| NDCG@5 | 0.0352 | ±0.0006 |
| NDCG@10 | 0.0443 | ±0.0006 |

**复现预期**：以论文主表数值为准，允许±标准差范围内的波动

#### F. 环境与硬件需求

**软件环境 [Repo: requirements.txt]：**
```
matplotlib>=3.1.3
torch>=1.7.0
numpy>=1.17.2
scipy==1.6.0
hyperopt>=0.2.4
pandas>=1.0.5
tqdm>=4.48.2
scikit_learn>=0.23.2
pyyaml>=5.1.0
colorlog==4.7.2
colorama==0.4.4
```

**硬件要求：**
- GPU：【推断】单GPU即可，模型参数量小（embedding 64维，2层Transformer）
- 显存：【推断】约2-4GB（batch=256，序列长度50，embedding 64）
- 依据：模型规模小，无分布式训练代码

**训练时长：**
- 【未知】论文和README未提供具体训练时间

#### G. 可直接照做的主实验复现步骤

**步骤1：获取代码与安装依赖**
```bash
# 克隆仓库
git clone https://github.com/RuihongQiu/DuoRec.git
cd DuoRec

# 创建conda环境（推荐）
conda create -n duorec python=3.8
conda activate duorec

# 安装依赖
pip install -r requirements.txt
pip install seaborn  # quick_start.py中使用但未在requirements中
```
- 目的：准备运行环境
- 关键配置：Python 3.8+, PyTorch 1.7+
- 预期产物：可用的Python环境

**步骤2：获取数据与放置路径**
```bash
# 创建数据目录
mkdir -p dataset/Amazon_Beauty

# 从RecSysDatasets下载数据
# 方式1：Google Drive手动下载
# https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI
# 下载Amazon_Beauty文件夹中的.inter和.item文件

# 方式2：使用gdown（需安装：pip install gdown）
# 【推断】具体文件ID需从Google Drive获取

# 放置路径
# dataset/Amazon_Beauty/Amazon_Beauty.inter
# dataset/Amazon_Beauty/Amazon_Beauty.item
```
- 目的：准备训练数据
- 关键配置：`seq.yaml`中`data_path: "./dataset/"` [Repo: seq.yaml 第51行]
- 预期产物：`dataset/Amazon_Beauty/`目录下有`.inter`和`.item`文件

**步骤3：修复随机种子初始化（确保可复现性）**
```bash
# 编辑 recbole/quick_start/quick_start.py
# 取消第31行的注释：
# 将 # init_seed(config['seed'], config['reproducibility'])
# 改为 init_seed(config['seed'], config['reproducibility'])
```
- 目的：确保实验可复现
- 依据：原代码注释了种子初始化 [Repo: recbole/quick_start/quick_start.py 第31行]

**步骤4：训练DuoRec模型**
```bash
python run_seq.py \
    --dataset='Amazon_Beauty' \
    --model='DuoRec' \
    --config_files='seq.yaml'
```
- 目的：训练模型
- 关键参数：
  - `--dataset`：数据集名称，需与`dataset/`下目录名一致
  - `--model`：模型名称
  - `--config_files`：配置文件
- 预期产物：
  - 日志：`./log/DuoRec/Amazon_Beauty/`目录下
  - 模型：`./log/DuoRec/Amazon_Beauty/xxx/model.pth`
  - 可视化：`./log/DuoRec/Amazon_Beauty/xxx/DuoRec-Amazon_Beauty.pdf`

**步骤5：查看评测结果**
```bash
# 训练结束后，结果会打印在终端和日志文件中
# 查看日志
cat ./log/DuoRec/Amazon_Beauty/*/DuoRec*.log | grep "test result"
```
- 目的：获取主表指标
- 预期输出格式：
```
test result: {'hit@5': 0.0546, 'hit@10': 0.0845, 'ndcg@5': 0.0352, 'ndcg@10': 0.0443, ...}
```

**步骤6：主表指标对齐**
- 代码输出`hit@K`对应论文`HR@K`
- 代码输出`ndcg@K`对应论文`NDCG@K`
- 对比论文Table 2中Beauty行的DuoRec列

#### H. 可复现性判断

**结论：部分可复现**

**依据清单：**
| 项目 | 状态 | 说明 |
|------|------|------|
| 数据可得性 | ✓ | RecSysDatasets公开可下载 |
| 代码完整性 | ✓ | 模型、训练、评测代码完整 |
| 配置完整性 | ⚠ | 默认参数与论文一致，但Dropout/λ最优值需调参 |
| 随机性控制 | ⚠ | init_seed被注释，需手动修复 |
| 预训练权重 | ✓ | 无需预训练权重 |
| 评测协议 | ✓ | 留一法+全排序与论文一致 |

**补救路径：**
1. 取消`init_seed`注释确保可复现性
2. 若结果偏差较大，尝试调整Dropout（0.1-0.5）和λ（0.1-0.5）
3. 多次运行取平均以对齐论文中的均值±标准差

#### I. 主实验专属排错要点

1. **数据集名称**：必须与`dataset/`下目录名完全一致（如`Amazon_Beauty`而非`amazon_beauty`）
2. **语义增强缓存**：首次运行会生成`semantic_augmentation.npy`，若数据变更需删除重新生成
3. **GPU内存**：若OOM，减小`train_batch_size`（如128）
4. **指标名称映射**：代码输出`hit`对应论文`HR`

---

### 【E2 主实验标题：Amazon Clothing数据集序列推荐】

#### A. 这个主实验在回答什么问题
- **实验目的**：验证DuoRec在Amazon Clothing数据集上的序列推荐性能
- **核心结论对应点**：DuoRec在所有指标上优于基线，HR@5提升14.88%
- **论文证据位置**：Table 2 (Clothing行), Section 5.2

#### B. 实验任务与工作原理
（与E1相同，此处不重复）

**任务定义**：用户历史交互序列 → 预测下一个物品

**方法关键流程**：Transformer编码 + 双重对比学习正则化

**最终设置**：完整DuoRec模型（`contrast='us_x'`），标准训练策略

#### C. 数据

**数据集名称与来源：**
- 名称：Amazon Clothing, Shoes and Jewelry [Paper: Section 5.1.1]
- 来源：RecSysDatasets
- 目录名：`Amazon_Clothing_Shoes_and_Jewelry` [Repo: README.md]

**数据量 [Table 1]：**
| 指标 | 值 |
|------|-----|
| # Users | 39,387 |
| # Items | 23,033 |
| # Avg. Length | 7.1 |
| # Actions | 278,677 |
| Sparsity | 99.97% |

**数据结构**：与E1相同

**预处理**：与E1相同（min 5交互，max 50长度）

#### D. 模型与依赖
（与E1完全相同）

#### E. 评价指标与论文主表预期结果

**论文主结果数值 [Table 2]：**
| 指标 | DuoRec | 标准差 |
|------|--------|--------|
| HR@5 | 0.0193 | ±0.0012 |
| HR@10 | 0.0302 | ±0.0009 |
| NDCG@5 | 0.0113 | ±0.0011 |
| NDCG@10 | 0.0148 | ±0.0008 |

#### F. 环境与硬件需求
（与E1相同）

#### G. 可直接照做的主实验复现步骤

**步骤1-3**：与E1相同

**步骤2补充：数据放置**
```bash
mkdir -p dataset/Amazon_Clothing_Shoes_and_Jewelry
# 下载并放置：
# dataset/Amazon_Clothing_Shoes_and_Jewelry/Amazon_Clothing_Shoes_and_Jewelry.inter
# dataset/Amazon_Clothing_Shoes_and_Jewelry/Amazon_Clothing_Shoes_and_Jewelry.item
```

**步骤4：训练**
```bash
python run_seq.py \
    --dataset='Amazon_Clothing_Shoes_and_Jewelry' \
    --model='DuoRec' \
    --config_files='seq.yaml'
```

**步骤5-6**：与E1相同，查看日志获取结果

#### H. 可复现性判断
**结论：部分可复现**（依据与E1相同）

#### I. 主实验专属排错要点
- 数据集目录名较长，注意拼写：`Amazon_Clothing_Shoes_and_Jewelry`

---

### 【E3 主实验标题：Amazon Sports数据集序列推荐】

#### A. 这个主实验在回答什么问题
- **实验目的**：验证DuoRec在Amazon Sports数据集上的序列推荐性能
- **核心结论对应点**：DuoRec在所有指标上显著优于基线，NDCG@5提升61.24%
- **论文证据位置**：Table 2 (Sports行), Section 5.2

#### B. 实验任务与工作原理
（与E1相同）

#### C. 数据

**数据集名称与来源：**
- 名称：Amazon Sports and Outdoors [Paper: Section 5.1.1]
- 目录名：`Amazon_Sports_and_Outdoors` [Repo: README.md]

**数据量 [Table 1]：**
| 指标 | 值 |
|------|-----|
| # Users | 35,598 |
| # Items | 18,357 |
| # Avg. Length | 8.3 |
| # Actions | 296,337 |
| Sparsity | 99.95% |

#### D. 模型与依赖
（与E1相同）

#### E. 评价指标与论文主表预期结果

**论文主结果数值 [Table 2]：**
| 指标 | DuoRec | 标准差 |
|------|--------|--------|
| HR@5 | 0.0326 | ±0.0007 |
| HR@10 | 0.0498 | ±0.0009 |
| NDCG@5 | 0.0208 | ±0.0010 |
| NDCG@10 | 0.0262 | ±0.0008 |

#### F. 环境与硬件需求
（与E1相同）

#### G. 可直接照做的主实验复现步骤

**步骤2补充：数据放置**
```bash
mkdir -p dataset/Amazon_Sports_and_Outdoors
# 下载并放置相应文件
```

**步骤4：训练**
```bash
python run_seq.py \
    --dataset='Amazon_Sports_and_Outdoors' \
    --model='DuoRec' \
    --config_files='seq.yaml'
```

#### H. 可复现性判断
**结论：部分可复现**

#### I. 主实验专属排错要点
- 数据集目录名：`Amazon_Sports_and_Outdoors`

---

### 【E4 主实验标题：MovieLens-1M数据集序列推荐】

#### A. 这个主实验在回答什么问题
- **实验目的**：验证DuoRec在MovieLens-1M数据集上的序列推荐性能
- **核心结论对应点**：DuoRec取得最大提升，NDCG@5提升109.97%
- **论文证据位置**：Table 2 (ML-1M行), Section 5.2

#### B. 实验任务与工作原理
（与E1相同）

#### C. 数据

**数据集名称与来源：**
- 名称：MovieLens-1M [Paper: Section 5.1.1, Reference 12]
- 目录名：`ml-1m` [Repo: README.md, duorec.sh]

**数据量 [Table 1]：**
| 指标 | 值 |
|------|-----|
| # Users | 6,041 |
| # Items | 3,417 |
| # Avg. Length | 165.5 |
| # Actions | 999,611 |
| Sparsity | 95.16% |

**特点**：序列平均长度最长（165.5），数据最稠密

#### D. 模型与依赖
（与E1相同）

#### E. 评价指标与论文主表预期结果

**论文主结果数值 [Table 2]：**
| 指标 | DuoRec | 标准差 |
|------|--------|--------|
| HR@5 | 0.2038 | ±0.0021 |
| HR@10 | 0.2946 | ±0.0018 |
| NDCG@5 | 0.1390 | ±0.0030 |
| NDCG@10 | 0.1680 | ±0.0032 |

#### F. 环境与硬件需求
（与E1相同）

#### G. 可直接照做的主实验复现步骤

**步骤2补充：数据放置**
```bash
mkdir -p dataset/ml-1m
# 下载并放置：
# dataset/ml-1m/ml-1m.inter
# dataset/ml-1m/ml-1m.item
# dataset/ml-1m/ml-1m.user (可选)
```

**步骤4：训练（使用仓库提供的示例命令）**
```bash
# 方式1：使用duorec.sh（需修正语法错误）
# 原命令有语法问题：lmd=0.1 应为 --lmd=0.1
python run_seq.py \
    --dataset='ml-1m' \
    --train_batch_size=256 \
    --lmd=0.1 \
    --lmd_sem=0.1 \
    --model='DuoRec' \
    --contrast='us_x' \
    --sim='dot' \
    --tau=1

# 方式2：使用默认配置
python run_seq.py --dataset='ml-1m' --model='DuoRec' --config_files='seq.yaml'
```
[Repo: duorec.sh 有语法错误，`lmd=0.1`应为`--lmd=0.1`]

#### H. 可复现性判断
**结论：可复现**
- 有示例脚本`duorec.sh`
- 数据公开可得
- 配置明确

#### I. 主实验专属排错要点
1. **duorec.sh语法错误**：`lmd=0.1`应改为`--lmd=0.1`
2. **序列长度**：ML-1M平均序列长度165.5，但max_length=50会截断，这是预期行为

---

### 【E5 主实验标题：Yelp数据集序列推荐】

#### A. 这个主实验在回答什么问题
- **实验目的**：验证DuoRec在Yelp数据集上的序列推荐性能
- **核心结论对应点**：DuoRec取得显著提升，NDCG@5提升150%
- **论文证据位置**：Table 2 (Yelp行), Section 5.2

#### B. 实验任务与工作原理
（与E1相同）

#### C. 数据

**数据集名称与来源：**
- 名称：Yelp [Paper: Section 5.1.1]
- 目录名：`yelp` [Repo: README.md]
- **特殊处理**：使用2019年1月1日之后的交易记录 [Paper: Section 5.1.1]

**数据量 [Table 1]：**
| 指标 | 值 |
|------|-----|
| # Users | 30,499 |
| # Items | 20,068 |
| # Avg. Length | 10.4 |
| # Actions | 317,182 |
| Sparsity | 99.95% |

**时间过滤说明：**
- 论文提到使用2019年1月1日之后的数据
- 代码中`seq.yaml`有注释掉的时间过滤配置：
```yaml
#lowest_val:
#    timestamp: 1546264800  # 2019-01-01 00:00:00 UTC
```
[Repo: seq.yaml 第29行]
- 【推断】RecSysDatasets提供的yelp数据可能已预处理，或需手动启用时间过滤

#### D. 模型与依赖
（与E1相同）

#### E. 评价指标与论文主表预期结果

**论文主结果数值 [Table 2]：**
| 指标 | DuoRec | 标准差 |
|------|--------|--------|
| HR@5 | 0.0441 | ±0.0006 |
| HR@10 | 0.0631 | ±0.0010 |
| NDCG@5 | 0.0325 | ±0.0004 |
| NDCG@10 | 0.0386 | ±0.0005 |

**注意**：论文Table 2中Yelp的NDCG@5=0.076可能是排版错误，应为0.0076或其他值【推断】

#### F. 环境与硬件需求
（与E1相同）

#### G. 可直接照做的主实验复现步骤

**步骤2补充：数据放置**
```bash
mkdir -p dataset/yelp
# 下载并放置：
# dataset/yelp/yelp.inter
# dataset/yelp/yelp.item
# dataset/yelp/yelp.user (可选)
```

**步骤4：训练**
```bash
python run_seq.py \
    --dataset='yelp' \
    --model='DuoRec' \
    --config_files='seq.yaml'
```

**若需启用时间过滤（与论文一致）：**
编辑`seq.yaml`，取消注释：
```yaml
lowest_val:
    timestamp: 1546264800
```

#### H. 可复现性判断
**结论：部分可复现**
- 时间过滤配置被注释，可能导致数据量不一致
- 需确认RecSysDatasets的yelp数据是否已预处理

#### I. 主实验专属排错要点
1. **时间过滤**：若结果偏差大，检查是否需要启用时间过滤
2. **数据版本**：确认使用的yelp数据与论文描述一致

---

## 3. 主实验一致性检查

### 论文主表指标与仓库脚本对应
- **可直接产出**：是，`run_seq.py`训练结束后自动输出test result，包含hit@K和ndcg@K
- **指标名称映射**：代码`hit`=论文`HR`，代码`ndcg`=论文`NDCG`

### 多个主实验共享入口
- **共享预处理**：所有数据集使用相同的`SequentialDataset`类处理
- **共享评测入口**：所有数据集使用相同的`Trainer.evaluate()`方法
- **共享配置**：`seq.yaml`为通用配置，`DuoRec.yaml`为模型默认参数
- **独立性**：每个实验只需更改`--dataset`参数

### 最小复现路径
**推荐顺序**：E4 (ML-1M) → E1 (Beauty) → E3 (Sports) → E2 (Clothing) → E5 (Yelp)

**理由**：
1. E4有示例脚本`duorec.sh`，最容易验证
2. E1-E3为Amazon系列，数据格式一致
3. E5需要确认时间过滤，放最后

**最快验证命令**：
```bash
# 1. 准备环境和数据（假设已完成）
# 2. 修复随机种子
sed -i 's/# init_seed/init_seed/' recbole/quick_start/quick_start.py

# 3. 运行ML-1M
python run_seq.py --dataset='ml-1m' --model='DuoRec' --config_files='seq.yaml'

# 4. 检查结果
grep "test result" ./log/DuoRec/ml-1m/*/DuoRec*.log
```

---

## 4. 未知项与需要补充的最小信息

| 问题 | 必要性 | 缺失后果 |
|------|--------|----------|
| 每个数据集的最优Dropout和λ值是多少？ | 中等 | 可能需要调参才能完全复现论文结果 |
| Yelp数据是否需要手动时间过滤？ | 高（仅E5） | 数据量可能与论文不一致，影响E5结果 |
| 论文Table 2中Yelp的NDCG@5=0.076是否为排版错误？ | 低 | 仅影响结果对比判断 |

**说明**：以上问题不影响代码运行，但可能影响结果与论文的精确对齐。建议先使用默认参数运行，若结果偏差超过标准差范围再进行调参。