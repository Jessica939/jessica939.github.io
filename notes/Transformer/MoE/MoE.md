---
date: 2026-02-27
---
# A Brief Introduction Of MoE
这是对[How to Train Really Large Models on Many GPUs?](../../AIInfra/How_to_Train_Really_Large_Models_on_Many_GPUs/notes.md#mixture-of-experts-moe)中MoE章节的补充

在大模型（LLM）遵循Scaling Law（缩放定律）不断攀升参数规模的背景下，计算成本与显存墙（Memory Wall）成为制约AGI发展的物理瓶颈。混合专家模型（Mixture of Experts, MoE）通过条件计算（Conditional Computation）实现了模型容量（Capacity）与推理计算量（FLOPs）的解耦，成为当前万亿参数模型的主流架构。

本文系统回顾了MoE从1991年雏形到2024年DeepSeek-V3/Mixtral爆发的完整演进史，深入解析Top-k门控机制、负载均衡损失函数等核心算法，并探讨其在训练稳定性与推理系统优化上的工程挑战。



## 稠密模型的困境与稀疏计算的崛起

在GPT-3时代，Dense（稠密）Transformer模型占据统治地位。其特点是：对于每一个输入Token，模型网络中的所有参数都会参与计算。这意味着，模型参数量（Parameters）与单次前向传播的计算量（FLOPs）呈严格的线性关系（$FLOPs \approx 2 \times Params \times Tokens$）。

当模型迈向万亿参数（Trillion Parameters）级别时，稠密架构面临两个不可持续的挑战：

1. **训练成本指数级上升**：集群通信开销与电力成本难以承受。
2. **推理延迟**：实时交互需要极高的吞吐量，全参数激活导致延迟过高。

MoE架构的核心思想是“**稀疏性 (Sparsity)**”。它将前馈神经网络（FFN）层分解为多个独立的“专家（Experts）”，并通过一个可学习的“门控网络（Gating Network/Router）”决定每个Token由哪些专家处理。

**核心收益**：

*   **训练效率**：在相同计算资源下，MoE能训练出比稠密模型大4-8倍参数量的模型。
*   **推理速度**：尽管总参数量巨大（如DeepSeek-V3总参数671B），但每个Token仅激活极少部分参数（如37B），推理成本仅相当于一个小模型。


## 历史溯源：从统计混合到深度学习的复兴 (1991-2017)

### 起源：Adaptive Mixtures of Local Experts (1991)

MoE的概念最早由Geoffrey Hinton、Michael Jordan等人于1991年提出。当时的初衷并非为了“大模型”，而是为了解决多任务学习中的“干扰”问题。

**数学原理**：假设输入为 $x$，系统输出 $y$ 是 $N$ 个局部专家网络 $E_i(x)$ 的线性加权组合，权重由门控网络 $G(x)$ 决定：

$$ y = \sum_{i=1}^{N} G(x)_i E_i(x) $$

其中，门控网络的输出满足概率约束：

$$ \sum_{i=1}^{N} G(x)_i = 1, \quad G(x)_i \ge 0 $$

早期使用的是Softmax函数作为门控，且所有专家都会被分配一定的权重（Softmax结果不为零）。这实际上是一种**集成学习（Ensemble Learning）**，计算上并未稀疏化。

### 深度学习时代的转折：Sparsely-Gated MoE (2017)

2017年，Google Brain的Noam Shazeer等人发表了《Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer》，这是现代MoE的基石。

**关键创新：Top-k Gating** 为了实现真正的稀疏计算，必须让大多数 $G(x)_i$ 变为0。Shazeer引入了Top-k机制，只保留输出值最大的k个专家。

$$ G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k)) $$

其中 $H(x) = x \cdot W_g$ 是门控网络的原始logits。为了保持训练时的可微性与探索性，还引入了高斯噪声（Gaussian Noise）：

$$ H(x)_i = (x \cdot W_g)_i + \text{StandardNormal}() \cdot \text{Softplus}((x \cdot W_{noise})_i) $$

这一改进使得LSTM网络在语言建模任务上实现了百倍的参数扩展，而计算量增加甚微。


## 架构爆发：Transformer与MoE的完美结合 (2020-2023)

Transformer架构中的FFN（Feed-Forward Network）层占据了约2/3的参数量，且各层之间独立，非常适合替换为MoE层。

### GShard (2020) 与 Switch Transformer (2021)

Google在这一时期主导了MoE的发展，主要解决工程**扩展性**问题。

*   **GShard**：首次将MoE扩展到Transformer中，使用Top-2 Gating。它定义了跨TPU Pod的分布式MoE训练范式（Expert Parallelism）。
*   **Switch Transformer**：将简化推向极致。提出 **Top-1 Gating**，即每个Token只路由给一个专家。
    *   **优势**：极大的减少了通信开销（Communication Overhead）。
    *   **结论**：证明了与其增加专家的复杂度，不如增加专家的数量。Switch Transformer实现了1.6T参数的模型，训练速度比T5-XXL快4倍。

## 负载均衡 (Load Balancing) 的数学博弈

MoE训练中最大的噩梦是**路由坍塌（Routing Collapse）**。即门控网络发现某几个专家表现稍好，就将所有Token都发给它们，导致这些专家过载（Overload），而其他专家“饿死”。这退化回了稠密模型。

为了解决此问题，必须引入**辅助损失函数（Auxiliary Loss）**：

$$ L_{aux} = N \sum_{i=1}^{N} f_i \cdot P_i $$

*   $N$：专家数量。
*   $f_i$：一批数据中路由给专家 $i$ 的Token比例（实际负载）。
*   $P_i$：门控网络输出给专家 $i$ 的平均概率（预期负载）。

该损失函数强制 $f_i$ 和 $P_i$ 接近均匀分布，确保专家被均衡利用。



## 现代MoE革命：细粒度与高性能 (2024-Present)

进入2023年后，开源社区与DeepSeek等新兴力量将MoE推向了新的高度，重点在于**更高效的参数利用率**和**更强的推理性能**。

### Mixtral 8x7B (Mistral AI)

Mixtral是MoE“平民化”的里程碑。它采用了Top-2路由，总参数47B，激活参数13B。

*   **突破**：证明了稀疏MoE在同等激活参数下，性能显著优于稠密模型（打败了LLaMA 2 70B）。
*   **无噪声路由**：放弃了复杂的噪声机制，发现简单的Top-k Softmax足够稳定。

### DeepSeek-V2/V3：DeepSeekMoE架构

DeepSeek团队提出的架构（DeepSeek-V2/V3）代表了当前MoE设计的SOTA水平，其核心是对“专家”概念的重构。

#### 核心创新 1：细粒度专家 (Fine-Grained Experts)

传统MoE（如Mixtral）专家数量少（8个），单个专家参数大。DeepSeek提出将一个大专家切分为多个小专家（例如将1个FFN切分为 $m$ 个碎片）。

*   **优势**：提高了知识组合的灵活性。不同的Token可以组合更精准的“碎片”来构建所需的知识表达。

#### 核心创新 2：共享专家 (Shared Experts)

传统MoE中，通用的语法知识或常识可能需要在每个专家中重复存储，造成参数冗余。
DeepSeek引入了**Shared Experts**，这些专家总是被激活，不参与路由竞争。

**DeepSeekMoE 的输出公式：**

$$ y = \sum_{i \in A_{shared}} E_i(x) + \sum_{i \in \text{TopK}(G(x))} (g_i E_i(x)) $$

*   $A_{shared}$：固定激活的共享专家集合。
*   TopK部分：自适应激活的路由专家。

在DeepSeek-V3中，总参数671B，每个Token激活37B（其中包含共享专家和路由专家），实现了极致的训练/推理性价比。

#### 核心创新 3：无辅助损失负载均衡 (Auxiliary-Loss-Free)

传统的 $L_{aux}$ 会干扰模型的主任务学习。DeepSeek-V3创新性地在Router的logits上添加一个动态的**Bias**项来实现负载均衡，仅在训练中通过统计各专家的负载来调整Bias，而不将负载惩罚加入梯度反向传播。



## 工程挑战与系统优化

MoE的理论很美，但工程落地极其困难。

### 通信瓶颈 (Communication Overhead)

在分布式训练中，MoE引入了**All-to-All**通信。这会导致通信瓶颈。

*   **Dispatch**：将Token从数据并行的GPU发送到存有对应专家的GPU。
*   **Combine**：专家计算完成后，将结果发回原GPU。
*   **解决方案**：算子重叠（Computation-Communication Overlap），利用流水线掩盖通信延迟；限制专家容量（Capacity Factor），允许丢弃部分Token（Token Dropping）。

### 显存碎片与KV Cache

MoE模型权重巨大，推理时需要加载所有专家到显存中（尽管只计算一部分）。这导致**显存带宽（Memory Bandwidth）** 成为推理瓶颈，而非计算速度。

*   对于Offloading场景（如单卡跑大MoE），受限于PCIe带宽，推理速度极慢。

### 训练不稳定性

MoE训练曲线容易出现尖峰（Spike）或发散。

*   **原因**：路由网络的梯度具有稀疏性和高方差。
*   **对策**：Router Z-Loss（限制logits数值过大）、梯度裁剪、更精细的初始化策略。



## MoE的未来展望

### 学术界关注点

1. **动态架构**：能否根据输入难度动态决定激活多少专家？（不仅仅是固定的Top-k）。
2. **异构专家**：不同专家是否可以是不同的架构（如有的专家是Attention，有的是Conv，有的是FFN）？
3. **多模态MoE**：ChamAeon等工作开始探索在Vision-Language模型中使用MoE，视觉Token和文本Token是否应该路由到不同专家？

### 工业界与投资逻辑

1. **推理成本护城河**：MoE使得提供API服务的公司（如OpenAI, DeepSeek）能以极低的边际成本（Token/$）提供高性能模型。DeepSeek-V3 API价格的大幅下降正是得益于此。
2. **端侧部署（On-Device AI）**：MoE是端侧大模型的希望。通过存储大量参数但仅激活少量，可以在手机有限的算力下实现高智商，但需解决存储空间（Flash Memory）读取速度的问题。
3. **硬件协同**：未来的AI芯片（NPU/LPU）可能会针对MoE的稀疏内存访问模式（Sparse Memory Access）进行专门优化。