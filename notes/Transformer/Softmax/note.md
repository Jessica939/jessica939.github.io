---
date: 2026-02-26
---
# A Brief Introduction of Softmax and Flash Attention
在 AI Infra 里，Softmax 和 Flash Attention 的优化，本质上就是一个**如何用数学公式的等价变形，来欺骗硬件、减少物理内存读写**的故事。

把知识链条拆解为三步：**朴素 Softmax -> Online Softmax -> Flash Attention**。我们一步步看数学是怎么拯救工程的。

## 第一步：朴素 Softmax 与它的工程灾难

### 1. 数学定义与溢出问题
Softmax 的数学定义很简单，给定一个向量 $x$，它的第 $i$ 个元素输出是：
$$ \text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}} $$
*   **理论问题**：指数函数 $e^x$ 增长极快。在 FP16 精度下，最大只能表示 65504。只要 $x_i > 11$，$e^{11} \approx 59874$，再大一点点就会**数值溢出（Overflow）**，变成 `NaN`。

### 2. Safe Softmax
为了防止溢出，数学上做了一个极其简单的等价变形：**分子分母同乘一个常数** $e^{-x_{max}}$。
$$ \text{Softmax}(x_i) = \frac{e^{x_i - x_{max}}}{\sum_{j} e^{x_j - x_{max}}} $$
因为 $x_i - x_{max} \le 0$，所以 $e^{x_i - x_{max}}$ 最大也就是 $e^0 = 1$，绝对不会溢出。

### 3. Infra 视角的灾难
理论很完美，但放到 GPU 上运算时，灾难来了。GPU 的计算极快，但**显存（HBM）读写极慢**。要计算 Safe Softmax，GPU 必须对同一个数组遍历 **3 次**（即 3 次 HBM 读写）：
1.  **Pass 1**：遍历所有 $x_i$，找出最大值 $x_{max}$。
2.  **Pass 2**：再次遍历所有 $x_i$，计算 $e^{x_i - x_{max}}$ 并求和，得到分母 $Sum$。
3.  **Pass 3**：第三次遍历 $x_i$，计算最终的 $\frac{e^{x_i - x_{max}}}{Sum}$ 并写回显存。

**结论**：计算量不大，但数据在“极慢的显存”和“极快的计算单元”之间来回搬运了 3 次，导致极大的延迟。


## 第二步：Online Softmax理论破局

既然遍历 3 次太慢，能不能**只遍历 1 次**，边读边算？
过去的痛点是：**在你没有读完所有数据之前，你根本不知道全局的 $x_{max}$ 是多少**，所以你没法提前算指数！

**Online Softmax（2018 年被提出）用了一个极其精妙的数学 Trick，实现了“增量计算”。**

### 1.核心推导：
假设我们已经遍历了前一部分数据，维护了两个局部变量：
*   当前的局部最大值：$m_{old}$
*   当前的局部分母求和：$d_{old} = \sum e^{x - m_{old}}$

现在，我们读入了一个新的数据块（或者单个元素），找到了一个新的局部最大值 $m_{new}$。
显然，真正的全局最大值应该是 $m = \max(m_{old}, m_{new})$。

**精妙之处来了：如何不回头重新遍历，就能修正之前的分母 $d_{old}$？**
数学上，我们只需把旧的分母乘以一个**修正系数（Scaling factor）**： $e^{m_{old} - m_{new}}$
$$ d_{new} = d_{old} \times e^{m_{old} - m_{new}} + \sum e^{x_{new} - m_{new}} $$

### 2.意义：
有了这个递推公式，我们**只需要顺序扫描一遍数据**。
每读一个新数据，就动态更新一次 $m$ 和 $d$。当扫描到末尾时，我们手里的 $m$ 就是全局最大值，$d$ 就是全局的分母！
这把原先的 3 趟内存访问，**硬生生通过数学变形降到了 2 趟**（一趟求总 $m$ 和 $d$，第二趟直接输出结果），如果是跟其他算子融合，甚至可以变成 1 趟。


## 第三步：Flash Attention：系统与理论的完美协同

理解了 Online Softmax，Flash Attention 就极其简单了，它就是 **Online Softmax 的应用**。

### 1. 传统Attention的痛点
Attention 的公式是：$\text{Attention}(Q, K, V) = \text{Softmax}(Q K^T) V$
假设序列长度 $N = 4096$。
*   传统做法：算完 $Q K^T$ 后，生成一个 $4096 \times 4096$ 的庞大矩阵，**存入显存（HBM）**。然后对这个巨型矩阵做 3 次 Pass 的 Softmax，再**存入显存**。最后再拿出来和 $V$ 相乘。
*   **痛点**：中间产生的这个 $O(N^2)$ 的巨型矩阵（Attention Map），把显存带宽彻底吃光了。

### 2. Flash Attention 的解决思路（Tiling + Fusion）
Flash Attention 的作者 Tri Dao想：**能不能根本就不在显存里存这个 $N \times N$ 的矩阵？**

他利用了 GPU 内存的分级结构：
*   **HBM（主显存）**：容量大（80GB），但贼慢。
*   **SRAM（共享内存）**：容量极小（每块几十MB/几百KB），但**速度跟光一样快**。


假设输入序列长度 $N=4096$，向量大小（Head Dimension）$d=64$。
传统 Attention 中，完整的 $Q, K, V$ 矩阵维度都是 $4096 \times 64$。计算完整的 $Q K^T$ 会在极慢的主显存（HBM）中生成一个 **$4096 \times 4096$ 的巨大中间分数矩阵**

为了彻底消灭这个 $4096 \times 4096$ 的大矩阵，我们沿着序列维度 $N$，将 $Q, K, V$ 切分成大小为 **$64 \times 64$** 的小块，并通过**内外两层循环**在极快的 SRAM中完成所有计算：

*   **Step 1（外层循环 - 锚定 Q）**：
    每次只从极慢的 HBM 中取出一块 $Q_{block}$（维度 $64 \times 64$，即 64 个词）搬入极快的 SRAM 中。同时在 SRAM 里为这 64 个词初始化极小的累加器：输出矩阵 $O_{local}$（$64 \times 64$）、最大值 $m$ 和分母 $d$。
*   **Step 2（内层循环 - 遍历 K、V 传送带）**：
    保持 $Q_{block}$ 在 SRAM 中不动，将 $K$ 和 $V$ 的小块（$K_{block}, V_{block}$，均为 $64 \times 64$）像传送带一样，一块一块地从 HBM 搬进 SRAM。
    在 SRAM 中计算 $Q_{block} \times K_{block}^T$，产生一个**极小的 $64 \times 64$ 局部分数矩阵**（彻底规避了 $4096 \times 4096$ 矩阵的显存读写）。
*   **Step 3（关键点：Online Softmax 动态修正）**：
    针对这 $64 \times 64$ 的局部分数，立刻使用 **Online Softmax** 公式找出局部最大值，并利用修正系数 $e^{m_{old} - m_{new}}$ 对 SRAM 中历史累加的 $O_{local}$ 和分母 $d$ 进行严密的数学等价缩放。这使得我们在不回头读取历史块的情况下，也能将“局部概率”完美修正为“全局概率”。
*   **Step 4（Fusion：即算即乘，用完即弃）**：
    修正完概率后，立刻在 SRAM 里与当前的 $V_{block}$（$64 \times 64$）相乘，并累加到 $O_{local}$ 中。
    **核心交易**：计算完成后，当前的 $K_{block}$ 和 $V_{block}$ 直接从 SRAM 中丢弃！宁可为了其他 $Q$ 块去 HBM 里**重复读取** $K$ 和 $V$，也绝不在显存里存放任何中间产物。
*   **Step 5（见证奇迹的 Write-Once）**：
    当内层循环把所有的 $K$ 和 $V$（一共 64 块，总计 4096 个词）全部遍历完后，SRAM 中的 $O_{local}$（$64 \times 64$）就已经吸满了全局信息，成为了绝对正确的最终 Attention 结果。
    此时，我们才做唯一一次极其昂贵的写操作，将这 $64 \times 64$ 的结果从 SRAM 一次性写回 HBM 的对应位置。随后清空 SRAM，进入下一个 $Q_{block}$ 的外层循环。


依靠 **Online Softmax 的局部可更新性**，Flash Attention 彻底消灭了 $O(N^2)$ 的中间矩阵读写。它的 FLOPs（浮点计算量）甚至比标准 Attention 还要稍微多一点点，但因为**极大地减少了 HBM 的访存**，它的实际运行速度快了 2~4 倍，同时节省了 10~20 倍的显存！



### Flash Attention原论文
[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)