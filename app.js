/* nvb20k 深度解剖 — 交互：档位切换 / 模块讲解 / 导航高亮 */
"use strict";

const TIERS = {
  "3e17": {
    id: "3e17", h: 640, L: 18, heads: 5, kv: 5, inter: 1728, veN: 9,
    front: 4, mid: 10, back: 4, params: 86657522,
    flopsPerTok: "1.5277e8", steps: 3260, tokens: "854,589,440",
    trainSeq: 2048, evalSeq: 3072, wd: "0.03"
  },
  "1e18": {
    id: "1e18", h: 768, L: 26, heads: 6, kv: 6, inter: 2048, veN: 13,
    front: 5, mid: 16, back: 5, params: 171652698,
    flopsPerTok: "2.3209e8", steps: 6694, tokens: "1,754,791,936",
    trainSeq: 2048, evalSeq: 3072, wd: "0.015"
  },
  "3e18": {
    id: "3e18", h: 1024, L: 29, heads: 8, kv: 8, inter: 2752, veN: 15,
    front: 6, mid: 17, back: 6, params: 336784918,
    flopsPerTok: "4.4029e8", steps: 6602, tokens: "2,596,012,032",
    trainSeq: 3072, evalSeq: 3072, wd: "0.015"
  },
  "1e19": {
    id: "1e19", h: 1152, L: 39, heads: 9, kv: 9, inter: 3072, veN: 20,
    front: 8, mid: 23, back: 8, params: 556503290,
    flopsPerTok: "6.6886e8", steps: 13951, tokens: "5,485,756,416",
    trainSeq: 3072, evalSeq: 3072, wd: "0.015"
  }
};

const NUMFMT = n => n.toLocaleString("en-US");
const CUR = { tier: "1e19" };
const tier = () => TIERS[CUR.tier];

const MODULES = [
  {
    key: "input", name: "输入与分词器", badge: "输入",
    tags: [["读", "attn"]],
    formula: "input_ids : [B, L]  int64\nL_train = 2048 (3e17/1e18) | 3072 (3e18/1e19)\nL_eval  = 3072 (四档一致)\nvocab_size = 49152",
    what: "文本先由 SmolLM2-135M 的 GPT-2 系 BPE 分词器编码（只复用分词规则，不复用任何模型权重；嵌入矩阵从随机初始化开始训练）。一篇文档写成 [EOS, ids...] 的形式打包进定长序列；EOS 参与前向但不计入 loss 与 EvalTime 的 token 数。",
    how: [
      "训练：官方 Dolma3 mix 的文本流 → 分词器 → 定长 token 分片，采样时按块随机取、拼成 global batch = 128 条序列。",
      "评测：长文档按 3072 切块、末块右侧 pad；只有原文 token 计分。",
      "<b>与词表嵌入的分工：</b>分词器负责“文本 ↔ 整数 id”（模型外、不训练、不产生梯度）；词表嵌入表负责“整数 id → h 维向量”（模型第一层、可训练）。两者共用同一套 49152 个 id，换分词器就必须换嵌入表。",
      "整个模型是“整段前向”：一次看完整条 3072 token 的序列，不做 KV cache 增量解码。"
    ],
    flops: "分词本身不计 FLOPs；词表 49152 让输入嵌入与输出层各占 49152×h 的权重（不计入 N/N_a）。"
  },
  {
    key: "embed", name: "词表嵌入 E[x]", badge: "查表",
    tags: [["训练", "attn"], ["查表", "ve"]],
    formula: "h⁰_t = E[x_t]  +  s_in · hint_rows_t\nE ∈ R^{49152×h}，初始 std=0.02\ns_in 零初始化 → 起步时等于纯 E[x]",
    what: "最普通的 token 嵌入表。它同时是“提示向量”的载体：n-gram 提示表查到一个候选 token 后，复用同一张 E 取它的嵌入，而不是另建一张提示词表。",
    how: [
      "<b>形状：</b>3e17 是 49152×640，1e19 是 49152×1152；与输出层不共享。",
      "<b>不是 GPT-2 的表：</b>复用的只是 SmolLM2 的 GPT-2 系 byte-level BPE 分词规则（词表 49152）；嵌入矩阵本身随机初始化后训练。GPT-2 的词表是 50257，形状也对不上。",
      "<b>E 不是注意力里的 V：</b>E 只出现在输入残差流（x⁰）；每一层的 V 是“当前隐状态 × W_v”重新算出来的；VE 则是另外查表后加到 V 上。三者不要混。",
      "<b>不计参数：</b>词表参数（输入嵌入 + 输出层）按官方 FAQ 第 5 条不计入 N 与 N_a，但它参与的矩阵乘照常计入 FLOPs。",
      "输入嵌入处是提示表唯一进入主干的位置：<code>h⁰ += s_in · hint_rows</code>，s_in 是可学习的全局强度，初始为 0。"
    ],
    flops: "查表 0 矩阵乘；输出层的矩阵乘另算（每 token 2·h·V）。"
  },
  {
    key: "hint_in", name: "提示向量（输入侧）", badge: "n-gram 提示",
    tags: [["只读表", "hint"], ["训练", "hint"]],
    formula: "hint_rows_t = Σ_{c=1..15} [ (1 + amount[cls_c]) · E[tok_c] + feat[cls_c] ]\nh⁰ ← h⁰ + s_in · hint_rows\n15 列 = 每阶每个名次一列（4+4+3+2+1+1）",
    what: "把 6 个阶（2/3/4/6/10/20-gram）查到的候选后继 token，变成向量加进输入嵌入。每个候选 token 被“信任度”加权：信任度按类别（出现次数 × 纯度）学习，初始为 0；类别嵌入是每个类别的可学习向量。",
    how: [
      "类别 = 阶序号 × 42 + 次数桶×6 + 纯度桶 + 1；42 = 7 个次数桶 × 6 个纯度桶。",
      "没有提示的位置类别为 0，该列贡献 0。",
      "同一位置 15 个候选的向量直接相加，再由一个全局标量 s_in 控制整体强度（零初始化）。",
      "整个分支没有矩阵乘，只有查表 + 逐元素乘加。"
    ],
    flops: "0 矩阵乘；信任度、类别向量都是嵌入表，不计入 N/N_a。"
  },
  {
    key: "layer", name: "全长层（逐 token）", badge: "Transformer 层",
    tags: [["训练", "attn"], ["训练", "mlp"]],
    formula: "x = x + Attn(RMSNorm(x))        # 预归一化\nx = x + MLP(RMSNorm(x))\nAttn: q,k,v = W_q,W_k,W_v · x\n      q,k 过 QK-Norm（head_dim=128）与 RoPE\n      v ← λ₀·v + λ₁·VE[x_t] + λ₂·NVE₂ + λ₃·NVE₃\n      o = gate ⊙ 因果注意力(q,k,v);  x += W_o·o\nMLP : W_d · ReLU²(W_u · x)，中间维 ≈ 2.7h",
    what: "前段与后段的层在完整的 3072 个 token 上逐 token 计算。它和标准 Qwen3 层的差别只有三处：Value 里混入查表向量、注意力输出门控、MLP 用两矩阵 ReLU²（不是三矩阵 SwiGLU）。",
    how: [
      "<b>注意力头：</b>h=640/768/1024/1152，head_dim 固定 128，头数 5/6/8/9；KV 头数 = 头数（普通 MHA，不是 GQA）。",
      "<b>QK-Norm：</b>每个头做 RMSNorm(128) 再算注意力，训练更稳。",
      "<b>输出门控：</b>全长层是“每头一个门” 2σ(W_g x)，W_g 形状 [heads, h]，零初始化 → 初始门值 1。",
      "<b>层数分配：</b>18 层 = 前 4 + 中 10 + 后 4；39 层 = 8 + 23 + 8（中间 60% 的层在缩短序列上算）。"
    ],
    flops: "全长层每 token：QKVO 4h² + 注意力核心 2·L·h（因果减半）+ MLP 4h·inter；这些是 FLOPs 的主体。"
  },
  {
    key: "attn", name: "注意力内部", badge: "Attention",
    tags: [["训练", "attn"]],
    formula: "q = q_norm(W_q x) → [B, heads, L, 128]\nk = k_norm(W_k x) → 先混入 NVE（可选）\nv = W_v x → 混入 VE / NVE\nq,k = RoPE(q,k, θ=10000)\no = 因果注意力(q,k,v)\no = 2σ(W_g x) ⊙ o      # 输出门控\ny = W_o · o",
    what: "标准因果自注意力，加了三类改造：QK-Norm、RoPE、输出门控，以及在 V 上做查表混合。注意力核心（QKᵀ 与 ·V）是唯一随序列长度平方增长的项，也是“缩短中段”省 FLOPs 的关键。",
    how: [
      "<b>因果：</b>用 cuDNN 的因果注意力内核，不需要显式 mask；官方口径按因果把注意力核心 FLOPs 减半。",
      "<b>门控位置：</b>在注意力输出之后、输出投影之前；全长层按头（heads 个门），中段按通道（h 个门）。",
      "<b>为什么门控有效：</b>给 V→O 的线性映射加上依赖当前 token 的非线性，几乎不增加逐 token 计算。"
    ],
    flops: "每 token 4h²（QKVO）+ 2·L·h（注意力核心，L 为当前段的序列长度）。"
  },
  {
    key: "vmix", name: "Value 混合（VE + NVE）", badge: "查表→V",
    tags: [["训练", "ve"], ["训练", "nve"]],
    formula: "v ← λ₀·v_proj(h) + λ₁·g₁⊙VE[x_t] + Σ_{n∈{2,3}} λ_n·g_n⊙NVE_n[hash(x_{t-n+1..t})]\nλ 每层 4 个可学习标量（初值 0.5）\ng_r = 2σ(W·h[:12]) ∈ (0,2)，按 KV 头；W 零初始化 → 起步 g=1",
    what: "这是本队最核心的“不加矩阵乘”的改动：不把 n-gram 信息投影回隐藏维再加残差（那样要两个投影矩阵），而是直接把查表向量混进注意力的 <b>Value</b>。Value 本来就要乘进注意力输出，等于免费获得了局部上下文信息。",
    how: [
      "<b>VE：</b>按当前 token 查表，向量维 = KV 总维（= h，因为 MHA 且 head_dim×heads=h）。",
      "<b>NVE：</b>按最近 2/3 个 token 的哈希查表，每个阶 4 组独立哈希，每组查 kv/4 维后拼成 kv 维。",
      "<b>上下文门控：</b>每个 KV 头、每种来源一个门，只看隐状态前 12 维，零初始化（2σ(0)=1），所以训练开始时等价于不加门控。",
      "<b>不计 FLOPs：</b>查表与逐元素乘加在官方口径里不算矩阵乘。"
    ],
    flops: "0 矩阵乘；门的权重是 [3·kv, 12] 的小矩阵，产生极少量 GEMM。"
  },
  {
    key: "ve", name: "Value Embeddings 表", badge: "训练表",
    tags: [["训练", "ve"], ["查表", "ve"]],
    formula: "G 张表，每张 [49152, kv_dim]  (kv_dim = heads×128 = h)\nG = 9 / 13 / 15 / 20（对称配对：第 i 层与第 L-1-i 层共用）\n例：18 层 → (0,17),(1,16),…,(8,9)",
    what: "每个“层对”一张 token 嵌入表，按当前 token 查一个 kv 维向量直接混进该层 Value。它给每个 token 的 value 附加一条“我是谁”的可学习旁路信号：不经过任何矩阵乘，也不改变注意力分数，只改变被聚合的 value 内容。首尾对称的层共用一张表，参数省一半、信息位置互补。",
    how: [
      "<b>用在哪：</b>注意力内部、V 的线性投影之后、分头与注意力之前。混合公式 v ← λ₀·V投影 + λ₁·g⊙VE[token]（再叠加 NVE 项），λ 是每层 4 个可学习标量（初值 0.5）。",
      "<b>怎么查：</b>行号就是 token id（普通嵌入查表，不是哈希）；每行 kv 维，且 kv 维 = 头数×128 = h。中段缩短时只在块尾 token 处查（每 16 个 token 一次），与中段注意力同频。",
      "<b>怎么训练：</b>表是随机初始化（N(0,0.02)）的嵌入矩阵，梯度从注意力输出反传到 value、再反传到被查到的那些行；一个 batch 里没出现的 token 行不更新（稀疏更新）。它和输入嵌入一起归入嵌入参数组，用单独学习率更新；λ 与门控也一起训练。",
      "<b>门控：</b>每个 KV 头、每种来源一个门 g = 2σ(W·h[:12])，零初始化 → 初始 g=1，等价于不门控。",
      "<b>与输入嵌入 E 的区别：</b>E 进入残差流、是所有层共享的输入；VE 只进入 Value、按层对各自独立、不参与残差。",
      "与 NVE 的区别：VE 只看当前 token，NVE 看后缀 n-gram。"
    ],
    flops: "查表 + 逐元素混合：0 矩阵乘；只有门控的小线性层产生极少量计算。"
  },
  {
    key: "nve", name: "N-gram Value 表（NVE）", badge: "训练表",
    tags: [["训练", "nve"], ["查表", "nve"]],
    formula: "2 个阶 × 4 组哈希 × 1,048,576 桶 = 8,388,608 行\n每行 kv/4 维：3e17 160 / 1e18 192 / 3e18 256 / 1e19 288\nhash_n(x) = mix( ⊕_{j=0..n-1} x[t-j]·m_j )  mod 2^20",
    what: "给 Value 再补一条“刚过去的一两个 token 是什么”的旁路：后缀 2-gram / 3-gram 经 4 组独立哈希映射到固定桶，查出 kv/4 维向量拼成 kv 维，直接混入该层 Value。它和 VE 互补——VE 只看当前 token，NVE 带局部上下文；两者都不经过矩阵乘。",
    how: [
      "<b>怎么查：</b>位置 t 取后缀 (x_{t-1}, x_t) 和 (x_{t-2}, x_{t-1}, x_t)；每个阶用 4 组独立乘数做“乘法-异或”哈希，再对 2^20 取模得到桶号，行号 = 桶号 + 该组的固定偏移。",
      "<b>为什么要 4 组：</b>哈希必然撞桶，且没有指纹校验；4 组独立哈希各查 kv/4 维、拼起来，单组碰撞只污染 1/4 的通道。总参数量与“一组 kv 维”相同，只是更稳。",
      "<b>用在哪：</b>和 VE 同一位置——V 投影之后、分头与注意力之前。混合系数 λ 与逐 KV 头门控和 VE 共用一套（每层 4 个 λ）。",
      "<b>训练更新：</b>随机初始化 N(0,0.02) 的嵌入表；梯度只回传到本 batch 命中的那些行（未命中行的梯度为 0、实际不变）。默认归入嵌入参数组（AdamW、单独学习率、无权重衰减）；也可以给它单独学习率，但这次没启用。",
      "<b>中段：</b>缩短后只在块尾 token（每 16 个一次）查，与中段注意力同频；前后段等宽时与前后段共用同一套表。",
      "<b>不是提示表：</b>提示表是只读的长 n-gram 统计（到 20 阶、带指纹），NVE 是可训练的 2/3 阶哈希表，只往 V 里加。"
    ],
    flops: "0 矩阵乘（纯整数哈希 + 查表 + 逐元素混合）；只有门控的小线性层有极少量计算。"
  },
  {
    key: "shorten", name: "压缩 k=16", badge: "分层前向",
    tags: [["训练", "attn"]],
    formula: "x_short[j] = W_down · concat( x[jk-15], …, x[jk] )\n[B, L, h] → [B, L/16, h]\n3e17: [B,3072,640] → [B,192,640]",
    what: "进入中段前，把相邻 16 个 token 的隐状态右移后拼接成 16h 维，再用一个线性层投影回 h 维。第 j 个位置因此只包含 token 15j-15…15j 的信息，严格因果。",
    how: [
      "<b>本质上是一次序列下采样：</b>3072 个 token → 192 个位置（÷16），但特征维 h 不变，所以是“分辨率减半再减半”而不是“通道降维”。",
      "<b>等价形式：</b>对 h 个通道做 kernel=16、stride=16、无 bias 的一维卷积，也就是 1D 版的 patch embedding；每个窗口互不重叠。",
      "<b>为什么省算力：</b>中段每一层只在 192 个位置上算，而全长是 3072 个位置——同样的参数，每个 token 分摊的矩阵乘只有 1/16。",
      "<b>中段 RoPE：</b>用位置 j=0..191（块序号），因果 mask 也是在 192 长度上。",
      "<b>参数：</b>16h×h = 10240×640（3e17）≈ 6.6M，计入 N 与 FLOPs；按 1/16 频率记账，每 token 摊销 2h²。",
      "<b>位置：</b>3e17 从第 4 层开始，1e19 从第 8 层开始（前 20% 层保持全长）。"
    ],
    flops: "每 16 个 token 一次 2·(16h)·h 的矩阵乘 → 每 token 2h²。"
  },
  {
    key: "middle", name: "中段层（缩短序列上）", badge: "Transformer 层",
    tags: [["训练", "attn"], ["训练", "mlp"]],
    formula: "中段：L_mid = ⌈L/16⌉ = 192（评测 3072）\n10/16/17/23 层，参数与全长层同构\n注意力在 192 个位置上是全局的（能看到整条序列的压缩摘要）\n输出门控改为逐通道 [h,h]",
    what: "中段层结构上仍是完整的 Transformer 层，但输入是压缩后的 192 个位置。它们的注意力在压缩序列上做全局注意力——相当于用 1/16 的成本获得整条 3072 token 的全局视野；展开时再把信息按内容门控分回每个 token。",
    how: [
      "中段每层同样有 Q/K/V/O、MLP、QK-Norm、RoPE，形状不变（宽度与前后段相同）。",
      "中段的 VE / NVE 只查块尾位置（每 16 个 token 一次）。",
      "中段输出 z_j 在展开前是 192 个位置的隐状态，展开时 token t 只读 j=⌊t/16⌋ 对应的块（因果）。",
      "中段层的注意力门控用逐通道（h 个门），参数使用频率只有全长层的 1/16。"
    ],
    flops: "中段每层：矩阵乘参数与全长层相同，但只在 L/16 个位置上执行 → 按实际计算量记账。"
  },
  {
    key: "expand", name: "展开 k=16 + 内容门控", badge: "分层前向",
    tags: [["训练", "attn"]],
    formula: "u = W_up · z_j → [B, n, 16h] → reshape [B, n, 16, h]\ntoken t 取 u[⌊t/16⌋][t mod 16]\ng_t = 2σ( W_out·W_in·RMSNorm(a_t) + β_{t mod 16} )   # 秩 64\na_t ← a_t + g_t ⊙ u_t        # a_t 是压缩前的全长隐状态",
    what: "中段算完的 192 个位置要变回 3072 个 token。先把 z 投影回 16h 再按块内位置切开（每块展开到 token jk…jk+15，都不早于 jk，保持因果），然后逐 token 乘一个内容门控：由当前 token 自己的全长隐状态决定“读回多少块信息”。",
    how: [
      "展开不是简单复制：块内 16 个 token 共享同一个 z_j，但每个 token 用自己的门 g_t 放大或抑制它，避免中段信息压过局部细节。",
      "门控是低秩的（h→64→h），零初始化 → 初始化时 g=2σ(0)=1，等价于直接加回。",
      "展开后与压缩前的全长残差 a_t 相加，所以整个中段是“残差里的一个块”，不会切断梯度。"
    ],
    flops: "展开投影每 16 token 一次 2·h·16h；门控每 token 约 4·h·64。"
  },
  {
    key: "hint_cls", name: "提示类别嵌入（可学习）", badge: "训练表",
    tags: [["训练", "hint"]],
    formula: "trust[cls]  = amount[cls]        (初始 0)\nfeat[cls]   = 类别向量            (h 维)\nbonus[cls]  = logit_score[cls]    (初始 0)\n向量 = (1 + trust)·E[tok] + feat\nlogit 加分 = bonus[cls]",
    what: "提示表本身是只读统计，但“该不该信这一类提示”是可学习的。模型只给 631 个类别（每列 42 类 + 空类）各一套小参数，而不是给每个 n-gram 一个参数——统计表里的细节由 token 嵌入承载，信任度按类共享。",
    how: [
      "amount 是 [631,1] 的缩放，feat 是 [631,h] 的类别偏置，logit_score 是 [631,1] 的 logit 加分。",
      "三者都是嵌入表，按规则不计入 N/N_a；它们只有 631×h 量级的参数，极少。",
      "初始值全零 → 模型起步时完全不使用提示，训练中自己学会信多少。"
    ],
    flops: "查表 + 逐元素，无矩阵乘。"
  },
  {
    key: "norm_out", name: "最终 RMSNorm + 输出侧提示", badge: "输出前",
    tags: [["训练", "attn"], ["训练", "hint"]],
    formula: "h = RMSNorm(h)\nh ← h + s_out · hint_rows\n（没有低秩输出投影：直接进输出层）",
    what: "最后一层出来后先做最终 RMSNorm，再把提示向量（与输入侧同一份 hint_rows，复用、不重新查表）按第二个标量 s_out 加一次。输入侧与输出侧共享同一组提示向量，相当于“读入时提示一次、输出时再提醒一次”。",
    how: [
      "两个强度标量：[s_in, s_out] 分别给输入嵌入与输出层之前，都零初始化。",
      "因为 hint_rows 已经在输入侧算好，这里只是复用，没有额外查表开销。"
    ],
    flops: "RMSNorm 是逐元素，不计 FLOPs。"
  },
  {
    key: "lm_head", name: "输出层", badge: "输出",
    tags: [["训练", "mlp"], ["查表", "ve"]],
    formula: "logits = h · W_outᵀ      # [B, L, 49152]\n训练：分阶段只算 12288 / 24576 / 49152 列\n评测：始终全词表",
    what: "标准线性输出层，输出 49152 维 logits。它是大模型里最贵的单层：每 token 2·h·V。为了在小预算下多训 token，训练时分阶段 sampled softmax，只对一部分词表列算 logit（FLOPs 按实际列数计）。",
    how: [
      "<b>评测：</b>全词表，BF16 保持不变（FP8 只用于隐藏层线性层，输出层不转）。",
      "<b>训练：</b>前 60% 预算只算 12288 列，接着 15% 算 24576 列，最后 25% 全词表；除真实目标外按均匀随机取负例列。",
      "<b>参数：</b>输出层不计入 N/N_a，但它的矩阵乘计入 FLOPs（因此小档预算里它占 18%~23%）。"
    ],
    flops: "每 token 2·h·V；训练时按实际列数（12288/24576/49152）记账。"
  },
  {
    key: "hint_logit", name: "提示 logit 加分", badge: "n-gram 提示",
    tags: [["只读表", "hint"], ["训练", "hint"]],
    formula: "logits[:, :, tok_c] += bonus[cls_c]     # 15 列逐个加\n同一 token 被多列命中时：分数相加\n（若开概率混合则是另一条路径，本模型未开）",
    what: "提示不仅以向量形式进入输入 / 输出，还直接给候选 token 的 logit 加分：类别学到的 bonus 加在对应词表列上。这是零矩阵乘的“偏置项”，相当于给统计上靠谱的后继 token 一个可学习的先验。",
    how: [
      "bonus 初始为 0 → 初始时对分布没有任何影响。",
      "只加在候选 token 的列上，不是全词表；其他列不动。",
      "评测时用专用小内核原地加，避免额外写一遍 [B,L,49152] 的 logits。",
      "加上之后才计算交叉熵 / bpb。"
    ],
    flops: "0 矩阵乘。"
  },
  {
    key: "loss", name: "损失与 bpb / EvalTime", badge: "评测",
    tags: [["评测", "attn"]],
    formula: "loss = CrossEntropy(logits, next_token)   # 全词表（评测）\nbpb = Σ NLL / (ln2 × UTF-8 字节数)\nEvalTime = 前向秒数 / ((N_a/1e9) × (计分 token/1e6))\nS = Performance × EvalTime^λ",
    what: "评测指标是 bits-per-byte（bpb，越低越好）与 EvalTime（每 1B 激活参数、每 1M token 的前向秒数）。官方拟合四档 bpb 与训练 FLOPs 的幂律，Performance 由拟合曲线决定，所以四档要“成比例地好”，不能只顾一档。",
    how: [
      "bpb 用原始 text 的 UTF-8 字节数归一化，EOS / padding 不计分但 EOS 参与前向。",
      "EvalTime 的分母只算非词表激活参数 N_a（查表表不计），所以查表架构在这项上占便宜。",
      "本页不展示具体 bpb / EvalTime 数值，只说明指标定义与计算口径。"
    ],
    flops: "评测时全词表输出层，因此每 token 2·h·V 的矩阵乘是 EvalTime 的主要部分之一。"
  }
];

const NODE_TO_MODULE = {
  input: "input", embed: "embed", hint_in: "hint_in",
  layer_full: "layer", middle: "middle", shorten: "shorten", expand: "expand",
  ve: "ve", nve: "nve", hint_tab: "#hint", hint_cls: "hint_cls",
  norm_out: "norm_out", lm_head: "lm_head", hint_logit: "hint_logit", loss: "loss"
};

function badgeHtml(tags) {
  return (tags || []).map(([txt, cls]) => `<span class="flag ${cls}">${txt}</span>`).join("");
}

function applyTier() {
  const t = tier();
  document.querySelectorAll("[data-tier]").forEach(el => {
    const f = el.getAttribute("data-tier");
    let v = t[f];
    if (f === "layers") v = `${t.L} 层（前 ${t.front} / 中 ${t.mid} / 后 ${t.back}）`;
    if (f === "heads") v = `${t.heads} × 128`;
    if (f === "fv") v = `${NUMFMT(t.params)}`;
    if (f === "tl") v = `${t.tensors} 个张量`;
    el.textContent = v;
  });
  document.querySelectorAll(".seg [data-set-tier]").forEach(b => {
    b.classList.toggle("on", b.getAttribute("data-set-tier") === CUR.tier);
  });
  document.querySelectorAll("#tierTable tbody tr").forEach(tr => {
    tr.classList.toggle("hi", tr.getAttribute("data-tier-row") === CUR.tier);
  });
  document.querySelectorAll("[data-tiermid]").forEach(el => {
    const f = el.getAttribute("data-tiermid");
    let v = "";
    if (f === "range") v = `L${t.front} … L${t.front + t.mid - 1}`;
    if (f === "shortenAt") v = `第 ${t.front} 层`;
    if (f === "expandAt") v = `第 ${t.front + t.mid} 层`;
    if (f === "midLen") v = `${Math.round(t.evalSeq / 16)}`;
    if (f === "seq") v = `${NUMFMT(t.evalSeq)}`;
    el.textContent = v;
  });
}

/* ============================================================
   渲染：模块详情
   ============================================================ */
function badgeHtml(tags) {
  return (tags || []).map(([txt, cls]) => `<span class="flag ${cls}">${txt}</span>`).join("");
}

function renderModule(key) {
  const m = MODULES.find(x => x.key === key) || MODULES[0];
  document.querySelectorAll("#modMenu button").forEach(b => b.classList.toggle("on", b.getAttribute("data-mod") === key));
  document.querySelectorAll("svg .node").forEach(n => n.classList.toggle("sel", NODE_TO_MODULE[n.getAttribute("data-node")] === key));
  const t = tier();
  const h = t.h, L = t.L;
  let formula = m.formula
    .replaceAll("{h}", h).replaceAll("{L}", L).replaceAll("{V}", 49152)
    .replaceAll("{heads}", t.heads).replaceAll("{inter}", t.inter)
    .replaceAll("{G}", t.veN).replaceAll("{Lmid}", Math.round(t.evalSeq / 16));
  const el = document.getElementById("modPanel");
  el.innerHTML = `
    <h3>${m.name}</h3>
    <div class="tags">${badgeHtml(m.tags)}<span class="flag ro">当前档位 ${CUR.tier} · h=${h} · L=${L}</span></div>
    <div class="formula">${formula}</div>
    <p style="color:var(--muted);font-size:14.5px;margin:6px 0 0">${m.what}</p>
    <div class="split">
      <div class="mini"><div class="h">怎么参与计算</div><div><ul>${m.how.map(x => `<li>${x}</li>`).join("")}</ul></div></div>
      <div class="mini"><div class="h">FLOPs / 参数口径</div><div>${m.flops}</div></div>
    </div>`;
  el.animate?.([{ opacity: .35, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }], { duration: 220, easing: "ease-out" });
}



function init() {
  try {
    const q = new URLSearchParams(location.search).get("tier");
    if (q && TIERS[q]) CUR.tier = q;
  } catch (e) { /* ignore */ }

  document.querySelectorAll(".seg [data-set-tier]").forEach(b => {
    b.classList.toggle("on", b.getAttribute("data-set-tier") === CUR.tier);
    b.addEventListener("click", () => {
      CUR.tier = b.getAttribute("data-set-tier");
      applyTier();
      renderModule(currentModule);
    });
  });

  const menu = document.getElementById("modMenu");
  menu.innerHTML = MODULES.map(m => `<button data-mod="${m.key}">${m.name}<span class="k">↗</span></button>`).join("");
  menu.querySelectorAll("button").forEach(b => b.addEventListener("click", () => renderModule(b.getAttribute("data-mod"))));

  document.querySelectorAll("svg .node").forEach(n => {
    n.addEventListener("click", () => {
      const mod = NODE_TO_MODULE[n.getAttribute("data-node")];
      if (!mod) return;
      if (mod.startsWith("#")) {
        document.querySelector(mod)?.scrollIntoView({ behavior: "smooth" });
        return;
      }
      renderModule(mod);
      document.getElementById("modPanel").scrollIntoView({ behavior: "smooth", block: "center" });
    });
  });

  const links = [...document.querySelectorAll(".nav a")];
  const secs = links.map(a => document.querySelector(a.getAttribute("href")));
  const updateNav = () => {
    const y = window.scrollY + 150;
    let idx = -1;
    secs.forEach((s, i) => {
      if (!s) return;
      const top = s.getBoundingClientRect().top + window.scrollY;
      if (top <= y) idx = i;
    });
    links.forEach((a, i) => a.classList.toggle("on", i === idx));
  };
  window.addEventListener("scroll", updateNav, { passive: true });
  updateNav();

  applyTier();
  renderModule(currentModule);
}

const currentModule = "layer";
document.addEventListener("DOMContentLoaded", init);
