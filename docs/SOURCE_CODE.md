# 🚀 vLLM 源码深度剖析与架构进阶

> 本文档面向有一年以上 AI Infra 经验的工程师，深入解析 vLLM 的代码实现流程、内存物理布局、分布式设计以及多模态适配机制。

---

## 目录

- [Phase 1: 核心调度与内存分配 (The Brain)](#phase-1-核心调度与内存分配-the-brain)
- [Phase 2: 物理内存管理与 Worker 执行 (The Body)](#phase-2-物理内存管理与-worker-执行-the-body)
- [Phase 3: Kernel 实现与 PagedAttention (The Heart)](#phase-3-kernel-实现与-pagedattention-the-heart)
- [Phase 4: 多模态扩展 (The Eyes)](#phase-4-多模态扩展-the-eyes)
- [完整请求生命周期](#完整请求生命周期)
- [分布式架构设计](#分布式架构设计)
- [实践指南与动手练习](#实践指南与动手练习)

---

## Phase 1: 核心调度与内存分配 (The Brain)

### 核心逻辑解析

vLLM 的调度系统采用了类似 **OS 虚拟内存管理** 的设计哲学。核心思想是：

1. **Scheduler 不做实际推理**，只负责"资源调度决策"
2. **逻辑块（Logical Block）** 对应请求的 token 序列，是调度器视角的抽象
3. **物理块（Physical Block）** 对应 GPU 显存中的实际存储位置

**调度循环的数据流：**

```
┌─────────────────────────────────────────────────────────┐
│  Scheduler.schedule()                                     │
├─────────────────────────────────────────────────────────┤
│  1. 优先调度 RUNNING 队列中的请求                         │
│     ↓                                                    │
│  2. kv_cache_manager.allocate_slots() → 分配 KV 块       │
│     ↓                                                    │
│  3. 如果分配失败 → 执行 Preemption (抢占低优先级请求)      │
│     ↓                                                    │
│  4. 调度 WAITING 队列中的新请求                           │
│     ↓                                                    │
│  5. 返回 SchedulerOutput (包含 block_ids, num_tokens 等)  │
└─────────────────────────────────────────────────────────┘
```

### 关键代码定位

| 组件 | 文件路径 | 核心类/函数 | 作用 |
|------|---------|------------|------|
| **调度器主逻辑** | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule()` (L191-L715) | 每个调度步的核心决策 |
| **KV 缓存管理器** | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.allocate_slots()` (L217-L332) | 分配/释放逻辑块 |
| **块池管理** | `vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock` (L107-L152) | 物理块的元数据结构 |
| **块哈希计算** | `vllm/v1/core/kv_cache_utils.py` | `BlockHash`, `make_block_hash_with_group_id()` | Prefix Caching 的块标识 |
| **协调器** | `vllm/v1/core/kv_cache_coordinator.py` | `get_kv_cache_coordinator()` | 管理多种 KV 缓存策略 |

**逻辑块 → 物理块映射的核心数据结构：**

```python
# vllm/v1/core/kv_cache_utils.py:107
@dataclass
class KVCacheBlock:
    block_id: int           # 物理块 ID (0 ~ num_gpu_blocks-1)
    ref_cnt: int = 0        # 引用计数 (Prefix Caching 共享)
    _block_hash: BlockHashWithGroupId | None = None  # 用于缓存命中检测
```

**Preemption 决策逻辑：**

在 `scheduler.py:279-337`，当 `allocate_slots()` 返回 `None` 时：

```python
# vllm/v1/core/sched/scheduler.py:290-296
if self.policy == SchedulingPolicy.PRIORITY:
    preempted_req = max(
        self.running,
        key=lambda r: (r.priority, r.arrival_time),  # 抢占优先级最低的
    )
else:
    preempted_req = self.running.pop()  # FCFS: 抢占最后加入的

self.kv_cache_manager.free(preempted_req)  # 释放其所有块
preempted_req.status = RequestStatus.PREEMPTED
preempted_req.num_computed_tokens = 0  # 重置，需要 Recomputation
```

### 自测问题 (Checklist)

1. **当 KV Cache 满了，vLLM 具体在代码的哪一行决定驱逐哪个 Sequence？**
   - 答案线索：查看 `scheduler.py:279-337` 中的 preemption 循环，特别是 `max(self.running, key=...)` 和 `self.running.pop()` 的选择逻辑

2. **Prefix Caching 是如何判断两个请求可以共享同一个 Block 的？`BlockHash` 是如何计算的？**
   - 答案线索：查看 `kv_cache_utils.py:47-67` 的 `make_block_hash_with_group_id()` 和 `kv_cache_manager.py:175-215` 的 `get_computed_blocks()`

3. **如果一个被 preempted 的请求重新被调度，它的 `num_computed_tokens` 为什么是 0？这意味着什么？**
   - 答案线索：`scheduler.py:326` 设置 `num_computed_tokens = 0`，意味着需要完全重新计算（Recomputation），因为其 KV Cache 已被释放

---

## Phase 2: 物理内存管理与 Worker 执行 (The Body)

### 核心逻辑解析

这是 **CPU 调度决策 → GPU 物理执行** 的桥梁。关键难点在于理解 `BlockTable` 如何从 Python 对象转换为 GPU Kernel 可访问的 Tensor。

**数据流转过程：**

```
┌──────────────────────────────────────────────────────────────┐
│  SchedulerOutput (CPU)                                        │
│  ├─ scheduled_new_reqs: [NewRequestData, ...]                │
│  ├─ scheduled_cached_reqs: CachedRequestData                 │
│  └─ num_scheduled_tokens: {req_id: num_tokens}               │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  GPUModelRunner._prepare_inputs() (CPU → GPU)                │
│  ├─ 构建 input_ids tensor                                    │
│  ├─ 构建 positions tensor                                    │
│  ├─ 更新 InputBatch.block_table (关键!)                      │
│  └─ 构建 AttentionMetadata                                   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Model Forward (GPU)                                          │
│  Attention Kernel 通过 block_tables tensor 访问 KV Cache     │
└──────────────────────────────────────────────────────────────┘
```

### 关键代码定位

| 组件 | 文件路径 | 核心类/函数 | 作用 |
|------|---------|------------|------|
| **GPU Worker** | `vllm/v1/worker/gpu_worker.py` | `Worker.execute_model()` (L543-L600+) | 调用 ModelRunner |
| **模型运行器** | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner.execute_model()` (L2650+) | 核心执行入口 |
| **输入批次管理** | `vllm/v1/worker/gpu_input_batch.py` | `InputBatch` | 管理 block_table tensor |
| **Attention 元数据构建** | `vllm/v1/attention/backends/utils.py` | `AttentionMetadataBuilder` | 构建传给 Kernel 的元数据 |

**BlockTable 传输机制 (关键难点)：**

在 `gpu_input_batch.py` 中，`InputBatch` 类维护了一个 **固定大小的 GPU Tensor** 用于存储 block tables：

```python
# vllm/v1/worker/gpu_input_batch.py (概念示例)
class InputBatch:
    def __init__(self, ...):
        # 预分配 GPU tensor，避免每次调度都重新分配
        self.block_table = torch.zeros(
            (max_num_reqs, max_blocks_per_seq),
            dtype=torch.int32,
            device=device
        )

    def update_block_table(self, req_id, block_ids):
        # 将调度器返回的 block_ids 写入对应位置
        req_index = self.req_id_to_index[req_id]
        self.block_table[req_index, :len(block_ids)] = torch.tensor(block_ids)
```

**MetaData 组装流程：**

在 `gpu_model_runner.py:2650+` 的 `execute_model()` 中：

1. 调用 `_update_states()` 处理新请求和缓存请求
2. 调用 `_prepare_inputs()` 构建 `input_ids`, `positions` 等
3. 调用 `_build_attention_metadata()` 或使用 `AttentionMetadataBuilder` 构建 attention metadata
4. 将所有数据传给模型 `forward()`

### 自测问题 (Checklist)

1. **在执行 `forward` 之前，MetaData（如 block tables, context lens）是如何被组装并传给 CUDA Kernel 的？**
   - 答案线索：查看 `gpu_model_runner.py` 中的 `_prepare_inputs()` 和 `gpu_input_batch.py` 中的 `InputBatch` 类的 `block_table` 字段更新逻辑

2. **`CacheEngine` 是如何初始化 GPU 上的 KV Cache 显存池的？显存是何时分配的？**
   - 答案线索：在 Worker 初始化阶段，通过 `initialize_kv_cache()` 方法分配，查看 `gpu_worker.py` 中相关代码

3. **为什么 vLLM 使用固定大小的预分配 Tensor 而不是动态分配？**
   - 答案：避免 CUDA 内存分配开销；支持 CUDA Graph 捕获（需要固定内存地址）

---

## Phase 3: Kernel 实现与 PagedAttention (The Heart)

### 核心逻辑解析

PagedAttention 的核心创新是：**允许 KV Cache 在物理内存中非连续存储，通过 block_table 间接寻址**。

这类似于 OS 的页表机制：
- **虚拟地址** → **Token 在序列中的位置**
- **页表** → **block_table**
- **物理页帧** → **GPU 显存中的 KV Block**

**Kernel 内部寻址逻辑：**

```
Token Position (e.g., 150)
     │
     ▼
Block Index = 150 / block_size (e.g., 150/16 = 9)
     │
     ▼
Physical Block ID = block_table[seq_id][9] (e.g., 42)
     │
     ▼
KV Cache Address = kv_cache_base + 42 * block_size * head_dim
```

### 关键代码定位

| 组件 | 文件路径 | 核心类/函数 | 作用 |
|------|---------|------------|------|
| **PagedAttention 接口** | `vllm/attention/ops/paged_attn.py` | `PagedAttention` 类 (L41-L263) | 封装 CUDA ops 调用 |
| **Decode Attention** | `paged_attn.py:94-199` | `forward_decode()` | Decode 阶段 (单 token) |
| **Prefill Attention** | `paged_attn.py:201-239` | `forward_prefix()` | Prefill 阶段 (多 tokens) |
| **底层 CUDA 算子** | `vllm/_custom_ops.py` | `ops.paged_attention_v1/v2()` | 实际 CUDA kernel |
| **Triton 实现** | `vllm/attention/ops/prefix_prefill.py` | `context_attention_fwd()` | Triton 版本的 prefill |

**Kernel 签名分析：**

```python
# vllm/attention/ops/paged_attn.py:94-112
@staticmethod
def forward_decode(
    query: torch.Tensor,           # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,       # [num_blocks, num_kv_heads, head_size/x, block_size, x]
    value_cache: torch.Tensor,     # [num_blocks, num_kv_heads, head_size, block_size]
    block_tables: torch.Tensor,    # [num_seqs, max_blocks_per_seq] ← 关键！
    seq_lens: torch.Tensor,        # [num_seqs]
    max_seq_len: int,
    kv_cache_dtype: str,
    num_kv_heads: int,
    scale: float,
    ...
) -> torch.Tensor:
```

**V1 vs V2 的选择启发式：**

```python
# vllm/attention/ops/paged_attn.py:134-136
use_v1 = max_seq_len <= 8192 and (
    max_num_partitions == 1 or num_seqs * num_heads > 512
)
```

- **V1**: 不分区，适合短序列或大批量
- **V2**: 分区计算后归约，适合长序列（避免 shared memory 不足）

### FlashInfer 集成

vLLM 也支持 FlashInfer 作为高性能后端：

```
vllm/v1/attention/backends/flashinfer.py
```

FlashInfer 提供了更高效的 PagedAttention 实现，特别是在解码阶段。

### 自测问题 (Checklist)

1. **PagedAttention Kernel 中，一个 Thread Block 处理多少个 Token？它是如何处理 Memory Coalescing（内存合并访问）的？**
   - 答案线索：查看 CUDA kernel 实现中的 `PARTITION_SIZE = 512` 和 key_cache 的特殊布局 `[num_blocks, num_kv_heads, head_size/x, block_size, x]`，其中 `x = 16 // element_size` 是为了对齐内存访问

2. **`forward_prefix()` 和 `forward_decode()` 有什么区别？为什么要分开实现？**
   - 答案：Prefill 处理多个 query tokens，使用 `context_attention_fwd`（类似 FlashAttention）；Decode 每次只处理一个 token，使用专门优化的 `paged_attention_v1/v2`

3. **KV Cache 的 shape 为什么是 `(2, num_blocks, block_size * num_kv_heads * head_size)` 而不是直接存储？**
   - 答案线索：查看 `paged_attn.py:47-54` 的 `get_kv_cache_shape()` 和 `split_kv_cache()` 方法，这种布局便于 block swap 和 copy 操作

---

## Phase 4: 多模态扩展 (The Eyes)

### 核心逻辑解析

vLLM 的多模态架构设计非常优雅，核心思想是：

1. **Image/Audio Features 被转换为 "Token-like" 的 embeddings**
2. **这些 embeddings 替换 placeholder tokens 的位置**
3. **它们也占用 KV Cache blocks**（这是很多人忽略的！）

**数据流：**

```
┌──────────────────────────────────────────────────────────┐
│  Raw Input: "Describe <image> this picture"              │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  Tokenizer: ["Describe", "<image>", "this", "picture"]   │
│             其中 <image> 是 placeholder token            │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  MultiModalProcessor:                                     │
│  1. Vision Encoder: image → [576 个 feature vectors]     │
│  2. 计算需要多少个 placeholder tokens (e.g., 576)        │
│  3. 在 embeddings 中替换对应位置                         │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────────────┐
│  Model Forward:                                           │
│  embeddings = [Describe_emb, img_feat_1, ..., img_feat_576, this_emb, picture_emb]
│  ↓                                                       │
│  这些 embeddings 全部经过 Attention，产生 KV Cache       │
└──────────────────────────────────────────────────────────┘
```

### 关键代码定位

| 组件 | 文件路径 | 核心类/函数 | 作用 |
|------|---------|------------|------|
| **多模态注册表** | `vllm/multimodal/registry.py` | `MultiModalRegistry` (L93-L361) | 模型到处理器的映射 |
| **处理器基类** | `vllm/multimodal/processing.py` | `BaseMultiModalProcessor` | 输入预处理 |
| **输入类型** | `vllm/multimodal/inputs.py` | `MultiModalKwargsItem`, `PlaceholderRange` | 数据结构定义 |
| **Encoder 缓存管理** | `vllm/v1/core/encoder_cache_manager.py` | `EncoderCacheManager` | 缓存 vision encoder 输出 |
| **LLaVA 实现** | `vllm/model_executor/models/llava.py` | `LlavaForConditionalGeneration` | 具体模型实现 |
| **Qwen2-VL 实现** | `vllm/model_executor/models/qwen2_5_vl.py` | `Qwen2_5_VLForConditionalGeneration` | 支持 M-RoPE |

**Placeholder Token 计算：**

在调度阶段，需要预先计算 image 需要多少个 token slots：

```python
# vllm/v1/core/sched/scheduler.py:848-857
for i, mm_feature in enumerate(mm_features):
    start_pos = mm_feature.mm_position.offset      # placeholder 开始位置
    num_encoder_tokens = mm_feature.mm_position.length  # 需要的 token 数量

    # 检查这个范围是否与当前调度的 token 范围重叠
    if start_pos >= num_computed_tokens + num_new_tokens:
        break  # 还没到这个 encoder input
```

**EncoderCacheManager 的作用：**

```python
# 在 scheduler.py 中
self.encoder_cache_manager = EncoderCacheManager(cache_size=encoder_cache_size)

# 当 encoder input 被处理后，缓存其输出
self.encoder_cache_manager.allocate(request, i)  # i 是 mm_feature 的索引

# 当 decoder 完成该位置的处理后，释放缓存
self.encoder_cache_manager.free_encoder_input(request, input_id)
```

### 自测问题 (Checklist)

1. **对于一张高分辨率图片，vLLM 如何预先计算它需要占用多少个 Token slots，以防止调度时显存不足？**
   - 答案线索：查看 `multimodal/registry.py:150-175` 的 `get_max_tokens_per_item_by_modality()` 和 `multimodal/profiling.py` 中的 `MultiModalProfiler`

2. **Image Features 是否占用 KV Cache Block？如果占用，它们的 block hash 是如何计算的？**
   - 答案：是的，占用！因为它们被转换为 embeddings 后参与 Attention 计算。Hash 计算在 `kv_cache_utils.py` 中，会考虑 `mm_feature.identifier`（如 image hash）

3. **`EncoderCacheManager` 和 `KVCacheManager` 有什么区别？为什么需要两个缓存管理器？**
   - 答案：`EncoderCacheManager` 缓存 vision encoder 的输出（固定大小，与请求关联）；`KVCacheManager` 管理 decoder 的 KV Cache（动态增长）。前者是为了避免重复计算同一张图片的 encoder 输出

---

## 完整请求生命周期

### LLM 文本请求完整流程

下面详细追踪一个文本请求从进入系统到返回结果的完整路径：

```
用户请求: "What is the capital of France?"
```

#### 阶段 1: API 层接收请求

**代码路径：**
```
vllm/entrypoints/openai/api_server.py
  → vllm/entrypoints/openai/serving_chat.py
    → create_chat_completion()
```

**输入：**
```python
{
    "model": "llama-7b",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 100,
    "temperature": 0.7
}
```

**输出：**
```python
# 转换为内部请求格式
prompt = "What is the capital of France?"
sampling_params = SamplingParams(max_tokens=100, temperature=0.7)
```

**为什么这么处理：** 统一不同 API 格式（OpenAI、HuggingFace 等）到内部表示

---

#### 阶段 2: Engine 接收并预处理

**代码路径：**
```
vllm/v1/engine/llm_engine.py
  → LLMEngine.add_request()
    → vllm/v1/engine/processor.py
      → Processor.process_inputs()
```

**处理内容：**
```python
# Tokenization
token_ids = tokenizer.encode("What is the capital of France?")
# 输出: [1, 1724, 338, 278, 7483, 310, 3444, 29973]

# 构建 Request 对象
request = Request(
    request_id="req-001",
    prompt_token_ids=token_ids,
    sampling_params=sampling_params,
    arrival_time=time.monotonic(),
)

# 计算 block hashes (用于 prefix caching)
request.block_hashes = compute_block_hashes(token_ids, block_size=16)
```

**输出：**
```python
Request(
    request_id="req-001",
    prompt_token_ids=[1, 1724, 338, 278, 7483, 310, 3444, 29973],
    num_prompt_tokens=8,
    num_computed_tokens=0,
    status=RequestStatus.WAITING,
    block_hashes=[hash1]  # 8 tokens < 16, 只需要 1 个 block
)
```

**为什么这么处理：**
- 预计算 block hashes 使得后续 prefix caching 查找 O(1)
- 统一请求格式便于调度器处理

---

#### 阶段 3: 调度器决策

**代码路径：**
```
vllm/v1/core/sched/scheduler.py
  → Scheduler.schedule()
    → KVCacheManager.get_computed_blocks()  # 检查缓存命中
    → KVCacheManager.allocate_slots()       # 分配物理块
```

**处理内容：**
```python
# 1. 检查 prefix cache 命中
computed_blocks, num_computed_tokens = kv_cache_manager.get_computed_blocks(request)
# 假设没有命中: computed_blocks=[], num_computed_tokens=0

# 2. 计算需要调度的 token 数
num_new_tokens = request.num_tokens - num_computed_tokens  # 8 - 0 = 8

# 3. 分配物理块
# 8 tokens, block_size=16, 需要 1 个 block
new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens=8)
# 返回: KVCacheBlocks(blocks=([KVCacheBlock(block_id=42)],))

# 4. 更新请求状态
request.status = RequestStatus.RUNNING
request.num_computed_tokens = 8
```

**输出：**
```python
SchedulerOutput(
    scheduled_new_reqs=[NewRequestData(
        req_id="req-001",
        prompt_token_ids=[1, 1724, 338, 278, 7483, 310, 3444, 29973],
        block_ids=([42],),  # 分配的物理块 ID
    )],
    num_scheduled_tokens={"req-001": 8},
    total_num_scheduled_tokens=8,
)
```

**为什么这么处理：**
- Prefix caching 检查可以复用之前计算的 KV
- 批量分配 blocks 而不是逐 token 分配，减少管理开销
- 返回完整的调度结果，便于 Worker 执行

---

#### 阶段 4: Worker 准备输入

**代码路径：**
```
vllm/v1/worker/gpu_worker.py
  → Worker.execute_model()
    → vllm/v1/worker/gpu_model_runner.py
      → GPUModelRunner._update_states()     # 更新请求状态
      → GPUModelRunner._prepare_inputs()    # 准备 GPU tensor
```

**处理内容：**
```python
# 1. 更新 InputBatch 状态
input_batch.add_request(
    req_id="req-001",
    token_ids=[1, 1724, 338, 278, 7483, 310, 3444, 29973],
    block_ids=[42],
)

# 2. 构建 GPU tensors
input_ids = torch.tensor([1, 1724, 338, 278, 7483, 310, 3444, 29973], device="cuda")
positions = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], device="cuda")

# 3. 更新 block_table tensor (关键!)
# block_table[req_index] = [42, 0, 0, ...]  # 后面补零
input_batch.block_table[0, 0] = 42

# 4. 构建 attention metadata
attn_metadata = AttentionMetadata(
    num_prefill_tokens=8,
    num_decode_tokens=0,
    seq_lens=[8],
    block_tables=input_batch.block_table[:1],  # 只取第一个请求
    ...
)
```

**输出：**
```python
# GPU 上的 tensors
input_ids: torch.Tensor([1, 1724, 338, 278, 7483, 310, 3444, 29973])  # [8]
positions: torch.Tensor([0, 1, 2, 3, 4, 5, 6, 7])  # [8]
attn_metadata: AttentionMetadata(...)
```

**为什么这么处理：**
- 预分配固定大小 tensor 支持 CUDA Graph
- Block table 映射逻辑块到物理块，实现非连续内存访问
- 批量处理多个请求提高 GPU 利用率

---

#### 阶段 5: 模型前向传播 (Prefill)

**代码路径：**
```
vllm/v1/worker/gpu_model_runner.py
  → GPUModelRunner.execute_model()
    → model.forward()
      → vllm/model_executor/models/llama.py
        → LlamaForCausalLM.forward()
          → 各层的 Attention + FFN
```

**处理内容：**
```python
# 1. Embedding
hidden_states = embed_tokens(input_ids)  # [8, 4096]

# 2. 各 Transformer 层
for layer in layers:
    # Self-Attention with PagedAttention
    # 这里会将 K, V 写入 block_id=42 的物理块
    attn_output = layer.self_attn(
        hidden_states,
        positions,
        kv_cache,        # GPU 上的 KV 缓存池
        attn_metadata,   # 包含 block_tables
    )

    # FFN
    hidden_states = layer.mlp(attn_output)

# 3. LM Head
logits = lm_head(hidden_states[-1:])  # 只取最后一个位置 [1, vocab_size]
```

**KV Cache 写入细节：**
```python
# 在 PagedAttention.write_to_paged_cache() 中
# slot_mapping 计算: token_position % block_size + block_id * block_size
# 例如: token 0~7 写入 block_id=42 的 slot 0~7

ops.reshape_and_cache(
    key,          # [8, num_heads, head_dim]
    value,        # [8, num_heads, head_dim]
    key_cache,    # [num_blocks, ...]
    value_cache,  # [num_blocks, ...]
    slot_mapping, # [8] -> [42*16+0, 42*16+1, ..., 42*16+7]
)
```

**输出：**
```python
logits: torch.Tensor  # [1, 32000] 最后一个位置的 logits
# KV Cache 已写入 block_id=42
```

**为什么这么处理：**
- Prefill 阶段一次性处理所有 prompt tokens
- KV Cache 写入物理块，后续 decode 可直接读取
- 只返回最后一个位置的 logits 用于采样

---

#### 阶段 6: 采样

**代码路径：**
```
vllm/v1/worker/gpu_model_runner.py
  → GPUModelRunner.execute_model()
    → Sampler.forward()
      → vllm/v1/sample/sampler.py
```

**处理内容：**
```python
# 1. 应用 logits processor
logits = logits / temperature  # temperature=0.7

# 2. Top-p/Top-k 采样
probs = softmax(logits)
# Top-p 采样...

# 3. 采样 token
sampled_token = torch.multinomial(probs, num_samples=1)
# 假设采样结果: 450 (对应 "The")
```

**输出：**
```python
SamplerOutput(
    sampled_token_ids=[[450]],  # "The"
)
```

---

#### 阶段 7: 循环 Decode

现在进入 decode 阶段，每次只处理一个 token：

**调度器：**
```python
# 请求已在 RUNNING 队列
num_new_tokens = 1  # decode 每次只生成 1 个 token
# 可能需要分配新的 block (当前 block 满了)
```

**Worker：**
```python
# 输入只有 1 个 token
input_ids = torch.tensor([450])  # "The"
positions = torch.tensor([8])     # 第 9 个位置

# Decode attention: query 只有 1 个，但 key/value 有 9 个
```

**循环直到：**
- 生成 EOS token
- 达到 max_tokens
- 其他停止条件

---

#### 阶段 8: 返回结果

**代码路径：**
```
vllm/v1/core/sched/scheduler.py
  → Scheduler.update_from_output()
    → 检查停止条件
    → 构建 EngineCoreOutput

vllm/v1/engine/output_processor.py
  → 解码 token ids 为文本
```

**输出：**
```python
CompletionOutput(
    text="The capital of France is Paris.",
    token_ids=[450, 7483, 310, 3444, 338, 3681, 29889],
    finish_reason="stop",
)
```

---

### 多模态请求完整流程

以 LLaVA 模型处理图文请求为例：

```
用户请求:
{
    "prompt": "Describe this image: <image>",
    "images": [PIL.Image]
}
```

#### 阶段 1-2: 与文本请求相同

API 层接收，Engine 预处理

#### 阶段 2.5: 多模态处理 (额外步骤)

**代码路径：**
```
vllm/v1/engine/processor.py
  → Processor.process_inputs()
    → vllm/multimodal/registry.py
      → MultiModalRegistry.create_processor()
        → vllm/multimodal/processing.py
          → BaseMultiModalProcessor.apply()
```

**处理内容：**
```python
# 1. Tokenize 文本
token_ids = tokenizer.encode("Describe this image: <image>")
# [1, 4002, 29581, 445, 1967, 29901, 32000]
# 其中 32000 是 <image> placeholder token

# 2. 处理图像
image_processor = model.get_image_processor()
pixel_values = image_processor(image)  # [1, 3, 336, 336]

# 3. 计算图像需要的 token 数量
# LLaVA: 336/14 * 336/14 = 576 个 patches
num_image_tokens = 576

# 4. 构建 MultiModal 信息
mm_features = [
    MultiModalFeature(
        mm_position=PlaceholderRange(offset=6, length=576),  # <image> 在位置 6
        pixel_values=pixel_values,
        identifier=hash(image),  # 用于缓存
    )
]

# 5. 扩展 token ids (替换 placeholder)
# [1, 4002, 29581, 445, 1967, 29901, 32000, 32000, ...(576个), ...]
expanded_token_ids = expand_with_placeholders(token_ids, num_image_tokens)
```

**输出：**
```python
Request(
    request_id="req-002",
    prompt_token_ids=expanded_token_ids,  # 长度 = 6 + 576 = 582
    mm_features=mm_features,
    num_computed_tokens=0,
)
```

**为什么这么处理：**
- Placeholder tokens 确保文本和图像 token 位置正确
- 预计算图像 token 数量，便于调度器分配内存
- 图像 hash 支持 encoder cache 复用

---

#### 阶段 3: 调度器决策 (考虑 encoder budget)

**代码路径：**
```
vllm/v1/core/sched/scheduler.py
  → Scheduler.schedule()
    → _try_schedule_encoder_inputs()
```

**处理内容：**
```python
# 1. 检查 encoder budget
if encoder_compute_budget < num_image_tokens:  # 576
    # 无法调度，需要等待
    return None

# 2. 分配 KV Cache blocks
# 582 tokens / 16 = 37 blocks
num_blocks = cdiv(582, 16)
new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens=582)

# 3. 分配 encoder cache
encoder_cache_manager.allocate(request, mm_feature_index=0)

# 4. 记录需要调度的 encoder inputs
scheduled_encoder_inputs["req-002"] = [0]  # 第 0 个 mm_feature
```

**输出：**
```python
SchedulerOutput(
    scheduled_new_reqs=[...],
    num_scheduled_tokens={"req-002": 582},
    scheduled_encoder_inputs={"req-002": [0]},  # 需要执行 vision encoder
)
```

---

#### 阶段 4: Worker 执行 Vision Encoder

**代码路径：**
```
vllm/v1/worker/gpu_model_runner.py
  → GPUModelRunner.execute_model()
    → _execute_mm_encoder()
```

**处理内容：**
```python
# 1. 执行 Vision Encoder
image_features = vision_tower(pixel_values)  # [1, 576, 1024]

# 2. 通过 projector
image_features = mm_projector(image_features)  # [1, 576, 4096]

# 3. 缓存结果 (用于可能的复用)
self.encoder_cache[mm_hash] = image_features
```

---

#### 阶段 5: 模型前向传播 (替换 embeddings)

**代码路径：**
```
vllm/model_executor/models/llava.py
  → LlavaForConditionalGeneration.forward()
```

**处理内容：**
```python
# 1. 获取文本 embeddings
inputs_embeds = embed_tokens(input_ids)  # [582, 4096]

# 2. 替换 placeholder 位置的 embeddings
# positions 6~581 替换为 image_features
inputs_embeds[6:582] = image_features.squeeze(0)

# 3. 正常的 Transformer 前向传播
for layer in layers:
    hidden_states = layer(inputs_embeds, ...)
```

**为什么这么处理：**
- Image features 被视为特殊的 "tokens"
- 统一了文本和图像的处理流程
- KV Cache 正常存储，不区分来源

---

## 分布式架构设计

### 整体架构

vLLM 支持多种分布式策略，核心是 **Tensor Parallelism (TP)** 和 **Pipeline Parallelism (PP)**：

```
┌─────────────────────────────────────────────────────────────┐
│                    EngineCore (单实例)                        │
│  ┌─────────┐    ┌───────────┐    ┌─────────────────────┐    │
│  │Scheduler│ →  │ Executor  │ →  │ Workers (多实例)    │    │
│  └─────────┘    └───────────┘    │  ┌─────┐ ┌─────┐   │    │
│                                   │  │GPU 0│ │GPU 1│   │    │
│                                   │  └─────┘ └─────┘   │    │
│                                   └─────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 关键代码定位

| 组件 | 文件路径 | 核心类 | 作用 |
|------|---------|--------|------|
| **Executor 抽象** | `vllm/v1/executor/abstract.py` | `Executor` | 执行器接口 |
| **单进程执行器** | `vllm/v1/executor/uniproc_executor.py` | `UniProcExecutor` | 单 GPU |
| **多进程执行器** | `vllm/v1/executor/multiproc_executor.py` | `MultiprocExecutor` | 多 GPU (TP/PP) |
| **Ray 分布式执行器** | `vllm/v1/executor/ray_executor.py` | `RayDistributedExecutor` | 多节点 |
| **并行状态管理** | `vllm/distributed/parallel_state.py` | 各种 group 函数 | TP/PP group 管理 |
| **通信原语** | `vllm/distributed/communication_op.py` | `tensor_model_parallel_all_reduce` 等 | 集合通信 |

### Tensor Parallelism (TP)

**原理：** 将权重矩阵按列或行切分到多个 GPU

```
         GPU 0          GPU 1
        ┌─────┐        ┌─────┐
Input → │W[:, │  →     │W[:, │ → AllReduce → Output
        │:2048]        │2048:]
        └─────┘        └─────┘
```

**代码实现：**

```python
# vllm/model_executor/layers/linear.py

class ColumnParallelLinear(nn.Module):
    """
    按列切分权重，每个 GPU 存储 weight[:, start:end]
    """
    def forward(self, x):
        # 每个 GPU 计算部分输出
        output = F.linear(x, self.weight)  # [batch, hidden/tp]
        # 结果拼接 (AllGather) 或保持分片
        return output

class RowParallelLinear(nn.Module):
    """
    按行切分权重，每个 GPU 存储 weight[start:end, :]
    """
    def forward(self, x):
        # 每个 GPU 计算部分结果
        output = F.linear(x, self.weight)
        # AllReduce 求和
        output = tensor_model_parallel_all_reduce(output)
        return output
```

**TP 通信优化：**

```python
# vllm/distributed/communication_op.py

def tensor_model_parallel_all_reduce(tensor):
    """
    跨 TP group 的 AllReduce
    使用 NCCL 实现高效通信
    """
    if get_tensor_model_parallel_world_size() == 1:
        return tensor
    torch.distributed.all_reduce(tensor, group=get_tensor_model_parallel_group())
    return tensor
```

### Pipeline Parallelism (PP)

**原理：** 将模型层切分到多个 GPU，形成流水线

```
GPU 0 (Layers 0-15)  →  GPU 1 (Layers 16-31)
     │                       │
     └── Send hidden ──────→ │
                             └── Continue forward
```

**代码实现：**

```python
# vllm/v1/worker/gpu_worker.py

def execute_model(self, scheduler_output):
    # PP 第一阶段
    if get_pp_group().is_first_rank:
        # 从 input_ids 开始
        intermediate_tensors = None
    else:
        # 接收上一阶段的 hidden states
        intermediate_tensors = recv_from_prev_pp_rank()

    # 执行本阶段的层
    output = self.model_runner.execute_model(
        scheduler_output,
        intermediate_tensors,
    )

    # PP 最后阶段
    if get_pp_group().is_last_rank:
        return output  # 返回最终结果
    else:
        # 发送给下一阶段
        send_to_next_pp_rank(output)
        return None
```

**PP 调度策略：**

```python
# 微批次调度 (1F1B)
# F: Forward, B: Backward (推理时只有 Forward)

# 简化的推理 pipeline:
for micro_batch in micro_batches:
    if is_first_stage:
        hidden = embed(micro_batch)
        send(hidden, next_stage)
    elif is_last_stage:
        hidden = recv(prev_stage)
        output = lm_head(forward_layers(hidden))
        outputs.append(output)
    else:
        hidden = recv(prev_stage)
        hidden = forward_layers(hidden)
        send(hidden, next_stage)
```

### Data Parallelism (DP)

**原理：** 每个 GPU 处理不同的请求

```
        Request 1 → GPU 0
Scheduler
        Request 2 → GPU 1
```

**代码实现：**

```python
# vllm/v1/worker/dp_utils.py

def coordinate_batch_across_dp(scheduler_output):
    """
    在 DP ranks 之间协调批次
    """
    # 每个 DP rank 处理不同的请求
    my_requests = scheduler_output.filter_by_dp_rank(get_dp_rank())
    return my_requests
```

### KV Cache 在分布式下的处理

**TP 下的 KV Cache：**

```python
# 每个 TP rank 只存储 num_kv_heads / tp_size 个 head 的 KV
# 例如 tp_size=2, num_kv_heads=8
# GPU 0: heads 0-3, GPU 1: heads 4-7

kv_cache_shape = (
    2,  # K and V
    num_blocks,
    block_size * (num_kv_heads // tp_size) * head_size
)
```

**PP 下的 KV Cache：**

```python
# 每个 PP stage 只存储自己层的 KV Cache
# GPU 0 (layers 0-15): kv_caches[0:16]
# GPU 1 (layers 16-31): kv_caches[16:32]
```

### 分布式初始化流程

```python
# vllm/distributed/parallel_state.py

def initialize_model_parallel(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    ...
):
    """
    初始化各种 process groups
    """
    # 1. TP group: 同一 PP stage 的所有 ranks
    # 2. PP group: 同一 TP position 的所有 ranks
    # 3. DP group: 相同 TP+PP position 的 ranks

    # 创建 NCCL groups
    for ranks in tp_groups:
        group = torch.distributed.new_group(ranks, backend="nccl")
        set_tensor_model_parallel_group(group)

    for ranks in pp_groups:
        group = torch.distributed.new_group(ranks, backend="nccl")
        set_pipeline_model_parallel_group(group)
```

### 分布式自测问题

1. **在 TP=2 的配置下，Attention 的 Q, K, V 投影是如何切分的？AllReduce 发生在哪里？**
   - 答案线索：查看 `model_executor/layers/linear.py` 的 `QKVParallelLinear`

2. **PP 模式下，中间层的 hidden states 是如何在 GPU 之间传输的？使用什么通信原语？**
   - 答案线索：查看 `distributed/communication_op.py` 的 `send_to_next_pp_rank()` 和 `recv_from_prev_pp_rank()`

3. **为什么 vLLM 的 Scheduler 是单点的，不做分布式调度？**
   - 答案：保持调度逻辑简单一致；避免分布式一致性问题；CPU 调度开销远小于 GPU 计算

---

## 实践指南与动手练习

### 环境准备

```bash
# 1. Clone vLLM 仓库
git clone https://github.com/vllm-project/vllm.git
cd vllm

# 2. 创建开发环境
conda create -n vllm-dev python=3.10
conda activate vllm-dev

# 3. 安装开发依赖
pip install -e ".[dev]"

# 4. 安装调试工具
pip install ipdb py-spy
```

### 练习 1: 追踪请求生命周期 (难度: ⭐⭐)

**目标：** 理解一个请求从进入到返回的完整路径

**步骤：**

1. **添加调试打印**

```python
# 修改 vllm/v1/core/sched/scheduler.py
def schedule(self) -> SchedulerOutput:
    print(f"\n{'='*50}")
    print(f"[Scheduler] Starting schedule step")
    print(f"[Scheduler] Waiting queue: {len(self.waiting)}")
    print(f"[Scheduler] Running queue: {len(self.running)}")

    # ... 原有代码 ...

    print(f"[Scheduler] Scheduled {total_num_scheduled_tokens} tokens")
    print(f"[Scheduler] New requests: {[r.req_id for r in scheduled_new_reqs]}")
    return scheduler_output
```

2. **运行简单测试**

```python
# test_trace.py
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")  # 小模型便于测试
outputs = llm.generate(
    ["What is AI?"],
    SamplingParams(max_tokens=10)
)
print(outputs[0].outputs[0].text)
```

3. **观察输出**

```bash
python test_trace.py

# 预期看到类似输出:
# ==================================================
# [Scheduler] Starting schedule step
# [Scheduler] Waiting queue: 1
# [Scheduler] Running queue: 0
# [Scheduler] Scheduled 5 tokens (prefill)
# ==================================================
# [Scheduler] Starting schedule step
# [Scheduler] Waiting queue: 0
# [Scheduler] Running queue: 1
# [Scheduler] Scheduled 1 tokens (decode)
# ...
```

**思考题：**
- 为什么 prefill 调度了 5 个 tokens 而 decode 只调度 1 个？
- `waiting` 和 `running` 队列的变化说明了什么？

---

### 练习 2: 可视化 Block 分配 (难度: ⭐⭐⭐)

**目标：** 理解 KV Cache Block 的分配和释放

**步骤：**

1. **创建可视化脚本**

```python
# visualize_blocks.py
import matplotlib.pyplot as plt
import numpy as np
from vllm import LLM, SamplingParams

class BlockVisualizer:
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.block_status = np.zeros(num_blocks)  # 0: free, 1: used
        self.history = []

    def update(self, allocated_blocks):
        self.block_status = np.zeros(self.num_blocks)
        for block_id in allocated_blocks:
            self.block_status[block_id] = 1
        self.history.append(self.block_status.copy())

    def plot(self):
        fig, ax = plt.subplots(figsize=(15, 5))
        data = np.array(self.history)
        im = ax.imshow(data.T, aspect='auto', cmap='RdYlGn_r')
        ax.set_xlabel('Schedule Step')
        ax.set_ylabel('Block ID')
        ax.set_title('KV Cache Block Allocation Over Time')
        plt.colorbar(im, label='0=Free, 1=Used')
        plt.savefig('block_allocation.png')
        print("Saved to block_allocation.png")

# 使用方法:
# 在 scheduler.py 中注入 visualizer.update() 调用
```

2. **修改 Scheduler 记录分配**

```python
# 在 scheduler.py 的 schedule() 末尾添加:
all_block_ids = []
for req in self.running:
    block_ids = self.kv_cache_manager.get_block_ids(req.request_id)
    all_block_ids.extend(block_ids[0])  # 假设单 group

# 调用可视化更新
if hasattr(self, 'visualizer'):
    self.visualizer.update(all_block_ids)
```

3. **运行多请求测试**

```python
llm = LLM(model="facebook/opt-125m", gpu_memory_utilization=0.5)

# 并发多个请求
prompts = [
    "Tell me a story about",
    "Explain quantum physics",
    "Write a poem about",
]
outputs = llm.generate(prompts, SamplingParams(max_tokens=50))
```

**思考题：**
- 为什么不同请求的 blocks 不连续？
- 当一个请求完成后，它的 blocks 被复用了吗？

---

### 练习 3: 实现简单的调度策略 (难度: ⭐⭐⭐⭐)

**目标：** 深入理解调度器如何做决策

**任务：** 实现一个 "Shortest Job First" 调度策略

```python
# vllm/v1/core/sched/request_queue.py

class SJFRequestQueue(RequestQueue):
    """
    Shortest Job First: 优先调度剩余 token 最少的请求
    """

    def add_request(self, request: Request):
        # 按 (remaining_tokens, arrival_time) 排序
        remaining = request.max_tokens - len(request.output_token_ids)
        heapq.heappush(self._queue, (remaining, request.arrival_time, request))

    def peek_request(self) -> Request:
        return self._queue[0][2]

    def pop_request(self) -> Request:
        return heapq.heappop(self._queue)[2]
```

**测试：**

```python
# 比较 FCFS 和 SJF 的平均延迟
requests = [
    ("Short prompt", SamplingParams(max_tokens=10)),
    ("Long prompt " * 100, SamplingParams(max_tokens=100)),
    ("Medium prompt " * 10, SamplingParams(max_tokens=50)),
]

# 测量不同策略下的平均完成时间
```

---

### 练习 4: 分析 PagedAttention Kernel (难度: ⭐⭐⭐⭐⭐)

**目标：** 深入 CUDA kernel 实现

**步骤：**

1. **使用 NSight Systems 分析**

```bash
nsys profile -o vllm_profile python test_inference.py
nsys-ui vllm_profile.nsys-rep
```

2. **定位 PagedAttention kernel**

在 NSight 中搜索:
- `paged_attention_v1_kernel`
- `paged_attention_v2_kernel`

3. **分析 kernel 参数**

```python
# 在 vllm/attention/ops/paged_attn.py 的 forward_decode() 前添加:
print(f"Query shape: {query.shape}")
print(f"Block tables shape: {block_tables.shape}")
print(f"Max seq len: {max_seq_len}")
print(f"Using V1: {use_v1}")
```

4. **实验不同配置**

```python
# 测试不同 batch size 和 seq len 对 kernel 选择的影响
for batch_size in [1, 8, 64, 512]:
    for seq_len in [128, 1024, 8192, 16384]:
        # 记录使用 V1 还是 V2
        # 记录执行时间
```

**思考题：**
- V1 和 V2 的选择启发式是否最优？能否改进？
- 不同 block size 对性能有什么影响？

---

### 练习 5: 多模态调试 (难度: ⭐⭐⭐)

**目标：** 理解图像如何变成 "tokens"

**步骤：**

1. **追踪 placeholder 展开**

```python
# 修改 vllm/multimodal/processing.py
class BaseMultiModalProcessor:
    def apply(self, ...):
        # 打印原始 token 数量
        print(f"Original tokens: {len(token_ids)}")

        result = self._apply(...)

        # 打印展开后数量
        print(f"After expansion: {len(result.prompt_token_ids)}")
        print(f"Image tokens added: {len(result.prompt_token_ids) - len(token_ids)}")

        return result
```

2. **可视化 image features**

```python
# test_multimodal.py
from vllm import LLM, SamplingParams
from PIL import Image

llm = LLM(
    model="llava-hf/llava-1.5-7b-hf",
    max_model_len=4096,
)

image = Image.open("test.jpg")
prompt = "<image>\nDescribe this image."

outputs = llm.generate(
    [{
        "prompt": prompt,
        "multi_modal_data": {"image": image}
    }],
    SamplingParams(max_tokens=100)
)
```

3. **检查 encoder cache 命中**

```python
# 相同图片发送两次，检查 encoder cache 是否复用
for i in range(2):
    outputs = llm.generate([{...}])
    # 检查日志中的 cache hit
```

---

### 练习 6: 分布式调试 (难度: ⭐⭐⭐⭐⭐)

**目标：** 理解 TP/PP 下的通信

**前提：** 需要多 GPU 环境

**步骤：**

1. **启动 TP=2 配置**

```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,
)
```

2. **追踪 AllReduce 操作**

```python
# 修改 vllm/distributed/communication_op.py
def tensor_model_parallel_all_reduce(tensor):
    if get_tensor_model_parallel_world_size() == 1:
        return tensor

    # 添加调试
    rank = get_tensor_model_parallel_rank()
    print(f"[TP Rank {rank}] AllReduce tensor shape: {tensor.shape}")

    torch.distributed.all_reduce(tensor, group=get_tensor_model_parallel_group())
    return tensor
```

3. **测量通信开销**

```python
import torch.cuda.nvtx as nvtx

def tensor_model_parallel_all_reduce(tensor):
    with nvtx.annotate("AllReduce", color="red"):
        torch.distributed.all_reduce(...)
```

然后用 NSight 查看 AllReduce 在总时间中的占比。

---

### 推荐学习路径

#### 第一周：熟悉入口
1. 运行官方 examples
2. 完成练习 1
3. 阅读 `scheduler.py` 前 200 行

#### 第二周：深入调度
1. 完成练习 2
2. 阅读 `kv_cache_manager.py` 全部
3. 尝试练习 3

#### 第三周：理解执行
1. 阅读 `gpu_model_runner.py` 的 `execute_model()`
2. 运行 NSight 分析
3. 开始练习 4

#### 第四周：专项深入
1. 如果对多模态感兴趣 → 练习 5
2. 如果对分布式感兴趣 → 练习 6
3. 如果对性能优化感兴趣 → 深入 CUDA kernel

### 调试技巧

1. **日志级别**
```bash
export VLLM_LOGGING_LEVEL=DEBUG
```

2. **打断点**
```python
import ipdb; ipdb.set_trace()
```

3. **GPU Profiling**
```bash
nsys profile -o output python script.py
```

4. **内存分析**
```python
torch.cuda.memory_summary()
```

5. **分布式调试**
```bash
# 只在 rank 0 打印
if torch.distributed.get_rank() == 0:
    print(...)
```

### 贡献代码建议

1. **从小 issue 开始**
   - 文档改进
   - 类型注解补充
   - 小 bug 修复

2. **阅读已有 PR**
   - 学习代码风格
   - 理解 review 流程

3. **参与讨论**
   - GitHub Issues
   - Discord 社区

4. **性能优化方向**
   - 新的调度策略
   - Kernel 优化
   - 内存管理改进

---

## 代码阅读路径总结

### 入门路径（2-3天）
```
scheduler.py → kv_cache_manager.py → gpu_model_runner.py (前500行)
```

### 深入路径（1周）
```
scheduler.py (全部)
  → kv_cache_utils.py (KVCacheBlock, FreeKVCacheBlockQueue)
  → gpu_model_runner.py (execute_model, _prepare_inputs)
  → paged_attn.py (forward_decode, forward_prefix)
```

### 多模态专项（3天）
```
multimodal/registry.py → multimodal/processing.py
  → encoder_cache_manager.py
  → model_executor/models/llava.py 或 qwen2_5_vl.py
```

### 分布式专项（3天）
```
distributed/parallel_state.py → executor/multiproc_executor.py
  → worker/gpu_worker.py
  → model_executor/layers/linear.py (Parallel Linear)
```

---

## 参考资源

1. **官方文档**: https://docs.vllm.ai/
2. **论文**: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
3. **源码**: https://github.com/vllm-project/vllm
4. **Discord**: https://discord.gg/vllm

---

> 本文档持续更新，最后更新时间: 2025-11-20
