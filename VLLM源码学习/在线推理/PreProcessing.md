# 1. 初始化Processor实例
**调用链**

run_server ->  run_server_worker -> build_async_engine_client -> build_async_engine_client_from_engine_args -> AsyncLLM -> Processor

**源码**
```python
self.processor = Processor(
    vllm_config=vllm_config,
    tokenizer=self.tokenizer,
    mm_registry=mm_registry,
)
```

## 1.1 传参
| 参数             | 类型                  | 作用                                   |
|------------------|-----------------------|----------------------------------------|
| `vllm_config`    | `VllmConfig`          | 全局配置（模型/缓存/解码等）            |
| `tokenizer`      | `TokenizerGroup`      | 支持LoRA的多分词器管理                 |
| `mm_registry`    | `MultiModalRegistry`  | 多模态处理器注册中心（如图像）         |

vllm_config创建时机：

```python
vllm_config = engine_args.create_engine_config(usage_context=usage_context)
```

## 1.2 单例初始化

部分参数初始化不直接从全局容器vllm_config获取，需要动态处理

### 1.2.1 动态获取模型生成配置
从模型配置中提取生成参数（如温度、top_p等），用于后续的推理请求默认值，返回字典如 {"temperature": 0.7, "top_p": 0.9}
```python
self.generation_config_fields = self.model_config.try_get_generation_config()

def try_get_generation_config(self) -> dict[str, Any]:
    # 如果配置来源是 "auto"/"vllm"，则从hf获取模型配置
    if self.generation_config in ("auto", "vllm"):
        config = try_get_generation_config(
            self.hf_config_path or self.model,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
        )
    else:
    # 显示指定模型配置
        config = try_get_generation_config(
            self.generation_config,
            trust_remote_code=self.trust_remote_code,
        )

    if config is None:
        return {}

    return config.to_diff_dict()
```

**附：模型生成配置参数介绍**
| 参数名                  | 类型     | 默认值  | 作用                                                                 | 推荐范围          |
|-------------------------|----------|---------|----------------------------------------------------------------------|-------------------|
| `temperature`           | `float`  | `1.0`   | 控制随机性：值越高输出越多样，越低越确定                              | `0.1~1.5`         |
| `top_p` (nucleus)       | `float`  | `1.0`   | 从概率质量前p%的token中采样，避免低质量输出                           | `0.5~0.95`        |
| `top_k`                 | `int`    | `∞`     | 仅从概率最高的k个token中采样                                         | `10~100`          |
| `frequency_penalty`     | `float`  | `0.0`   | 惩罚重复token（正值降低重复，负值增加重复）                           | `-2.0~2.0`        |
| `presence_penalty`      | `float`  | `0.0`   | 惩罚已出现的token（与`frequency_penalty`的区别：不计重复次数）        | `-2.0~2.0`        |
| `repetition_penalty`    | `float`  | `1.0`   | 直接乘性惩罚重复token（`>1`降低重复，`<1`增加重复）                   | `1.0~1.5`         |
| `max_tokens`            | `int`    | `16`    | 生成的最大token数（包括输入）                                         | -                 |
| `min_tokens`            | `int`    | `0`     | 生成的最小token数（强制模型至少输出多少token）                         | -                 |
| `length_penalty`        | `float`  | `1.0`   | 对生成长度的调节（`>1`鼓励更长输出，`<1`鼓励更短）                     | -                 |

多模态参数

| 参数名           | 类型     | 作用                                      |
|------------------|----------|------------------------------------------|
| `image_quality`  | `int`    | 图像输入的质量等级（1-100，影响图像编码精度） |
| `audio_temp`     | `float`  | 音频生成的temperature（独立于文本temperature） |

### 1.2.2 InputPreprocessor
InputPreprocessor 是 VLLM 中负责统一处理输入数据的核心组件，主要完成：

+ 文本标准化：分词、截断、特殊标记处理
+ 多模态适配：图像/音频等非文本输入的编码转换
+ LoRA 适配：动态切换分词器应对不同的 LoRA 适配器
+ 输入验证：确保输入符合模型要求

# 2. 预处理
**调用链**

OpenAI接口访问时执行。使用beam search与否前处理方法不同。

| 处理阶段       | Beam Search 模式                          | 常规采样模式                     |
|----------------|------------------------------------------|----------------------------------|
| 参数验证       | 检查 n (beam width) 和 best_of            | 检查 temperature/top_p           |
| 输入准备       | 需要初始化多个候选序列                   | 单序列处理                       |
| 调度策略       | 维护多个候选束（Beam）并并行解码         | 单序列随机采样                   |
| 停止条件       | 所有候选序列达到停止条件才终止           | 单个序列满足条件即可终止         |

/v1/chat/completions -> create_chat_completion -> self.engine_client.generate(不使用beam search) -> generate(async_llm.py) -> add_request -> process_inputs

**目的**

input转换为EngineCoreRequest

**源码**
```python
prompt_str, request = self.processor.process_inputs(
    request_id, prompt, params, arrival_time, lora_request,
    tokenization_kwargs, trace_headers, prompt_adapter_request,
    priority, data_parallel_rank)
```

## 2.1 传参
| 参数名                  | 类型                      | 必选  | 作用说明                                                                 |
|-------------------------|---------------------------|-------|-------------------------------------------------------------------------|
| `request_id`            | `str`                     | ✓     | 请求唯一标识符，用于跟踪和日志记录                                      |
| `prompt`                | `PromptType`              | ✓     | 输入内容（支持文本字符串、多模态字典等）                                |
| `params`                | `SamplingParams` / `PoolingParams` | ✓ | 生成策略参数（如温度采样/束搜索配置）                                   |
| `arrival_time`          | `float`                   | ✗     | 请求到达时间戳（默认自动生成）                                          |
| `lora_request`          | `LoRARequest`             | ✗     | LoRA适配器配置（如适配器路径、权重缩放）                                |
| `tokenization_kwargs`   | `Dict[str, Any]`          | ✗     | 分词器额外参数（例如 `add_special_tokens=False`）                       |
| `trace_headers`         | `Mapping[str, str]`       | ✗     | 分布式追踪的上下文头（当前版本不支持）                                  |
| `prompt_adapter_request`| `PromptAdapterRequest`    | ✗     | 动态提示模板配置（当前版本不支持）                                      |
| `priority`              | `int`                     | ✗     | 请求优先级（0-100，数值越高优先级越高）                                |
| `data_parallel_rank`    | `int`                     | ✗     | 数据并行训练中的设备编号（用于分布式推理）                              |

## 2.2 核心处理流程
1. 检查lora适配器的有效性：适配器路径需存在，且符合模型架构
2. 验证采样参数
3. 将原始输入（文本/多模态）转换为模型可处理的标准化格式（***）。详见2.2.1
4. 检查平台兼容性
5. 采样参数处理，包括自动计算最大生成长度（总长度限制减去输入长度）、合并模型默认生成配置（如 eos_token_id）
6. 多模态数据排序与缓存，详见2.2.2
7. 返回EngineCoreRequest

### 2.2.1 input_preprocessor.preprocess
**源码**
```python
processed_inputs: ProcessorInputs = self.input_preprocessor.preprocess(
    prompt,
    tokenization_kwargs=tokenization_kwargs,
    lora_request=lora_request,
    prompt_adapter_request=prompt_adapter_request,
    return_mm_hashes=self.use_hash,
)

def preprocess(
    self,
    prompt: PromptType,
    tokenization_kwargs: Optional[dict[str, Any]] = None,
    lora_request: Optional[LoRARequest] = None,
    prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    return_mm_hashes: bool = False,
) -> ProcessorInputs:
    """Preprocess the input prompt."""
    # 模型类型检查，如果是Encoder-Decoder 模型（如 T5）：需要分别处理 encoder 和 decoder 输入，并且当前版本暂不支持多模态哈希返回
    if self.model_config.is_encoder_decoder:
        assert not return_mm_hashes, (
            "Multimodal hashes for encoder-decoder models should not be ",
            "returned until they are supported on vLLM V1.")
        # Encoder-decoder model requires special mapping of
        # input prompts to encoder & decoder
        return self._process_encoder_decoder_prompt(
            prompt, tokenization_kwargs)

    if is_explicit_encoder_decoder_prompt(prompt):
        raise ValueError("Cannot pass encoder-decoder prompt "
                         "to decoder-only models")

    # Decoder-only operation
    # Decoder-only 模型处理，如自回归模型
    return self._process_decoder_only_prompt(
        prompt,
        tokenization_kwargs=tokenization_kwargs,
        lora_request=lora_request,
        prompt_adapter_request=prompt_adapter_request,
        return_mm_hashes=return_mm_hashes,
    )
```

这段代码是 VLLM 输入预处理的核心逻辑，负责将原始输入（文本/多模态）转换为模型可处理的标准化格式。主要完成：

+ 输入路由：根据模型类型（Encoder-Decoder 或 Decoder-only）选择处理路径

Encoder-Decoder输出示例：
```json
{
    "encoder": {
        "type": "text",
        "prompt_token_ids": [1, 34, 56, 2],  # 编码器输入token IDs
        "multi_modal_data": None
    },
    "decoder": {
        "type": "text",
        "prompt_token_ids": [0]  # 解码器起始标记
    }
}
```

**Decoder-only模型处理:**

<img width="310" height="446" alt="image" src="https://github.com/user-attachments/assets/4c743f44-850b-4779-b6cf-c03d9c13edfc" />

### 2.2.2 input_preprocessor.preprocess后多模态的处理
**参数说明**
| 变量名            | 类型               | 作用                                                                 |
|--------------------|--------------------|---------------------------------------------------------------------|
| `decoder_inputs`   | `Dict`             | 预处理后的输入，包含 `type`/`mm_kwargs`/`mm_placeholders` 等字段     |
| `MultiModalKwargs` | `TypedDict`        | 存储多模态特征和元数据的容器类                                      |
| `PlaceholderRange` | `NamedTuple`       | 描述多模态元素在序列中的位置 (`start`, `end`)                       |

**示例**
```json
decoder_inputs = {
    "type": "multimodal",  # 输入类型标识
    "mm_kwargs": {         # 各模态的特征数据
        "image": [img1_feat, img2_feat], 
        "audio": [audio_feat]
    },
    "mm_placeholders": {   # 各模态的位置信息
        "image": [PlaceholderRange(5,6), PlaceholderRange(10,11)],
        "audio": [PlaceholderRange(8,9)]
    },
    "mm_hashes": {         # 各模态的哈希值（如果启用缓存）
        "image": ["hash1", "hash2"],
        "audio": ["hash3"]
    }
}
```

**源码**
```python
# Multimodal related.
sorted_mm_inputs: Optional[Sequence[Optional[MultiModalKwargs]]] = None
sorted_mm_positions: Optional[list[PlaceholderRange]] = None
sorted_mm_hashes: Optional[list[str]] = None
if decoder_inputs["type"] == "multimodal":
    decoder_mm_inputs = decoder_inputs["mm_kwargs"]

    # Merge and flatten multimodal placeholders, hashes and inputs
    # from dictionaries to lists, and sort them by each item's position
    # in the input sequence.
    (
        sorted_item_modalities,
        sorted_mm_positions,
        sorted_mm_hashes,
    ) = merge_and_sort_multimodal_metadata(
        decoder_inputs["mm_placeholders"],
        decoder_inputs["mm_hashes"] if self.use_hash else None,
    )

    # The output of merged multi-modal processor (`decoder_mm_inputs`)
    # is a single MultiModalKwargs for all items from all modalities.
    # This code flattens kwargs for individual items in a list and
    # sorts them by each item's position in the input sequence if there
    # are multiple modalities.
    unique_modalities = set(sorted_item_modalities)
    if len(unique_modalities) > 1:
        orig_sorted_mm_inputs = []
        used_indices = {modality: 0 for modality in unique_modalities}

        for modality in sorted_item_modalities:
            items = decoder_mm_inputs.get_items(modality)
            item = items[used_indices[modality]]

            orig_sorted_mm_inputs.append(
                MultiModalKwargs.from_items([item]))
            used_indices[modality] += 1
    else:
        orig_sorted_mm_inputs = [
            MultiModalKwargs.from_items([item]) for item in
            decoder_mm_inputs.get_items(sorted_item_modalities[0])
        ]

    if sorted_mm_hashes is not None:
        sorted_mm_inputs = self.mm_input_cache_client.get_and_update_p0(
            orig_sorted_mm_inputs, sorted_mm_hashes)
    else:
        sorted_mm_inputs = orig_sorted_mm_inputs
```

+ 元数据合并与排序，将不同模态的占位符和哈希值扁平化为列表、按照在输入序列中的出现顺序排序
+ 特征数据重组，确保特征数据与位置信息严格对应。包括多模态交叉场景（如图像和音频交替）和单一模态场景
+ 缓存处理，使用哈希值作为键查询缓存，若命中则直接返回缓存特征，否则存储新特征














