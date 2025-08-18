vLLM 本身不实现自己的 tokenizer，而是直接调用底层模型（如 Hugging Face 的 Transformers 模型）提供的 tokenizer。

# 1. 初始化tokenizer

**调用链**

命令行启动创建engineClient时候执行

run_server ->  run_server_worker -> build_async_engine_client -> build_async_engine_client_from_engine_args -> AsyncLLM -> init_tokenizer_from_configs

**源码**

```python
def init_tokenizer_from_configs(
    model_config: ModelConfig,
    scheduler_config: SchedulerConfig,
    lora_config: Optional[LoRAConfig]):

    # 根据 runner_type 决定文本截断方向
    runner_type = model_config.runner_type
    if runner_type == "generate" or runner_type == "draft":
        truncation_side = "left"  # 自回归或草稿模型生成任务从左截断（保留右侧最新内容）
    elif runner_type == "pooling":
        truncation_side = "right"  # 池化任务从右截断（如分类任务保留左侧主体）
    else:
        assert_never(runner_type)  # 确保 runner_type 合法

    # 返回 TokenizerGroup 封装对象
    return TokenizerGroup(
        tokenizer_id=model_config.tokenizer,          # Tokenizer 名称或路径
        enable_lora=bool(lora_config),               # 是否启用 LoRA
        max_num_seqs=scheduler_config.max_num_seqs,  # 最大序列数（批处理容量）
        max_loras=lora_config.max_loras if lora_config else 0,  # 最大 LoRA 适配器数
        max_input_length=None,                       # 输入长度限制（None 表示无限制）
        tokenizer_mode=model_config.tokenizer_mode,  # Tokenizer 模式（如 "auto"）
        trust_remote_code=model_config.trust_remote_code,  # 是否信任远程代码（如自定义 Tokenizer）
        revision=model_config.tokenizer_revision,    # Tokenizer 版本（如 Git commit hash）
        truncation_side=truncation_side,             # 截断方向（left/right）
    )
```

# 2. 预处理阶段对tokenizer操作

**调用链**

/v1/chat/completions -> create_chat_completion -> self.engine_client.generate(不使用beam search) -> generate(async_llm.py) -> add_request -> process_inputs -> get_eos_token_id

**源码**

```python
eos_token_id = self.input_preprocessor.get_eos_token_id(lora_request)

sampling_params.update_from_generation_config(
    self.generation_config_fields, eos_token_id)
```

+ 从 input_preprocessor 中获取当前模型的 默认 EOS Token ID。如果启用了 LoRA，会根据 lora_request 选择对应适配器的 EOS Token。
+ 处理eos_token_id，update_from_generation_config
+ 将eos_token_id作为EngineCoreRequest参数返回

## 2.1 update_from_generation_config
**源码**

```python
def update_from_generation_config(
        self,
        generation_config: dict[str, Any],
        model_eos_token_id: Optional[int] = None) -> None:
    """Update if there are non-default values from generation_config"""

    if model_eos_token_id is not None:
        # Add the eos token id into the sampling_params to support
        # min_tokens processing.
        self._all_stop_token_ids.add(model_eos_token_id)

    # Update eos_token_id for generation
    if (eos_ids := generation_config.get("eos_token_id")) is not None:
        # it can be either int or list of int
        eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
        if model_eos_token_id is not None:
            # We don't need to include the primary eos_token_id in
            # stop_token_ids since it's handled separately for stopping
            # purposes.
            eos_ids.discard(model_eos_token_id)
        if eos_ids:
            self._all_stop_token_ids.update(eos_ids)
            if not self.ignore_eos:
                eos_ids.update(self.stop_token_ids)
                self.stop_token_ids = list(eos_ids)
```

+ 将模型/LoRA 的默认 EOS Token 加入 _all_stop_token_ids（内部集合，用于最小 Token 长度检查）。
+ 解析用户定义的 eos_token_id（可能是单个值或列表）。
+ 如果用户配置的 EOS Token 包含模型默认值，移除默认值（避免重复停止）。
+ 将用户定义的 EOS Token 和原有 stop_token_ids 合并，最终赋值给 self.stop_token_ids。

**EOS Token ID处理原则：**

默认由 Tokenizer 的 eos_token_id 属性提供（如 tokenizer.eos_token_id）。

如果用户通过 generation_config 覆盖，则优先使用自定义值。

**LoRA的影响：**

当 LoRA 适配器修改了 Tokenizer（例如扩展词汇表），get_eos_token_id() 会返回适配器对应的 EOS Token。





