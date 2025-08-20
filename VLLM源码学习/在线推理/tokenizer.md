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

## 2.1 get_tokenizer
**调用链**
get_eos_token_id -> get_lora_tokenizer -> get_lora_tokenizer -> get_tokenizer
**源码**
```python
def get_tokenizer(
    tokenizer_name: Union[str, Path],
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    download_dir: Optional[str] = None,
    **kwargs,
) -> AnyTokenizer:
    """Gets a tokenizer for the given model name via HuggingFace or ModelScope.
    """
    if envs.VLLM_USE_MODELSCOPE:
        # download model from ModelScope hub,
        # lazy import so that modelscope is not required for normal use.
        # pylint: disable=C.
        from modelscope.hub.snapshot_download import snapshot_download

        # avoid circuit import
        from vllm.model_executor.model_loader.weight_utils import get_lock

        # Only set the tokenizer here, model will be downloaded on the workers.
        if not os.path.exists(tokenizer_name):
            # Use file lock to prevent multiple processes from
            # downloading the same file at the same time.
            with get_lock(tokenizer_name, download_dir):
                tokenizer_path = snapshot_download(
                    model_id=tokenizer_name,
                    cache_dir=download_dir,
                    revision=revision,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    # Ignore weights - we only need the tokenizer.
                    ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"])
                tokenizer_name = tokenizer_path

    # 强制使用慢速但兼容性好的分词器
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if "truncation_side" not in kwargs:
        kwargs["truncation_side"] = "left"

    # Separate model folder from file path for GGUF models
    # GGUF文件格式需要额外处理，分离文件和路径
    is_gguf = check_gguf_file(tokenizer_name)
    if is_gguf:
        kwargs["gguf_file"] = Path(tokenizer_name).name
        tokenizer_name = Path(tokenizer_name).parent

    # if tokenizer is from official mistral org
    is_from_mistral_org = str(tokenizer_name).split("/")[0] == "mistralai"
    # 选择tokenizer模式
    if is_from_mistral_org and tokenizer_mode != "mistral":
        warnings.warn(
            'It is strongly recommended to run mistral models with '
            '`--tokenizer-mode "mistral"` to ensure correct '
            'encoding and decoding.',
            FutureWarning,
            stacklevel=2)

    tokenizer: AnyTokenizer
    # tokenizer模式为mistral，需要特殊的分词处理方式
    if tokenizer_mode == "mistral":
        tokenizer = MistralTokenizer.from_pretrained(str(tokenizer_name),
                                                     revision=revision)
    # tokenizer模式为custom，从注册表中获取tokenizer
    elif tokenizer_mode == "custom":
        from vllm.transformers_utils.tokenizer_base import TokenizerRegistry
        tokenizer = TokenizerRegistry.get_tokenizer(str(tokenizer_name),
                                                    *args,
                                                    revision=revision,
                                                    download_dir=download_dir,
                                                    **kwargs)
    else:
        # 默认模式，执行自定义代码，从huggingface获取tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )
        except ValueError as e:
            # If the error pertains to the tokenizer class not existing or not
            # currently being imported,
            # suggest using the --trust-remote-code flag.
            if not trust_remote_code and (
                    "does not exist or is not currently imported." in str(e)
                    or "requires you to execute the tokenizer file" in str(e)):
                err_msg = ("Failed to load the tokenizer. If the tokenizer "
                           "is a custom tokenizer not yet available in the "
                           "HuggingFace transformers library, consider "
                           "setting `trust_remote_code=True` in LLM or using "
                           "the `--trust-remote-code` flag in the CLI.")
                raise RuntimeError(err_msg) from e
            else:
                raise e

        # The special_tokens in tokenizer should also be
        # controlled by do_lower_case in encoder_config
        # 句子变换器通常将所有输入转换为小写，但特殊token可能保持原大小写。因此需要确保特殊token与文本处理方式一致
        encoder_config = get_sentence_transformer_tokenizer_config(
            tokenizer_name, revision)
        if isinstance(encoder_config, dict) and encoder_config.get(
                "do_lower_case", False):
            special_tokens_map = {
                k: v.lower()
                for k, v in tokenizer.special_tokens_map.items()
            }
            tokenizer.add_special_tokens(special_tokens_map)

        # NOTE: We can remove this after https://github.com/zai-org/ChatGLM3/issues/1324
        # ChatGLM 分词器默认 padding_side="right"，但 VLLM 需要 padding_side="left"。因此patch_padding_side 函数会将padding方向改为 left
        if type(tokenizer).__name__ in ("ChatGLMTokenizer",
                                        "ChatGLM4Tokenizer"):
            assert isinstance(tokenizer, PreTrainedTokenizer)
            patch_padding_side(tokenizer)
        
        # 在高速推理场景下，慢速分词器可能降低整体吞吐量 30-50%。不是快速tokenizer则给出警告
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            logger.warning(
                "Using a slow tokenizer. This might cause a significant "
                "slowdown. Consider using a fast tokenizer instead.")
        # 缓存分词器实例，避免重复加载相同配置的分词器
        tokenizer = get_cached_tokenizer(tokenizer)

    return tokenizer

```
<img width="701" height="928" alt="image" src="https://github.com/user-attachments/assets/6c97621e-852a-4dbb-8495-318f93aafa62" />

+ 多源加载：从 HuggingFace、ModelScope 或本地文件加载分词器
+ 模式适配：支持不同特化的分词模式（slow/fast/mistral/custom）
+ 配置优化：自动设置截断方向、缓存策略等关键参数
+ 错误处理：提供清晰的错误提示和修复建议


## 2.2 update_from_generation_config
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





