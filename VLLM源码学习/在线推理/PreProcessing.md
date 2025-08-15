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












