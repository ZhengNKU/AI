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









