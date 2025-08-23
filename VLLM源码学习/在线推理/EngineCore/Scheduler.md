# 1. 初始化Scheduler
**调用链**

run_server ->  run_server_worker -> build_async_engine_client -> build_async_engine_client_from_engine_args -> AsyncLLM -> make_async_mp_client -> DPAsyncMPClient -> AsyncMPClient(父类) -> MPClient(父类) -> launch_core_engines -> run_engine_core -> EngineCoreProc -> EngineCore -> Scheduler

**源码**
```python
# Setup scheduler.
# 未指定Scheduler则默认v1 schedluer
if isinstance(vllm_config.scheduler_config.scheduler_cls, str):
    Scheduler = resolve_obj_by_qualname(
        vllm_config.scheduler_config.scheduler_cls)
else:
    Scheduler = vllm_config.scheduler_config.scheduler_cls
    
    
self.scheduler: SchedulerInterface = Scheduler(
    vllm_config=vllm_config,
    kv_cache_config=kv_cache_config,
    structured_output_manager=self.structured_output_manager,
    include_finished_set=vllm_config.parallel_config.data_parallel_size
    > 1,
    log_stats=self.log_stats,
)
```



# 2. 调度策略
**调用链**

EngineCoreProc -> step -> schedule

<details>
  <summary>源码</summary>


</details>





