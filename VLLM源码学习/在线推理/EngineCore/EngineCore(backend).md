# 1. 初始化EngineCore

**调用链**

run_server ->  run_server_worker -> build_async_engine_client -> build_async_engine_client_from_engine_args -> AsyncLLM -> make_async_mp_client -> DPAsyncMPClient -> AsyncMPClient(父类) -> MPClient(父类) -> launch_core_engines -> run_engine_core -> EngineCoreProc -> EngineCore 

**源码**

```python
class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(self,
                 vllm_config: VllmConfig,
                 executor_class: type[Executor],
                 log_stats: bool,
                 executor_fail_callback: Optional[Callable] = None):

        # plugins need to be loaded at the engine/scheduler level too
        from vllm.plugins import load_general_plugins
        # 加载通用插件，扩展引擎功能
        load_general_plugins()

        self.vllm_config = vllm_config
        logger.info("Initializing a V1 LLM engine (v%s) with config: %s",
                    VLLM_VERSION, vllm_config)

        self.log_stats = log_stats

        # Setup Model.
        # 创建模型执行器，负责实际的模型推理
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(
                executor_fail_callback)

        self.available_gpu_memory_for_kv_cache = -1

        # Setup KV Caches and update CacheConfig after profiling.
        # kv cache初始化，分析可用内存，计算 GPU/CPU 能容纳的 KV 缓存块数
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = \
            self._initialize_kv_caches(vllm_config)

        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
        self.collective_rpc("initialize_cache",
                            args=(num_gpu_blocks, num_cpu_blocks))

        # 处理 JSON 等结构化输出格式
        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        if isinstance(vllm_config.scheduler_config.scheduler_cls, str):
            Scheduler = resolve_obj_by_qualname(
                vllm_config.scheduler_config.scheduler_cls)
        else:
            Scheduler = vllm_config.scheduler_config.scheduler_cls

        # This warning can be removed once the V1 Scheduler interface is
        # finalized and we can maintain support for scheduler classes that
        # implement it
        if Scheduler is not V1Scheduler:
            logger.warning(
                "Using configured V1 scheduler class %s. "
                "This scheduler interface is not public and "
                "compatibility may not be maintained.",
                vllm_config.scheduler_config.scheduler_cls)

        if len(kv_cache_config.kv_cache_groups) == 0:
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            logger.info("Disabling chunked prefill for model without KVCache")
            vllm_config.scheduler_config.chunked_prefill_enabled = False
        
        # 创建调度器实例
        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=vllm_config.parallel_config.data_parallel_size
            > 1,
            log_stats=self.log_stats,
        )

       # 支持图像、音频等多模态输入的缓存预处理
        self.mm_input_cache_server = MultiModalInputCacheServer(
            vllm_config.model_config, MULTIMODAL_REGISTRY)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        # 批处理队列（流水线并行）
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: Optional[queue.Queue[tuple[Future[ModelRunnerOutput],
                                                     SchedulerOutput]]] = None
        if self.batch_queue_size > 1:
            logger.info("Batch queue is enabled with size %d",
                        self.batch_queue_size)
            self.batch_queue = queue.Queue(self.batch_queue_size)

        # 支持前缀缓存（prefix caching）
        self.request_block_hasher: Optional[Callable[[Request],
                                                     list[BlockHash]]] = None
        if (self.vllm_config.cache_config.enable_prefix_caching
                or self.scheduler.get_kv_connector() is not None):

            block_size = vllm_config.cache_config.block_size
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo)
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                block_size, caching_hash_fn)
```

# 2. 启动EngineCore后台进程
**调用链**

run_server ->  run_server_worker -> build_async_engine_client -> build_async_engine_client_from_engine_args -> AsyncLLM -> make_async_mp_client -> DPAsyncMPClient -> AsyncMPClient(父类) -> MPClient(父类) -> launch_core_engines -> run_engine_core

**源码**

```python
@staticmethod
def run_engine_core(*args,
                    dp_rank: int = 0,
                    local_dp_rank: int = 0,
                    **kwargs):
    """Launch EngineCore busy loop in background process."""

    # Signal handler used for graceful termination.
    # SystemExit exception is only raised once to allow this and worker
    # processes to terminate without error
    shutdown_requested = False

    # Ensure we can serialize transformer config after spawning
    maybe_register_config_serialize_by_value()
    
    # # 优雅退出
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit()

    # Either SIGTERM or SIGINT will terminate the engine_core
    signal.signal(signal.SIGTERM, signal_handler)  # kill命令
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C

    engine_core: Optional[EngineCoreProc] = None
    try:
        parallel_config: ParallelConfig = kwargs[
            "vllm_config"].parallel_config
        # 是否是数据并行都分别创建EngineCore进程，详见2.2
        if parallel_config.data_parallel_size > 1 or dp_rank > 0:
            set_process_title("DPEngineCore", str(dp_rank))
            decorate_logs()
            # Set data parallel rank for this engine process.
            parallel_config.data_parallel_rank = dp_rank
            parallel_config.data_parallel_rank_local = local_dp_rank
            engine_core = DPEngineCoreProc(*args, **kwargs)
        else:
            set_process_title("EngineCore")
            decorate_logs()
            engine_core = EngineCoreProc(*args, **kwargs)
        
        # 启动循环，处理输入输出请求(***)，详见2.1
        engine_core.run_busy_loop()

    except SystemExit:
        logger.debug("EngineCore exiting.")
        raise
    except Exception as e:
        if engine_core is None:
            logger.exception("EngineCore failed to start.")
        else:
            logger.exception("EngineCore encountered a fatal error.")
            # 通知EngineCoreClient引擎死亡
            engine_core._send_engine_dead()
        raise e
    finally:
        if engine_core is not None:
            engine_core.shutdown()
```
<img width="720" height="508" alt="image" src="https://github.com/user-attachments/assets/904a0545-7af2-41cb-8f67-491d54ddca05" />

## 2.1 run_busy_loop
**源码**

```python
def run_busy_loop(self):
    """Core busy loop of the EngineCore."""

    # Loop until process is sent a SIGINT or SIGTERM
    while True:
        # 1) Poll the input queue until there is work to do.
        self._process_input_queue()
        # 2) Step the engine core and return the outputs.
        self._process_engine_step()
        
        
def _process_input_queue(self):
    """Exits when an engine step needs to be performed."""

    waited = False
    while not self.engines_running and not self.scheduler.has_requests():
        if logger.isEnabledFor(DEBUG) and self.input_queue.empty():
            # 等待阶段。如果EngineCore没有启动成功，或者没有请求，并且请求输入队列为空
            logger.debug("EngineCore waiting for work.")
            waited = True
        # 队列不为空则阻塞等待请求，处理请求
        req = self.input_queue.get()
        self._handle_client_request(*req)

    if waited:
        logger.debug("EngineCore loop active.")

    # Handle any more client requests.
    # 非阻塞处理阶段（清空队列），处理队列中剩余的所有请求
    while not self.input_queue.empty():
        req = self.input_queue.get_nowait()
        self._handle_client_request(*req)
        
        
def _process_engine_step(self) -> bool:
    """Called only when there are unfinished local requests."""

    # Step the engine core.
    # 返回输出结果和执行状态
    outputs, model_executed = self.step_fn()
    # Put EngineCoreOutputs into the output queue.
    # 将输出放入输出队列
    for output in (outputs.items() if outputs else ()):
        self.output_queue.put_nowait(output)

    return model_executed
```

## 2.2 EngineCoreProc
**源码**

```python
class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b'ENGINE_CORE_DEAD'

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: Optional[str] = None,
        engine_index: int = 0,
    ):
        # 输入输出队列初始化
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[Union[tuple[int, EngineCoreOutputs],
                                              bytes]]()
        # 当模型执行器失败时，回调向输入队列发送故障通知
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b''))
        
        # 为每个engineCore进程创建唯一标识，用于 ZeroMQ 的套接字标识
        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False
        
        # 前后端（和EngineCoreClient）握手，建立连接
        with self._perform_handshakes(handshake_address, identity,
                                      local_client, vllm_config,
                                      client_handshake_address) as addresses:
            self.client_count = len(addresses.outputs)

            # Set up data parallel environment.
            # dp环境处理
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address)
            logger.debug("Has DP Coordinator: %s, stats publish address: %s",
                         self.has_coordinator,
                         self.frontend_stats_publish_address)
            # Only publish request queue stats to coordinator for "internal"
            # and "hybrid" LB modes .
            # 决定是否向协调器发布负载统计，如果是外部LB模式则不发送
            self.publish_dp_lb_stats = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb)
            
            # dp环境，一些参数初始化，如dp_rank
            self._init_data_parallel(vllm_config)
            
            # 调用父类EngineCore初始化其组件
            super().__init__(vllm_config, executor_class, log_stats,
                             executor_fail_callback)

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            # 输入线程创建，从前端或协调器(dp)接收请求并放入输入队列
            ready_event = threading.Event()
            input_thread = threading.Thread(target=self.process_input_sockets,
                                            args=(addresses.inputs,
                                                  addresses.coordinator_input,
                                                  identity, ready_event),
                                            daemon=True)
            input_thread.start()
            
            # 输出线程创建，从输出队列取结果并发送到前端或协调器(dp)
            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(addresses.outputs, addresses.coordinator_output,
                      self.engine_index),
                daemon=True)
            self.output_thread.start()

            # Don't complete handshake until DP coordinator ready message is
            # received.
            # 等待数据并行协调器发送就绪信号
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError(
                        "Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")
        
        # 选择普通模式还是流水线并行模式批处理队列，详见2.2.1，2.2.2
        self.step_fn = (self.step if self.batch_queue is None else
                        self.step_with_batch_queue)
```

✅ 进程封装器: 让 EngineCore 能在独立进程中运行

✅ 通信管理器: 处理所有 ZeroMQ 网络通信

✅ 线程协调器: 管理IO线程和主推理线程的协作

✅ 资源隔离层: 提供故障隔离和资源管理

+ ZMQ负责EngineCore和EngineCoreClient或协调器coordinator的通信: process_input_sockets, process_output_sockets
+ IO线程管理EngineCore和EngineCoreClient或协调器coordinator的通信：input_thread，output_thread
+ EngineCore主线程专注调度推理

### 2.2.1 step
**源码**
```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    """Schedule, execute, and make output.

    Returns tuple of outputs and a flag indicating whether the model
    was executed.
    """

    # Check for any requests remaining in the scheduler - unfinished,
    # or finished and not yet removed from the batch.
    if not self.scheduler.has_requests():
        return {}, False
    scheduler_output = self.scheduler.schedule()
    model_output = self.execute_model_with_error_logging(
        self.model_executor.execute_model,  # type: ignore
        scheduler_output)
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output)  # type: ignore

    return (engine_core_outputs,
            scheduler_output.total_num_scheduled_tokens > 0)
```

Scheduler完成调度 -> 等待推理完成 -> 更新Scheduler状态

### 2.2.2 step_with_batch_queue(PP模式)
**源码**
```python
def step_with_batch_queue(
        self) -> tuple[Optional[dict[int, EngineCoreOutputs]], bool]:
    """Schedule and execute batches with the batch queue.
    Note that if nothing to output in this step, None is returned.

    The execution flow is as follows:
    1. Try to schedule a new batch if the batch queue is not full.
    If a new batch is scheduled, directly return an empty engine core
    output. In other words, fulfilling the batch queue has a higher priority
    than getting model outputs.
    2. If there is no new scheduled batch, meaning that the batch queue
    is full or no other requests can be scheduled, we block until the first
    batch in the job queue is finished.
    3. Update the scheduler from the output.
    """
    assert self.batch_queue is not None

    engine_core_outputs = None
    scheduler_output = None
    # Try to schedule a new batch if the batch queue is not full, but
    # the scheduler may return an empty batch if all requests are scheduled.
    # Note that this is not blocking.
    # 批处理队列未满时尝试调度
    if not self.batch_queue.full():
        scheduler_output = self.scheduler.schedule()
        if scheduler_output.total_num_scheduled_tokens > 0:
            # 执行模型推理并将Future放入队列
            future = self.model_executor.execute_model(scheduler_output)
            self.batch_queue.put_nowait(
                (future, scheduler_output))  # type: ignore
    
    # 检查是否成功调度
    scheduled_batch = (scheduler_output is not None
                       and scheduler_output.total_num_scheduled_tokens > 0)

    # If no more requests can be scheduled and the job queue is not empty,
    # block until the first batch in the job queue is finished.
    # TODO(comaniac): Ideally we should peek the first batch in the
    # job queue to check if it's finished before scheduling a new batch,
    # but peeking the first element in a queue is not thread-safe,
    # so we need more work.
    # 没有新批次且队列不为空，取出批次并等待模型执行完成。用输出更新调度器状态
    if not scheduled_batch and not self.batch_queue.empty():
        future, scheduler_output = self.batch_queue.get_nowait()

        # Blocking until the first result is available.
        model_output = self.execute_model_with_error_logging(
            lambda _: future.result(), scheduler_output)

        self.batch_queue.task_done()
        engine_core_outputs = (self.scheduler.update_from_output(
            scheduler_output, model_output))

    return engine_core_outputs, scheduled_batch
```

1. 尝试调度新批次到Scheduler（如果队列未满）
2. 如果没有新批次可调度到Scheduler，等待队列中的批次调度推理完成  
3. 从完成批次更新调度器状态


