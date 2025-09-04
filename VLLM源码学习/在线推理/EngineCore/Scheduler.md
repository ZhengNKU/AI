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
```python
class Scheduler(SchedulerInterface):

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        # 多引擎场景: 用于跟踪请求的完整生命周期
        # 单引擎场景: 可以省略以节省内存
        self.finished_req_ids_dict: Optional[dict[int, set[str]]] = (
            defaultdict(set) if include_finished_set else None)

        # Scheduling constraints.
        # 控制单批次的大小和复杂度
        # 防止内存溢出和性能下降
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events)

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        # 分布式推理: 处理多个worker间的KV缓存传输。角色区分: Scheduler 负责协调，Worker 负责具体操作
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            assert len(self.kv_cache_config.kv_cache_groups) == 1, (
                "Multiple KV cache groups are not currently supported "
                "with KV connectors")
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_rank,
        )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        
        # 全局请求表: 按request_id索引所有请求
        # 等待队列: 基于策略（优先级/FCFS）的排队。优先级调度: 按优先级处理请求。先到先服务: 按到达顺序处理
        # 运行列表: 当前正在处理的请求
        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        if self.scheduler_config.policy == "priority":
            self.policy = SchedulingPolicy.PRIORITY
        elif self.scheduler_config.policy == "fcfs":
            self.policy = SchedulingPolicy.FCFS
        else:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}")
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        # 跟踪需要清理的请求，确保资源正确释放
        # 本轮完成的请求
        self.finished_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        # 完成KV接收的请求
        self.finished_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        # 视觉编码器: 处理图像等多模态输入
        # 缓存管理: 优化编码器计算和存储
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed). Currently, we assume that the encoder also
        # has the Transformer architecture (e.g., ViT).
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config
        
        # Eagle 推测解码
        # 推测执行: 提前执行可能的下一个token
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
```
✅ 请求管理: 完整的生命周期跟踪

✅ 资源约束: 基于实际限制的调度决策

✅ 多策略支持: 可配置的调度算法

✅ 高级功能: 多模态、推测解码支持

✅ 分布式就绪: KV传输和协调能力

✅ 性能优化: 缓存管理和流水线支持


# 2. 调度策略
**调用链**

EngineCoreProc -> step -> schedule

<details>
  <summary>源码</summary>
    
  ```python
      def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to the running request index.
        # This will helps us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}

        req_to_new_block_ids: dict[str, tuple[list[int], ...]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = (request.num_tokens_with_spec +
                              request.num_output_placeholders -
                              request.num_computed_tokens)
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_budget = encoder_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_budget)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                    else:
                        preempted_req = self.running.pop()

                    self.kv_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            if request.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[request.request_id] = req_index
            req_to_new_block_ids[request.request_id] = (
                new_blocks.get_block_ids())
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id)
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (self.lora_config and request.lora_request and
                    (len(scheduled_loras) == self.lora_config.max_loras and
                     request.lora_request.lora_int_id not in scheduled_loras)):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = \
                        self.kv_cache_manager.get_computed_blocks(
                            request)

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens))

                    # Total computed tokens (local + external).
                    num_computed_tokens = (num_new_local_computed_tokens +
                                           num_external_computed_tokens)
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list())
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                new_encoder_budget = encoder_budget

                # KVTransfer: loading remote KV, do not allocate for new work.
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                # Number of tokens to be scheduled.
                else:
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    if (0 < self.scheduler_config.long_prefill_token_threshold
                            < num_new_tokens):
                        num_new_tokens = (
                            self.scheduler_config.long_prefill_token_threshold)

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if not self.scheduler_config.chunked_prefill_enabled and \
                        num_new_tokens > token_budget:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (encoder_inputs_to_schedule, num_new_tokens,
                         new_encoder_budget
                         ) = self._try_schedule_encoder_inputs(
                             request, num_computed_tokens, num_new_tokens,
                             encoder_budget)
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (0 if request.num_computed_tokens
                                              == 0 else
                                              self.num_lookahead_tokens)

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                if request.use_structured_output:
                    structured_output_request_ids[request.request_id] = (
                        req_index)
                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_block_ids[request.request_id] = (
                    self.kv_cache_manager.get_block_ids(request.request_id))
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        grammar_bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduled_spec_decode_tokens,
        )
        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_block_ids,
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        events = self.kv_cache_manager.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        self._update_after_schedule(scheduler_output)
        return scheduler_output
    ```
</details>

## 2-1 数据结构初始化

（1）调度结果收集

scheduled_new_reqs：用于装从waiting队列中调度的新鲜请求。新鲜指那些不是从抢占状态中恢复的请求

scheduled_resumed_reqs：用于装从waiting队列中调度的抢占请求，也即从抢占状态恢复的请求

scheduled_running_reqs：用于装从running队列中调度的请求

preempted_reqs：用于装本次调度中刚被设置为抢占状态的请求

最终本次调度的SchedulerOutput将由前3者共同组成

（2）输出结构

structured_output_request_ids: dict[str, int] = {}  # 索引映射: request_id -> 运行请求索引

（3）资源预算管理

req_to_new_block_ids: dict[str, tuple[list[int], ...]] = {}  # 块分配

num_scheduled_tokens: dict[str, int] = {}                   # token分配

token_budget = self.max_num_scheduled_tokens    # token预算：每次调度步骤中最多允许计算的token数量，它用来决定每次调度中最多允许“计算（计算kv值并存成cache）”多少个token。这个token_budget可以由用户通过scheduler_config.max_num_batched_tokens进行配置。

scheduled_encoder_inputs: dict[str, list[int]] = {}         # 编码器输入

encoder_budget = self.max_num_encoder_input_tokens          # 编码器预算

scheduled_spec_decode_tokens: dict[str, list[int]] = {}     # 推测解码token

多维度资源: 管理KV缓存块、token数、编码器资源等

预算机制: 每个资源类型有独立的预算限制

（4）调度时间管理

scheduled_timestamp = time.monotonic()

## 2-2 调度RUNNING requests

1. 按顺序处理每个running的请求并且确保还有剩余的token预算。

2. 计算本次请求要处理的token数，它即表示当前这个请求有多少token还没做计算，预留输出位置，不包含已计算的token。同时，限制单个请求一次获取太多token，不能超过token_budget和模型最大长度。
request.num_tokens_with_spec +request.num_output_placeholder = len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids)

prompt_token_ids：表示这个请求的prompt

output_token_ids：表示这个请求当前已经生成的response。举例来说，刚做完prefill的请求，会产生第一个output_token装入output_token_ids。后续的decode过程中，每产出一个token，都会装入这里，这个列表是动态变化的。

spec_token_ids：推测解码策略相关的token_ids，我们可以将其长度视为0。
```python
num_new_tokens = (request.num_tokens_with_spec +
                  request.num_output_placeholders -
                  request.num_computed_tokens)
if (0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens):
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
num_new_tokens = min(num_new_tokens, token_budget)
num_new_tokens = min(num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens)
```

3. 处理多模态输入，视觉编码器会消耗额外的token，因此需要动态调整token_budget。

4. 没有可调度的token时跳过该请求，可能原因: 预算耗尽、编码器缓存满、达到长度限制等。

5. KV缓存分配及抢占循环
   
①为请求分配缓存块

②如果无缓存块可分配，则根据调度策略抢占请求，抢占机制：

优先级策略: 抢占优先级最低的请求（数值最小），优先级相同则看到达时间

FCFS策略: 抢占最早运行的请求（队列末尾）

③抢占操作：资源回收: 释放被抢占请求占用的KV缓存，状态管理: 标记请求为"被抢占"状态，进度重置: 清空已计算token数（需要重新计算）

④被抢占的请求放入等待队列并用preempted_reqs记录，同时防止自我抢占

⑤循环退出条件: （1）如果有kv缓存块可分配则标记can_schedule=True直接退出。（2）如果无缓存块可分配，则不断抢占请求，直到无请求可抢占标记can_schedule=True退出

```python
while True:
    # 分配缓存块: 为请求分配GPU/CPU缓存空间
    new_blocks = self.kv_cache_manager.allocate_slots(
        request,
        num_new_tokens,
        num_lookahead_tokens=self.num_lookahead_tokens)
    # 缓存块分配失败，执行缓存抢占机制
    if new_blocks is None:
        # The request cannot be scheduled.
        # Preempt the lowest-priority request.
        if self.policy == SchedulingPolicy.PRIORITY:
            preempted_req = max(
                self.running,
                key=lambda r: (r.priority, r.arrival_time),
            )
            self.running.remove(preempted_req)
        else:
            preempted_req = self.running.pop()
        # 执行抢占操作
        self.kv_cache_manager.free(preempted_req)
        preempted_req.status = RequestStatus.PREEMPTED
        preempted_req.num_computed_tokens = 0
        if self.log_stats:
            preempted_req.record_event(
                EngineCoreEventType.PREEMPTED, scheduled_timestamp)
        # 放回等待队列前端
        self.waiting.prepend_request(preempted_req)
        # 记录被抢占的请求
        preempted_reqs.append(preempted_req)
        # 直到无请求可以抢占，退出循环
        if preempted_req == request:
            # No more request to preempt.
            can_schedule = False
            break
    else:
        # The request can be scheduled.
        can_schedule = True
        break
if not can_schedule:
    break
# 完整性检查: 确保成功分配了缓存块
assert new_blocks is not None
```

抢占机制设计理念：

+ 贪婪抢占，持续抢占直到当前请求可以调度，确保高优先级请求能够获得资源
+ 公平性保障，基于策略选择要抢占的请求，避免随意抢占导致的不公平
+ 避免死锁，防止自我抢占的无限循环，确保算法总能终止

6. 执行running队列请求调度
   
①将成功调度的请求加入 scheduled_running_reqs

②调度请求记录。记录使用结构化输出（如JSON格式）的请求，记录分配给请求的KV缓存块ID，记录调度的token数量

③推测解码处理。计算推测token数: 确定需要多少推测token，移除已调度的推测token，将推测token加入调度结果

④编码器缓存管理。记录要处理的编码器输入（如图像特征），为每个编码器输入分配缓存空间

⑤LoRA适配器管理。收集所有需要LoRA适配器的请求

```python
# Schedule the request.
    scheduled_running_reqs.append(request)
    if request.use_structured_output:
        # PERF: in case of chunked prefill,
        # request might not include any new tokens.
        # Therefore, we might introduce some additional
        # cycle to fill in the bitmask, which could be a big no-op.
        # 记录使用结构化输出（如JSON格式）的请求
        structured_output_request_ids[request.request_id] = req_index
    # 缓存块记录: 记录分配给请求的KV缓存块ID
    req_to_new_block_ids[request.request_id] = (
        new_blocks.get_block_ids())
    # Token数记录: 记录调度的token数量
    num_scheduled_tokens[request.request_id] = num_new_tokens
    # 从总token预算中扣除已分配的数量
    token_budget -= num_new_tokens
    req_index += 1

    # Speculative decode related.
    # 计算推测token数: 确定需要多少推测token。spec_token_ids: 请求的推测token列表
    if request.spec_token_ids:
        num_scheduled_spec_tokens = (num_new_tokens +
                                     request.num_computed_tokens -
                                     request.num_tokens)
        if num_scheduled_spec_tokens > 0:
            # Trim spec_token_ids list to num_scheduled_spec_tokens.
            # 移除已调度的推测token
            del request.spec_token_ids[num_scheduled_spec_tokens:]
            # 将推测token加入调度结果
            scheduled_spec_decode_tokens[request.request_id] = (
                request.spec_token_ids)

    # Encoder-related.
    if encoder_inputs_to_schedule:
        # 记录要处理的编码器输入（如图像特征）
        scheduled_encoder_inputs[request.request_id] = (
            encoder_inputs_to_schedule)
        # Allocate the encoder cache.
        # 为每个编码器输入分配缓存空间
        for i in encoder_inputs_to_schedule:
            self.encoder_cache_manager.allocate(request, i)
        # 更新剩余的编码器预算
        encoder_budget = new_encoder_budget

# Record the LoRAs in scheduled_running_reqs
scheduled_loras: set[int] = set()
if self.lora_config:
    scheduled_loras = set(
        req.lora_request.lora_int_id for req in scheduled_running_reqs
        if req.lora_request and req.lora_request.lora_int_id > 0)
    assert len(scheduled_loras) <= self.lora_config.max_loras
```
<img width="720" height="816" alt="image" src="https://github.com/user-attachments/assets/2efcd5c6-7d5b-44e9-a3ec-be4e136b7653" />

## 2-3 调度WAITING requests
1. 创建skipped_waiting_requests队列，临时存放因各种原因需要跳过的请求，最后会重新放回主等待队列前端

2. 创建waiting处理循环。条件：非抢占请求，waiting队列非空，token预算充足，不超过最大running请求数
   
3. 跳过远程KV等待状态的请求。如果仍在等待远程kv请求，则该请求更新状态为WAITING。否则，从waiting队列中取出，跳过该请求
```python
# KVTransfer: skip request if still waiting for remote kvs.
if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
    is_ready = self._update_waiting_for_remote_kv(request)
    if is_ready:
        request.status = RequestStatus.WAITING
    else:
        logger.debug(
            "%s is still in WAITING_FOR_REMOTE_KVS state.",
            request.request_id)
        self.waiting.pop_request()
        skipped_waiting_requests.prepend_request(request)
        continue
```

4. 跳过文法FSM编译等待状态的请求。等待文法有限状态机编译完成

5. LoRA数量限制。已调度的LoRA优先，新LoRA请求跳过。跳过超过最大LoRA数的请求

6. 如果num_computed_tokens=0，计算已缓存Token：本地kv缓存token + 远程kv缓存token。否则，更新num_computed_tokens
   
7. Token数计算和约束

①更新num_new_tokens为request.num_tokens - num_computed_tokens

②长预填充限制，防止单个请求占用过多资源。更新num_new_tokens为long_prefill_token_threshold

③分块预填充检查，跳过不分块且预算不足的请求
```python
# KVTransfer: loading remote KV, do not allocate for new work.
if load_kv_async:
    assert num_external_computed_tokens > 0
    num_new_tokens = 0
# Number of tokens to be scheduled.
else:
    # We use `request.num_tokens` instead of
    # `request.num_prompt_tokens` to consider the resumed
    # requests, which have output tokens.
    num_new_tokens = request.num_tokens - num_computed_tokens
    if (0 < self.scheduler_config.long_prefill_token_threshold
            < num_new_tokens):
        num_new_tokens = (
            self.scheduler_config.long_prefill_token_threshold)

    # chunked prefill has to be enabled explicitly to allow
    # pooling requests to be chunked
    if not self.scheduler_config.chunked_prefill_enabled and \
        num_new_tokens > token_budget:
        self.waiting.pop_request()
        skipped_waiting_requests.prepend_request(request)
        continue

    num_new_tokens = min(num_new_tokens, token_budget)
    assert num_new_tokens > 0

    # Schedule encoder inputs.
    if request.has_encoder_inputs:
        (encoder_inputs_to_schedule, num_new_tokens,
         new_encoder_budget
         ) = self._try_schedule_encoder_inputs(
             request, num_computed_tokens, num_new_tokens,
             encoder_budget)
        if num_new_tokens == 0:
            # The request cannot be scheduled.
            break
```

8. KV缓存分配，包括本地kv cache和远程kv cache
   
9. 请求状态转换
   
①从waiting队列取出请求，检查是否还在加载kv，如果是则跳过该条请求

②请求加入running队列

③请求分类记录。WAITING请求加入scheduled_new_reqs，PREEMPTED请求加入scheduled_resumed_reqs

④更新num_scheduled_tokens、token_budget

⑤更新请求状态为RUNNING

⑥兼容编码器输入
```python
# Request was already popped from self.waiting
# unless it was re-added above due to new_blocks being None.
request = self.waiting.pop_request()
if load_kv_async:
    # If loading async, allocate memory and put request
    # into the WAITING_FOR_REMOTE_KV state.
    skipped_waiting_requests.prepend_request(request)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    continue

if request.use_structured_output:
    structured_output_request_ids[request.request_id] = (
        req_index)
req_index += 1
self.running.append(request)
if self.log_stats:
    request.record_event(EngineCoreEventType.SCHEDULED,
                         scheduled_timestamp)
if request.status == RequestStatus.WAITING:
    scheduled_new_reqs.append(request)
elif request.status == RequestStatus.PREEMPTED:
    scheduled_resumed_reqs.append(request)
else:
    raise RuntimeError(
        f"Invalid request status: {request.status}")

if self.lora_config and request.lora_request:
    scheduled_loras.add(request.lora_request.lora_int_id)
req_to_new_block_ids[request.request_id] = (
    self.kv_cache_manager.get_block_ids(request.request_id))
num_scheduled_tokens[request.request_id] = num_new_tokens
token_budget -= num_new_tokens
request.status = RequestStatus.RUNNING
request.num_computed_tokens = num_computed_tokens
# Count the number of prefix cached tokens.
if request.num_cached_tokens < 0:
    request.num_cached_tokens = num_computed_tokens
# Encoder-related.
if encoder_inputs_to_schedule:
    scheduled_encoder_inputs[request.request_id] = (
        encoder_inputs_to_schedule)
    # Allocate the encoder cache.
    for i in encoder_inputs_to_schedule:
        self.encoder_cache_manager.allocate(request, i)
    encoder_budget = new_encoder_budget
```

10. skipped_waiting_requests请求处理。加回waiting队列首端
    
11. 构建SchedulerOutput返回
```python
cached_reqs_data = self._make_cached_request_data(
    scheduled_running_reqs,
    scheduled_resumed_reqs,
    num_scheduled_tokens,
    scheduled_spec_decode_tokens,
    req_to_new_block_ids,
)
scheduler_output = SchedulerOutput(
    scheduled_new_reqs=new_reqs_data,
    scheduled_cached_reqs=cached_reqs_data,
    num_scheduled_tokens=num_scheduled_tokens,
    total_num_scheduled_tokens=total_num_scheduled_tokens,
    scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
    scheduled_encoder_inputs=scheduled_encoder_inputs,
    num_common_prefix_blocks=num_common_prefix_blocks,
    # finished_req_ids is an existing state in the scheduler,
    # instead of being newly scheduled in this step.
    # It contains the request IDs that are finished in between
    # the previous and the current steps.
    finished_req_ids=self.finished_req_ids,
    free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
    structured_output_request_ids=structured_output_request_ids,
    grammar_bitmask=grammar_bitmask,
)
```
<img width="702" height="2236" alt="image" src="https://github.com/user-attachments/assets/caa5f927-26f0-449d-ac47-4d46155f1419" />














