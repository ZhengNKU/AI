# 1. 启动独立进程make_async_mp_client
**调用链**

run_server ->  run_server_worker -> build_async_engine_client -> build_async_engine_client_from_engine_args -> AsyncLLM -> make_async_mp_client

**源码**
```python
@staticmethod
def make_async_mp_client(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    client_addresses: Optional[dict[str, str]] = None,
    client_count: int = 1,
    client_index: int = 0,
) -> "MPClient":
    parallel_config = vllm_config.parallel_config
    client_args = (vllm_config, executor_class, log_stats,
                   client_addresses, client_count, client_index)
    if parallel_config.data_parallel_size > 1:
        if parallel_config.data_parallel_external_lb:
            # External load balancer - client per DP rank.
            return DPAsyncMPClient(*client_args)
        # Internal load balancer - client balances to all DP ranks.
        return DPLBAsyncMPClient(*client_args)
    return AsyncMPClient(*client_args)
```

声明一个用于管理输出处理的异步任务变量，output_handler 是一个后台任务，它的职责是持续地、异步地从EngineCore客户端读取生成的输出（tokens），并将其放入正确的输出流中。初始化为 None 表示这个后台任务尚未启动。
```python
self.output_handler: Optional[asyncio.Task] = None
try:
    # Start output handler eagerly if we are in the asyncio eventloop.
    asyncio.get_running_loop()
    self._run_output_handler()
except RuntimeError:
    pass
```
<img width="955" height="656" alt="image" src="https://github.com/user-attachments/assets/e5694e54-45da-4da9-baf8-e0b11d3dd435" />

# 1.1 MPClient
不同进程启动策略都通过MPClient初始化EngineCore客户端。

示例：

<img width="632" height="542" alt="image" src="https://github.com/user-attachments/assets/9a934660-d913-40b1-9181-33e1e8ee7022" />

```python
class MPClient(EngineCoreClient):
    """
    MPClient: base client for multi-proc EngineCore.
        EngineCore runs in a background process busy loop, getting
        new EngineCoreRequests and returning EngineCoreOutputs

        * pushes EngineCoreRequests via input_socket
        * pulls EngineCoreOutputs via output_socket
    
        * AsyncMPClient subclass for AsyncLLM usage
        * SyncMPClient subclass for LLM usage
    """

    def __init__(
        self,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: Optional[dict[str, str]] = None,
    ):
        self.vllm_config = vllm_config
        # Serialization setup.
        # 使用 MessagePack 进行高效的消息序列化和反序列化。指定 EngineCoreOutputs 为解码目标类型，提高效率。
        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(EngineCoreOutputs)

        # ZMQ setup.
        # ZMQ上下文初始化
        sync_ctx = zmq.Context(io_threads=2)
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx

        # This will ensure resources created so far are closed
        # when the client is garbage collected, even if an
        # exception is raised mid-construction.
        # 资源管理与安全清理
        self.resources = BackgroundResources(ctx=sync_ctx)
        self._finalizer = weakref.finalize(self, self.resources)
        success = False
        try:
            # State used for data parallel.
            self.engines_running = False

            self.stats_update_address: Optional[str] = None
          
            if client_addresses is not None:
                # 连接外部引擎。使用提供的地址直接连接。这用于 DPAsyncMPClient 等场景，引擎由外部系统（如K8s）管理。
                # Engines are managed externally to this client.
                input_address = client_addresses["input_address"]
                output_address = client_addresses["output_address"]
                self.stats_update_address = client_addresses.get(
                    "stats_update_address")
            else:
                # Engines are managed by this client.
                # 内部启动引擎。调用 launch_core_engines
                with launch_core_engines(vllm_config, executor_class,
                                         log_stats) as (engine_manager,
                                                        coordinator,
                                                        addresses):
                    self.resources.coordinator = coordinator
                    self.resources.engine_manager = engine_manager

                (input_address, ) = addresses.inputs
                (output_address, ) = addresses.outputs
                self.stats_update_address = (
                    addresses.frontend_stats_publish_address)
                if coordinator is not None:
                    assert self.stats_update_address == (
                        coordinator.get_stats_publish_address())

            # Create input and output sockets.
            # ROUTER Socket (Input)：这是一个可以处理多个连接的先进Socket。它允许客户端通过指定的 identity 与多个引擎中的某一个进行通信。这是实现负载均衡的基础。
            # PULL Socket (Output)：这是一个订阅/拉取Socket，用于从引擎接收输出消息。
            self.input_socket = self.resources.input_socket = make_zmq_socket(
                self.ctx, input_address, zmq.ROUTER, bind=True)
            self.resources.output_socket = make_zmq_socket(
                self.ctx, output_address, zmq.PULL)

            parallel_config = vllm_config.parallel_config
            dp_size = parallel_config.data_parallel_size
            dp_rank = parallel_config.data_parallel_rank
            dp_local_size = parallel_config.data_parallel_size_local
            offline_mode = parallel_config.data_parallel_rank_local is not None
            # Client manages local+remote EngineCores in pure internal LB case.
            # Client manages local EngineCores in hybrid and external LB case.
            local_engines_only = (parallel_config.data_parallel_hybrid_lb
                                  or parallel_config.data_parallel_external_lb)

            num_ranks = dp_local_size if local_engines_only else dp_size
            self.engine_ranks_managed = [dp_rank] if offline_mode else list(
                range(dp_rank, dp_rank + num_ranks))
            assert parallel_config.data_parallel_size_local <= len(
                self.engine_ranks_managed)

            # ZMQ identity of each engine that this client will talk to.
            # 为每个被管理的引擎创建一个唯一的身份标识（ZMQ Identity），通常就是它的rank。
            self.core_engines: list[EngineIdentity] = [
                rank.to_bytes(2, "little")
                for rank in self.engine_ranks_managed
            ]

            # Wait for ready messages from each engine on the input socket.
            # 代码会等待所有引擎通过Input Socket发送一个初始消息（"ready"信号）。这确保了所有引擎进程都已成功启动并连接，客户端之后才开始发送实际请求。
            identities = set(self.core_engines)
            sync_input_socket = zmq.Socket.shadow(self.input_socket)
            while identities:
                if not sync_input_socket.poll(timeout=600_000):
                    raise TimeoutError("Timed out waiting for engines to send"
                                       "initial message on input socket.")
                identity, _ = sync_input_socket.recv_multipart()
                identities.remove(identity)

            self.core_engine: EngineIdentity = self.core_engines[0]
            self.utility_results: dict[int, AnyFuture] = {}

            # Request objects which may contain pytorch-allocated tensors
            # that we need to keep references to until zmq is done with the
            # underlying data.
            self.pending_messages = deque[tuple[zmq.MessageTracker, Any]]()

            # Start monitoring engine core processes for unexpected failures
            # 启动一个后台线程，监控引擎进程的健康状态。如果某个引擎进程意外崩溃，客户端能够检测到并可能触发错误处理或重启机制。
            self.start_engine_core_monitor()

            success = True
        finally:
            if not success:
                self._finalizer()

```
<img width="679" height="1995" alt="image" src="https://github.com/user-attachments/assets/f3195ce1-7476-4da8-80aa-80d3b7aaf928" />

## 1.2 不同进程启动策略
（1）没有启用数据并行，只有一个模型副本在运行

类名：AsyncMPClient

1个客户端进程 ↔ 1个EngineCore进程

**工作流程：**

+ 客户端接收请求。
+ 通过进程间通信（如 ZMQ）直接将请求发送给唯一的引擎进程。
+ 从 outputs_queue 中读取该引擎的返回结果。

（2）多数据并行(Multiple Data Parallelism)且 存在外部负载均衡器。例如，使用 Kubernetes Service 或 NGINX 来分发请求

类名：DPAsyncMPClient，继承AsyncMPClient

N个客户端进程 ↔ N个引擎进程（一对一）。每个客户端实例只与一个特定的引擎副本通信。外部负载均衡器负责将请求路由到不同的客户端。

**工作流程：**

+ 外部负载均衡器接收用户请求。
+ 外部LB根据其策略（轮询、最少连接等）将请求转发给众多 DPAsyncMPClient 实例中的一个。
+ 该客户端实例将请求转发给它所专属的那个引擎副本。
+ 从该副本获取结果并返回。

（3）多数据并行但没有外部负载均衡器，需要客户端自己实现负载均衡

类名：DPLBAsyncMPClient，继承DPAsyncMPClient

1个客户端进程 ↔ N个引擎进程（一对多）。一个客户端实例知晓所有引擎副本，并负责决定将请求发送给谁。

**工作流程：**

+ 客户端接收请求。
+ 客户端根据负载均衡算法（如查看 lb_engines）选择一个最合适的引擎副本。
+ 将请求发送给该引擎，并在 reqs_in_flight 中记录映射关系 {request_id: engine_id}。
+ 当该引擎返回结果后，从映射表中清除记录。
+ 如果收到 abort(request_id) 指令，客户端会查找 reqs_in_flight 找到对应的 engine_id，然后向那个特定的引擎发送中止命令。


# 2. MPClient转发请求

**调用链**

/v1/chat/completions -> create_chat_completion -> self.engine_client.generate(不使用beam search) -> generate(async_llm.py) -> add_request -> _add_request -> add_request_async

<img width="1279" height="918" alt="image" src="https://github.com/user-attachments/assets/c8aa40f7-46c7-4966-a402-75923d057c15" />

**源码**

```python
await self.engine_core.add_request_async(request)

async def add_request_async(self, request: EngineCoreRequest) -> None:
    # 确保统计更新任务运行，详见2.1
    self._ensure_stats_update_task()
    
    # 为请求添加元数据，包括当前波次标识和客户端索引标识
    request.current_wave = self.current_wave
    request.client_index = self.client_index
    
    # engine选择策略，详见2.2
    chosen_engine = self.get_core_engine_for_request(request)
    # 异步发送ADD类型的请求到选定的引擎，返回一个可等待对象。
    to_await = self._send_input(EngineCoreRequestType.ADD, request,
                                chosen_engine)
    if not self.engines_running:
        # 当引擎尚未运行时，通过ZMQ通知apiserver启动引擎进程
        # Notify coordinator that we're sending a request
        req_msg = msgspec.msgpack.encode(("FIRST_REQ", chosen_engine))
        await self.first_req_send_socket.send(req_msg)

    await to_await
    
    # 创建一个输出队列处理任务EngineCoreOutputQueueTask，负责异步接收和处理来自engineCore的输出结果
    self._ensure_output_queue_task()
```
<img width="762" height="233" alt="image" src="https://github.com/user-attachments/assets/80fcaa7b-620e-4e2e-b41b-f1106253c3fd" />

## 2.1 _ensure_stats_update_task
<details> 
    <summary>源码</summary>
    
    ```python
    def _ensure_stats_update_task(self):
    resources = self.resources
    # 任务已存在，直接返回
    if resources.stats_update_task is not None:
        return

    assert self.stats_update_address is not None
    assert len(self.engine_ranks_managed) > 0
    # NOTE: running and waiting counts are all global from
    # the Coordinator include all global EngineCores. This
    # slice includes just the cores managed by this client.
    count_slice = slice(self.engine_ranks_managed[0],
                        self.engine_ranks_managed[-1] + 1)

    async def run_engine_stats_update_task():
        # 创建两个ZMQ套接字。
        # XSUB用于接收套接字,既能接收来自PUB/XPUB的消息,也能发送消息给XPUB。订阅协调器的状态广播
        # PAIR套接字，全双工通信
        with (make_zmq_socket(self.ctx,
                              self.stats_update_address,
                              zmq.XSUB,
                              linger=0) as socket,
              make_zmq_socket(self.ctx,
                              self.first_req_sock_addr,
                              zmq.PAIR,
                              bind=False,
                              linger=0) as first_req_rcv_socket):
            assert isinstance(socket, zmq.asyncio.Socket)
            assert isinstance(first_req_rcv_socket, zmq.asyncio.Socket)
            self.resources.stats_update_socket = socket
            self.resources.first_req_rcv_socket = first_req_rcv_socket
            # Send subscription message.
            # 发送订阅消息
            await socket.send(b'\x01')
            
            poller = zmq.asyncio.Poller()
            poller.register(socket, zmq.POLLIN) 
            # 首次请求/扩缩容
            poller.register(first_req_rcv_socket, zmq.POLLIN)

            while True:
                events = await poller.poll()
                if not self.engines_running and len(events) == 2 or (
                        events[0][0] == first_req_rcv_socket):
                    # Check if this is a regular request notification or
                    # scale up notification
                    # 通过PAIR套接字接收扩缩容指令
                    buf = first_req_rcv_socket.recv(
                        flags=zmq.NOBLOCK).result()

                    decoded = msgspec.msgpack.decode(buf)
                    # 弹性扩缩容处理
                    if isinstance(
                            decoded,
                        (list, tuple)) and len(decoded) == 2 and decoded[
                            0] == "SCALE_ELASTIC_EP":
                        # Extract new engine count from the decoded message
                        # 获取新引擎数量
                        new_engine_count = decoded[1] 
                        # Send scale up notification to coordinator
                        # 通过XSUB套接字转发扩缩容消息给协调器
                        scale_msg = msgspec.msgpack.encode(
                            ("SCALE_ELASTIC_EP", new_engine_count))
                        await socket.send(scale_msg)
                        continue

                    # we're sending a request while the engines are
                    # paused, so that it can wake the others up
                    # (to run dummy EP loop).
                    # 首次请求标记引擎已启动，通知协调器目标引擎信息
                    assert decoded[0] == "FIRST_REQ"
                    target_eng_index = decoded[1]
                    self.engines_running = True
                    msg = msgspec.msgpack.encode(
                        (target_eng_index, self.current_wave))
                    await socket.send(msg)

                buf = None
                while True:
                    # Drain all stats events (we only care about latest).
                    # 清空所有待处理状态消息（只关心最新状态）
                    future: asyncio.Future[bytes] = socket.recv(
                        flags=zmq.NOBLOCK)
                    if isinstance(future.exception(), zmq.Again):
                        break
                    # 获取最新消息
                    buf = future.result()
                if buf is None:
                    continue

                # Update local load-balancing state.
                # 更新负载均衡状态
                counts, wave, running = msgspec.msgpack.decode(buf)
                self.current_wave = wave
                self.engines_running = running
                if counts is not None:
                    sliced_counts = counts[count_slice]
                    self.lb_engines = sliced_counts
                    logger.debug("Received counts: %s (%s)", sliced_counts,
                                 count_slice)

    resources.stats_update_task = asyncio.create_task(
        run_engine_stats_update_task())
    ```
</details>

+ 订阅协调器状态广播
+ 监听首次请求和扩缩容通知
+ 处理扩缩容请求并转发
+ 更新本地负载均衡状态
+ 持续循环保持状态同步

**扩缩容：**

系统不是固定数量的引擎，而是根据负载动态调整：

扩容（Scale Up）: 当负载增加时启动更多引擎

缩容（Scale Down）: 当负载减少时关闭闲置引擎

## 2.2 get_core_engine_for_request ***

**源码**

```python
def get_core_engine_for_request(
        self, request: EngineCoreRequest) -> EngineIdentity:
    # Engines are in rank order.
    if (eng_index := request.data_parallel_rank) is None:
        # 不是dp获取各引擎的负载状态
        current_counts = self.lb_engines
        # TODO use P2C alg for larger DP sizes
        num_engines = len(current_counts)
        min_score = sys.maxsize
        eng_index = 0
        # 负载均衡算法
        for i in range(num_engines):
            # Start from client_index to help with balancing when engines
            # are empty.
            idx = (self.eng_start_index + i) % num_engines  # 轮询起始点，确保多个客户端从不同位置开始，避免所有客户端同时选择同一个"最优"引擎
            waiting, running = current_counts[idx]  # 获取等待数和运行数
            score = waiting * 4 + running  # 加权评分算法
            if score < min_score:
                min_score = score
                eng_index = idx
        # Increment local waiting count for better balancing between stats
        # updates from the coordinator (which happen every 100ms).
        # 临时增加等待计数，在协调器状态更新间隔期间保持负载均衡
        current_counts[eng_index][0] += self.client_count  
    
    # 如果是数据并行(dp)，则直接指定engine
    chosen_engine = self.core_engines[eng_index]
    # Record which engine is chosen for this request, to handle aborts.
    self.reqs_in_flight[request.request_id] = chosen_engine
    return chosen_engine
```

整体功能：根据请求特性和引擎当前负载，选择最优的引擎来处理请求。

（1）如果是数据并行(dp)，则根据engine_ranks_managed直接指定engine

（2）如果不是dp，需要自己实现负载均衡，算法：score = waiting * 4 + running。等待队列的权重是运行任务的4倍，避免队列堆积比减少运行任务更重要





