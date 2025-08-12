**EntryPoints**:
离线批量推理：vllm/entrypoints/api_server.py 或 LLM.generate()（vllm/llm.py）

在线服务：vllm/entrypoints/openai/api_server.py（启动 OpenAI 兼容 API）


# 1. 在线服务
参考链接：https://zhuanlan.zhihu.com/p/1896477903434258024 

python -m vllm.entrypoints.openai.api_server --model modelname

<img width="4854" height="2193" alt="image" src="https://github.com/user-attachments/assets/0513162f-bcc5-4614-af24-a75411e74ca5" />


## 1.1 api_server关键步骤解读
### 1.1.1 参数解析

源码：make_arg_parser

make_arg_parser 是 vLLM API 服务器的 命令行参数配置中心，负责定义所有与 API 服务相关的配置选项。它扩展了 FlexibleArgumentParser，支持以下功能：

服务网络配置（如主机、端口、SSL）

请求处理控制（如跨域、中间件、日志）

模型扩展功能（如 LoRA、Prompt Adapter）

调试与监控（如日志级别、请求跟踪）

***基础服务配置***
| 参数名                      | 类型   | 默认值   | 说明                                                                 |
|-----------------------------|--------|----------|----------------------------------------------------------------------|
| `--host`                    | str    | None     | 服务监听的主机名（None表示所有接口）                                  |
| `--port`                    | int    | 8000     | 服务监听的端口号                                                     |
| `--uvicorn-log-level`       | str    | "info"   | Uvicorn日志级别（debug/info/warning/error/critical/trace）            |
| `--disable-uvicorn-access-log` | bool | False    | 禁用Uvicorn访问日志                                                  |

***安全与跨域配置***
| 参数名                | 类型   | 默认值  | 说明                                                                 |
|-----------------------|--------|---------|----------------------------------------------------------------------|
| `--allow-credentials` | bool   | False   | 允许跨域请求携带凭据                                                 |
| `--allowed-origins`   | JSON   | ["*"]   | 允许的跨域来源（如["http://localhost"]）                             |
| `--allowed-methods`   | JSON   | ["*"]   | 允许的HTTP方法（如["GET","POST"]）                                   |
| `--allowed-headers`   | JSON   | ["*"]   | 允许的请求头                                                         |
| `--api-key`           | str    | None    | API密钥认证                                                          |

***模型扩展功能***
| 参数名                      | 类型   | 默认值  | 说明                                                                 |
|-----------------------------|--------|---------|----------------------------------------------------------------------|
| `--lora-modules`            | str    | None    | LoRA适配器配置（格式：name=path或JSON）                              |
| `--prompt-adapters`         | str    | None    | Prompt适配器配置（格式：name=path）                                  |
| `--chat-template`           | str    | None    | 自定义聊天模板文件或内容                                             |
| `--chat-template-content-format` | str | "auto" | 聊天内容渲染格式（string/openai）                                   |

***高级请求控制***
| 参数名                      | 类型   | 默认值  | 说明                                                                 |
|-----------------------------|--------|---------|----------------------------------------------------------------------|
| `--enable-auto-tool-choice` | bool   | False   | 自动选择工具调用                                                     |
| `--tool-call-parser`        | str    | None    | 工具调用解析器（如openai）                                           |
| `--middleware`              | str    | []      | 添加ASGI中间件（支持动态导入）                                       |
| `--root-path`               | str    | None    | 反向代理路径前缀                                                     |

***调试与监控***
| 参数名                      | 类型   | 默认值  | 说明                                                                 |
|-----------------------------|--------|---------|----------------------------------------------------------------------|
| `--log-config-file`         | str    | 环境变量 | 日志配置文件路径                                                     |
| `--max-log-len`             | int    | None    | 日志中打印的最大提示长度                                             |
| `--enable-request-id-headers` | bool | False   | 在响应中添加X-Request-Id头                                          |
| `--enable-server-load-tracking` | bool | False | 启用服务器负载指标监控                                               |

### 1.1.2 run_server

**（1）预启动配置函数** 

**源码：listen_address, sock = setup_server(args)**

+ 参数校验

+ 防止端口竞争的安全措施：调用create_server_socket方法创建一个服务器套接字，绑定到指定的主机和端口。需要确保在引擎初始化之前绑定端口, 避免与其他进程（如Ray）发生竞争。

+ 调整系统的文件描述符限制，避免高并发请求时因资源不足导致问题

+ 信号处理设置

+ 监听地址格式化


**（2）启动一个独立的 vLLM API 服务 Worker**

**源码：await run_server_worker(listen_address, sock, args, \*\*uvicorn_kwargs)**

+ 调用 build_async_engine_client 方法异步创建引擎客户端，用于与后端引擎通信

+ 构建 FastAPI 应用程序，配置路由、中间件和异常处理

+ init_app_state初始化 FastAPI 应用的状态管理。包括支持单模型多别名、通过 max_log_len 限制日志中提示文本长度，防止敏感信息泄露、聊天模板解析（当自定义模板与 HuggingFace 官方模板不一致时发出警告）、LoRA 模块合并、创建模型管理实例并初始化静态 LoRA、为后续实现基于负载的自动扩缩容预留接口

+ 调用serve_http方法启动Uvicorn HTTP 服务，并返回一个任务对象，用于等待服务器关闭。包括路由信息打印、Uvicorn 初始化、主服务任务和watchdog任务异步启动、SSL 热更新、优雅关闭

+ 释放端口

```python
async def run_server_worker(listen_address,
                            sock,
                            args,
                            client_config=None,
                            **uvicorn_kwargs) -> None:
    """Run a single API server worker."""

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    server_index = client_config.get("client_index", 0) if client_config else 0

    # Load logging config for uvicorn if specified
    # 配置日志文件格式
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        uvicorn_kwargs['log_config'] = log_config

    async with build_async_engine_client(args, client_config) as engine_client:
        # 构建 FastAPI 应用程序，配置路由、中间件和异常处理
        app = build_app(args)
            
        # 从vllm引擎客户端获取配置
        vllm_config = await engine_client.get_vllm_config()
        # 初始化 FastAPI 应用的状态管理
        await init_app_state(engine_client, vllm_config, app.state, args)

        logger.info("Starting vLLM API server %d on %s", server_index,
                    listen_address)
        # 调用serve_http方法启动 HTTP 服务，并返回一个任务对象，用于等待服务器关闭
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()
```




