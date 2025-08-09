**EntryPoints**:
离线批量推理：vllm/entrypoints/api_server.py 或 LLM.generate()（vllm/llm.py）。
在线服务：vllm/entrypoints/openai/api_server.py（启动 OpenAI 兼容 API）


# 1. 在线服务
参考链接：https://zhuanlan.zhihu.com/p/1896477903434258024 
python -m vllm.entrypoints.openai.api_server --model modelname

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





