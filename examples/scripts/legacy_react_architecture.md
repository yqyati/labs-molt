# 旧框架 Legacy ReAct 说明

本文说明当前科学智能体项目中旧框架（Legacy ReAct）的架构、入口文件和一次评测的调用链。旧框架与 `science-agent-next` 的 Native AgentLoop 是两条不同的执行路径。

## 1. 代码位置

旧框架核心代码位于：

```text
/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/liuzhuo/science-agent
```

评测调度和结果后处理代码位于：

```text
/mnt/shared-storage-gpfs2/ai4sgi-gpfs2/scientific_reasoning_and_research/yangqingyu/science-agent-next
```

`science-agent-next` 保留了 Legacy 评测兼容路径，同时新增了 Native AgentLoop、Team、checkpoint 和工具调度能力。使用 Legacy 评测脚本时，实际单题推理仍由 `liuzhuo/science-agent` 中的旧 ReAct 实现完成。

## 2. 调用链

```text
RJob / 本地启动脚本
  |
  v
science-agent-next/runs/<run_name>/eval/run_eval_external.sh
  |
  v
science-agent-next/scripts/agent_eval_suite.sh
  |
  v
science-agent-next/inference/run_multi_benchmark_react_tools.py
  |
  v
liuzhuo/science-agent/inference/run_multi_react_tools.py
  |
  v
liuzhuo/science-agent/inference/react_agent_openai_api_tools.py
  |
  +--> OpenAI-compatible model API
  |
  +--> Tool Gateway: search / visit / python / google_scholar
  |
  v
rollout JSONL
  |
  v
Step2 normalization -> Step3 Judge -> Step5 Excel
```

## 3. 关键入口文件

### 3.1 评测总入口

```text
science-agent-next/scripts/agent_eval_suite.sh
```

作用：

- 读取模型地址、模型名、API key、数据集、工具和并发配置；
- 选择 benchmark 并生成 `agent_input.jsonl`；
- 创建评测 manifest；
- 启动 Legacy 多 benchmark runner；
- 生成 Step2、Step3 Judge 和 Step5 Excel 结果。

典型调用：

```bash
BENCHMARKS=hle_all_mini@noguard \
MODEL_BACKEND=api \
TOOLS=search,visit,python,google_scholar \
MAX_WORKERS=8 \
bash scripts/agent_eval_suite.sh
```

### 3.2 Legacy 多题调度入口

```text
science-agent-next/inference/run_multi_benchmark_react_tools.py
```

作用：

- 读取 `agent_eval_suite.sh` 生成的 manifest；
- 按题目和 benchmark 调度任务；
- 使用 worker pool 控制跨题并发；
- 为每道题调用旧版 ReAct runner；
- 写入 rollout 日志和统计信息。

### 3.3 Legacy 多题 runner

```text
liuzhuo/science-agent/inference/run_multi_react_tools.py
```

作用：

- 初始化旧框架的模型客户端、工具注册和运行参数；
- 读取多题输入；
- 调用单题 `MultiTurnReactAgent`；
- 持久化每题的消息、工具调用、最终答案和终止状态。

### 3.4 Legacy 单题核心 Agent

```text
liuzhuo/science-agent/inference/react_agent_openai_api_tools.py
```

这是旧框架真正执行 ReAct 循环的核心文件，负责：

- 构造 system/user/tool 消息；
- 请求 OpenAI-compatible `/v1/chat/completions`；
- 解析 assistant 文本和 tool calls；
- 执行工具并把结果回填到下一次模型请求；
- 根据模型返回决定继续调用工具或结束；
- 保存 reasoning、最终输出、工具计数和错误信息。

## 4. 旧框架一次推理的基本流程

```text
读取一道题
  |
  v
构造初始 prompt + 工具定义
  |
  v
请求模型
  |
  +-- 无 tool call --> 保存最终回答并结束
  |
  +-- 有 tool call --> 执行工具
                         |
                         v
                   将工具结果追加到历史消息
                         |
                         v
                   再次请求模型
```

旧框架默认按消息顺序处理工具调用。一次 assistant 响应中产生多个工具调用时，是否并行执行取决于旧 runner 和工具客户端的具体配置；它没有 Native AgentLoop 的统一 `SCIENCE_AGENT_MAX_PARALLEL_TOOLS` 调度层。因此不能仅凭工具列表判断旧框架一定会并行。

## 5. 工具层

旧框架的工具实现和适配器位于：

```text
liuzhuo/science-agent/inference/tool_search_mcp_r1.py
liuzhuo/science-agent/inference/tool_visit_mcp_r1.py
liuzhuo/science-agent/inference/tool_python.py
liuzhuo/science-agent/inference/tool_scholar.py
liuzhuo/science-agent/inference/tool_runtime_registry.py
liuzhuo/science-agent/inference/tool_provider_strategy.py
```

工具通常通过 Gateway 调用远程服务。模型服务和工具 Gateway 是两个不同的网络请求目标：

```text
模型请求：BASE_URL / OPENAI_API_BASE
工具请求：SCIENCE_AGENT_GATEWAY_URL 或 TOOL_GATEWAY_URL
```

## 6. Gateway 配置

Gateway 是旧框架工具调用的关键中间层。Legacy 进程通常不直接访问 Search、网页、Python 或 Scholar 的外部服务，而是把工具请求发送到 Gateway，再由 Gateway 路由到对应后端。

当前评测环境使用的 Gateway 地址通常是：

```bash
SCIENCE_AGENT_GATEWAY_URL=http://100.102.21.127:8765
```

旧评测脚本会把它同时导出为兼容变量：

```bash
export SCIENCE_AGENT_GATEWAY_URL
export TOOL_GATEWAY_URL="$SCIENCE_AGENT_GATEWAY_URL"
export AGENT_TOOL_BACKEND=remote
```

如果 `SCIENCE_AGENT_GATEWAY_URL` 为空，脚本才会回退到本地工具模式：

```bash
AGENT_TOOL_BACKEND=local
```

正式远程 Gateway 配置通常还包括：

```bash
SCIENCE_AGENT_GATEWAY_TIMEOUT=21600
SCIENCE_AGENT_GATEWAY_CACHE_READ=1
SCIENCE_AGENT_GATEWAY_CACHE_WRITE=1
SCIENCE_AGENT_GATEWAY_CACHE_REFRESH=0
SCIENCE_AGENT_GATEWAY_HEARTBEAT_INTERVAL_SEC=60
SCIENCE_AGENT_RUN_HEARTBEAT_TTL_SEC=120
SCIENCE_AGENT_GATEWAY_CANCEL_ON_EXIT=1
SCIENCE_AGENT_GATEWAY_CANCEL_TIMEOUT_SEC=10
```

这些变量的作用是：

| 变量 | 作用 |
|---|---|
| `SCIENCE_AGENT_GATEWAY_URL` | 远程 Gateway 根地址，工具请求发往这里 |
| `TOOL_GATEWAY_URL` | 旧工具客户端使用的兼容别名 |
| `AGENT_TOOL_BACKEND=remote` | 强制使用远程 Gateway 工具后端 |
| `SCIENCE_AGENT_GATEWAY_TIMEOUT` | Gateway 工具请求总超时时间，旧模板默认 21600 秒 |
| `SCIENCE_AGENT_GATEWAY_CACHE_READ` | 是否读取 Gateway 缓存 |
| `SCIENCE_AGENT_GATEWAY_CACHE_WRITE` | 是否写入 Gateway 缓存 |
| `SCIENCE_AGENT_GATEWAY_CACHE_REFRESH` | 是否忽略旧缓存并刷新 |
| `SCIENCE_AGENT_GATEWAY_HEARTBEAT_INTERVAL_SEC` | 运行期间向 Gateway 发送心跳的间隔 |
| `SCIENCE_AGENT_RUN_HEARTBEAT_TTL_SEC` | Gateway 认为运行仍存活的 TTL |
| `SCIENCE_AGENT_GATEWAY_CANCEL_ON_EXIT` | RJob 退出时是否取消 Gateway 侧运行 |

Gateway 相关的 Judge 和 CritPt 路由也使用同一 Gateway：

```bash
JUDGE_PROXY_URL=http://100.102.21.127:8765/v1/judge
JUDGE_BASE_URL=http://100.102.21.127:8765/v1/judge
CRITPT_EVAL_PROXY_URL=http://100.102.21.127:8765/v1/critpt/evaluate
```

模型服务地址与 Gateway 地址必须分别配置。例如：

```bash
# 模型服务
BASE_URL=http://s-20260723104913-x2p77.ailab-ai4sgi.pjh-service.org.cn/v1
OPENAI_API_BASE="$BASE_URL"
MODEL_NAME=deepseek_v4_flash
OPENAI_API_KEY=123

# 工具 Gateway
SCIENCE_AGENT_GATEWAY_URL=http://100.102.21.127:8765
TOOL_GATEWAY_URL="$SCIENCE_AGENT_GATEWAY_URL"
AGENT_TOOL_BACKEND=remote
```

不要把模型服务 URL 填到 `SCIENCE_AGENT_GATEWAY_URL`，也不要把 Gateway URL 填到 `BASE_URL`。前者负责 `/v1/chat/completions`，后者负责工具路由，例如 `/v1/tool_call`。

### Gateway 与代理的关系

RJob 内的 Legacy 进程访问内网 Gateway 时，应把 Gateway 和模型服务加入 `NO_PROXY`，避免内网请求经过外部 HTTP 代理。旧模板中的典型设置为：

```bash
NO_PROXY="localhost,127.0.0.1,::1,100.102.21.127,\
s-20260723141551-4q489.ailab-ai4sgi.pjh-service.org.cn,\
.pjh-service.org.cn,pjh-service.org.cn,.h.pjlab.org.cn,h.pjlab.org.cn"
export NO_PROXY
export no_proxy="$NO_PROXY"
```

这里的 `NO_PROXY` 只控制评测进程到内网服务的访问路径。Search、网页读取等外部联网所需的代理由 Gateway 服务端自己的部署环境负责，属于 Gateway 启动配置，不应该在每个 Legacy 评测脚本里重复硬编码。Gateway 后端的 Search、Visit、Python、Scholar 超时和密钥通常配置在 Gateway 的 `configs/env/gateway_pool.env` 或对应服务环境中。

提交前可以先检查 Gateway 是否可用：

```bash
test -n "$SCIENCE_AGENT_GATEWAY_URL"
curl -sS --max-time 10 \
  "$SCIENCE_AGENT_GATEWAY_URL/v1/admin/router/backends"
```

健康检查失败时，Legacy 模型请求可能仍然正常，但所有工具调用会失败；这类失败会表现为技术失败题，不能当作模型答错。

## 7. 运行配置的主要环境变量

| 变量 | 作用 |
|---|---|
| `BASE_URL` / `OPENAI_API_BASE` | OpenAI-compatible 模型服务地址 |
| `API_KEY` / `OPENAI_API_KEY` | 模型服务认证信息 |
| `MODEL_NAME` / `SERVED_MODEL_NAME` | 请求使用的模型名 |
| `BENCHMARKS` | 数据集及 guard 规格，例如 `hle_all_mini@noguard` |
| `TOOLS` | 工具集合，例如 `search,visit,python,google_scholar` |
| `MAX_WORKERS` | 跨题并发数 |
| `MAX_OUTPUT_TOKENS` | 单次模型输出上限 |
| `MAX_LLM_CALL_PER_RUN` | 单题模型调用预算 |
| `SCIENCE_AGENT_GATEWAY_URL` | 工具 Gateway 地址 |
| `AGENT_EVAL_ENABLE_MODEL_IMAGE_INPUTS` | 是否把题目图片传给模型 |
| `AGENT_EVAL_URL_GUARD` | 是否启用 URL guard |
| `RETRY_FAILED_ROLLOUTS` | 是否重试失败题目 |

旧框架评测中，`MAX_WORKERS` 主要控制不同题目之间的并发；它不等于单题内部工具并发数。

## 8. 结果文件

一个典型运行目录如下：

```text
science-agent-next/output/agent_eval/<run_name>/<benchmark>_noguard/
├── agent_input.jsonl                         # 实际送入 rollout 的题目
├── rollout/
│   ├── iter1.jsonl                           # Legacy 原始轨迹
│   └── agent_eval_stats.json                 # 运行统计
├── <run>.<step2>.jsonl                       # 规范化轨迹
├── <run>_o3-mini.step3.judged.jsonl         # Judge 结果
├── <run>_o3-mini.xlsx                       # Excel 汇总
├── rollout.log
└── judge.log
```

常用验收项：

```text
agent_input.jsonl 行数 = 目标题数
Step2 行数 = 成功导出的轨迹数
Step3 judged 行数 = 被判分轨迹数
unresolved = 0
Excel 文件非空且出现 Exported to Excel
```

技术失败题不应被当作有效回答；完整数据集准确率需要把技术失败计入分母。

## 9. 与 Native AgentLoop 的区别

Native 入口是：

```text
science-agent-next/scripts/run_hle_agentloop.py
```

Native 直接构造 `AgentLoop` 和可选 `DelegationManager`，并使用：

```text
science-agent-next/science_agent/agent_loop/
science-agent-next/science_agent/runtime/tool_scheduler.py
science-agent-next/science_agent/runtime/team.py
science-agent-next/science_agent/runtime/checkpoint.py
```

Legacy 路径则继续使用：

```text
liuzhuo/science-agent/inference/react_agent_openai_api_tools.py
```

因此，修改 Native 的 Team、checkpoint 或工具调度代码，不会自动改变 Legacy 行为；反过来也一样。做速度或准确率对比时，应固定模型地址、数据集、工具、图片输入开关、Judge 和并发，只改变框架路径。
