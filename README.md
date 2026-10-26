# ExpAgent：读懂训练实验的工具调用 Agent

这是一个小型 LLM Agent，用来回答关于机器学习训练实验的问题，比如：

- "lego 场景里哪个实验的 LPIPS 最好？"
- "lego_high_lr 为什么没有最终结果？"
- "room_short 计划训练多少步？"

Agent 不会自己去猜数字，所有数值都通过工具从实验目录里读出来。同一套工具还有三种用法：作为 **MCP 服务**接入 Claude Code 等客户端，作为 **HTTP API** 供后端调用，以及在命令行里直接调用。项目自带**评测集**和 **43 个单元测试**，只用 Python 标准库，没有任何第三方依赖。

> 示例数据说明：`examples/runs/` 里的 5 个实验是 [`examples/make_sample_runs.py`](examples/make_sample_runs.py) 用固定随机种子生成的**模拟数据**，曲线形状仿照 3D Gaussian Splatting 的训练日志，**不是真实训练结果**。

## 架构

```
              ┌──────────── 入口 ────────────┐
  CLI (ask/call)    HTTP API (/ask, /tools)    MCP stdio 服务
        │                  │                        │
        ▼                  ▼                        │
   ┌───────────────── Agent 循环 ─────────┐         │
   │ 1. 把问题和工具 schema 发给 LLM        │         │
   │ 2. LLM 返回 tool_calls → 校验并执行    │         │
   │ 3. 结果（或错误）作为 tool 消息回传    │         │
   │ 4. 重复，直到 LLM 直接给出答案或到步数上限 │      │
   └──────────────────┬───────────────────┘         │
                      ▼                             ▼
              Toolbox：参数校验 + 6 个工具  ◄─────────┘
                      ▼
              RunStore：只读访问实验目录（白名单防路径穿越）
```

| 文件 | 作用 |
|---|---|
| [`expagent/tools.py`](expagent/tools.py) | 6 个工具、JSON Schema 参数校验、NaN 处理 |
| [`expagent/agent.py`](expagent/agent.py) | Agent 循环、错误回传、重复调用拦截、步数上限、调用轨迹 |
| [`expagent/llm.py`](expagent/llm.py) | OpenAI 兼容接口客户端（标准库 urllib 实现，带重试），以及测试用的 ScriptedLLM |
| [`expagent/mcp_server.py`](expagent/mcp_server.py) | 手写的 MCP 服务（JSON-RPC 2.0 over stdio） |
| [`expagent/server.py`](expagent/server.py) | HTTP JSON API |
| [`evals/`](evals/) | 评测集、评分逻辑、非 LLM 的关键词路由基线 |
| [`tests/`](tests/) | 单元测试 |

## 快速开始

需要 Python ≥ 3.10。项目没有第三方依赖，用虚拟环境只是为了和系统 Python 隔离：

```bash
git clone https://github.com/lfy20040505666/exp-agent.git
cd exp-agent
python3 -m venv .venv
source .venv/bin/activate
```

不需要 API key 就能做的事：

```bash
python -m expagent tools                                         # 查看工具及参数 schema
python -m expagent call diagnose_run '{"run": "lego_high_lr"}'   # 直接调用工具
python -m unittest discover -s tests                             # 跑单元测试
python -m evals.run_eval --llm baseline                          # 跑基线评测
```

`diagnose_run` 的实际输出：

```json
{
  "run": "lego_high_lr",
  "healthy": false,
  "issues": [
    "no final evaluation: eval.json is missing",
    "loss became nan at step 7000 (training diverged)",
    "loss spiked 5.9x at step 6500 (0.03857 -> 0.2273)",
    "training stopped early: last logged step 7500 of 30000"
  ],
  "last_logged_step": 7500,
  "planned_iterations": 30000
}
```

接入真实 LLM（任何 OpenAI 兼容接口都可以；默认是 DeepSeek）：

```bash
export EXPAGENT_API_KEY=你的key
export EXPAGENT_BASE_URL=https://api.deepseek.com   # 可选，换成其他兼容服务的地址
export EXPAGENT_MODEL=deepseek-chat                 # 可选
python -m expagent ask "lego 场景里哪个实验的 LPIPS 最好？"
```

运行时会把每一步工具调用打印到 stderr；加 `--json` 可以输出完整轨迹（每一步的参数、结果、耗时，以及 token 用量）。

用自己的实验目录：`--runs-dir /path/to/runs`，或者设置环境变量 `EXPAGENT_RUNS_DIR`。每个实验一个子文件夹，里面放 `config.json`（必需）、`metrics.jsonl`（每行一条日志）和 `eval.json`（最终指标，可以没有）。

## 工具

| 工具 | 作用 |
|---|---|
| `list_runs` | 列出所有实验，以及场景、计划步数、是否有最终结果；可按场景过滤 |
| `get_run_config` | 读取某个实验的训练配置 |
| `get_final_metrics` | 读取最终的 PSNR / SSIM / LPIPS；没有结果时明确返回 `no_eval`，而不是报错 |
| `compare_runs` | 按某个指标给多个实验排序（知道 LPIPS 越低越好），跳过没有结果的实验 |
| `get_metric_curve` | 训练曲线降采样，保留首尾两个点，并给出最后一个值 |
| `diagnose_run` | 检查缺少 eval、loss 出现 NaN/inf、loss 突增、提前停止、高斯数量异常 |

## MCP 服务

`python -m expagent mcp` 会启动一个 stdio MCP 服务，实现了 `initialize`（协议版本协商）、`ping`、`tools/list`、`tools/call`。工具出错时返回 `isError: true` 的结果，让模型能看到错误并重试；未知工具和未知方法返回 JSON-RPC 错误码。stdout 只输出协议消息，日志一律写到 stderr。

接入 Claude Code 的示例（把路径换成你本地的绝对路径）：

```bash
claude mcp add exp-agent -e PYTHONPATH=/abs/path/exp-agent -- /abs/path/exp-agent/.venv/bin/python -m expagent mcp
```

Claude Desktop 等使用 JSON 配置的客户端：

```json
{
  "mcpServers": {
    "exp-agent": {
      "command": "/abs/path/exp-agent/.venv/bin/python",
      "args": ["-m", "expagent", "--runs-dir", "/abs/path/to/runs", "mcp"],
      "env": {"PYTHONPATH": "/abs/path/exp-agent"}
    }
  }
}
```

## HTTP API

```bash
python -m expagent serve --port 8000
curl localhost:8000/health
curl localhost:8000/runs
curl -X POST localhost:8000/tools/compare_runs -d '{"runs": ["room_baseline", "room_short"], "metric": "ssim"}'
curl -X POST localhost:8000/ask -d '{"question": "lego_high_lr 为什么失败了？"}'
```

状态码：参数错误或请求体不是合法 JSON 返回 400；路由或工具不存在返回 404；请求体超过 1MB 返回 413；没有配置 API key 返回 503；LLM 调用失败返回 502。

## 评测

[`evals/cases.jsonl`](evals/cases.jsonl) 有 10 个用例，覆盖：单工具查询、需要多步的比较、失败诊断、对不存在的实验不编造结果、拼错实验名、英文提问。

每个用例从两方面打分：

- **工具**：期望的工具是否都调用了（可以写成"几个工具任选其一"）。
- **答案**：必须包含 / 至少包含其一 / 不能包含的字符串（不区分大小写）。

此外还会统计 LLM 调用次数、工具报错次数和耗时。

关键词路由基线（[`evals/baseline.py`](evals/baseline.py)）不是 LLM：它按关键词选一个工具，把工具的原始输出直接当作答案。实际结果如下：

```
pass 80% | tools 80% | answers 100% | avg LLM calls 2.0 | tool errors 1
❌ best_lpips_scene  只调用了 list_runs，没有做比较
❌ english_compare   只调用了 list_runs，没有做比较
```

基线的"答案"指标是 100%，这正好暴露了字符串匹配的局限：把整个工具输出原样输出，里面自然包含所有实验名。所以必须结合工具调用一起判断，后续可以改用 LLM 打分或者结构化答案。

评测真实模型：

```bash
python -m evals.run_eval --llm api --out evals/results/deepseek.json
```

## 设计中处理的几个问题

- **路径穿越**：实验名必须在 `list_runs` 列出的名单里，`../../etc` 这类输入会被拒绝。
- **NaN**：很多训练日志会直接写出 `NaN`，而 `NaN` 不是合法 JSON，有些 API 会因此拒收整个请求。工具输出里统一把它转成字符串 `"nan"`。
- **模型的参数写错了**：JSON 格式错误、缺少参数、参数类型不对、实验名拼错，都会作为错误信息回传给模型，让它自己改正，而不是让程序崩溃。
- **死循环**：用相同参数重复调用同一个工具会被拦截；达到步数上限后，会禁用工具、强制模型给出最终答案。
- **工具本身有 bug**：工具内部的异常会被捕获，变成错误信息回传给模型，不会中断整个对话。

## 局限

- 示例数据是模拟的，工具目前只认识类似 3DGS 的日志格式（`loss`、`num_gaussians`、PSNR/SSIM/LPIPS）。
- 评测集只有 10 条，答案靠字符串匹配，比较粗糙。
- MCP 只实现了工具相关的部分（没有 resources、prompts，也不支持 HTTP 传输）。
- 没有流式输出，也没有跨会话记忆。

## License

MIT
