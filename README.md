# ArgSpaceFuzz-TF: TensorFlow Deep Learning Library Fuzzing Framework

基于 **参数空间表示（Parameter Space Representation, PSR）+ LLM Harness 生成 + 双通道 UCB 筛选执行** 的 TensorFlow API 覆盖率导向模糊测试框架。

本仓库是 ArgSpaceFuzz 在 **TensorFlow** 框架上的实现版本，主要用于 `tf.raw_ops.*`、`tf.nn.*`、`tf.math.*`、`tf.image.*`、`tf.signal.*` 等 TensorFlow API 的参数空间建模、Fuzz Harness 生成、覆盖率收集和潜在 crash 挖掘。

> 通用方法原理、整体算法思想、双通道 UCB 调度策略、Reward 计算方式、Profile 池演进机制等框架无关内容，请参考 PyTorch 版仓库 README：  
> https://github.com/yyds1233/Dl_fuzzer/blob/main/README.md  
>
> 本 README 只重点说明 TensorFlow 版本的目录结构、阶段命名、脚本入口、文件命名和运行方式差异。

---

## 项目概述

TensorFlow 版 ArgSpaceFuzz 的整体流程与 PyTorch 版保持一致：

1. 从 TensorFlow API 列表、运行时 schema、API 文档中提取参数信息；
2. 构建 TensorFlow API 的结构化参数空间 YAML；
3. 基于 YAML、API 文档和 Atheris 说明，用 LLM 生成可执行 Fuzz Harness；
4. 使用 Screen 模块对大量 harness 进行双通道 UCB 筛选执行；
5. 结合快速反馈和慢速覆盖率审计，将 fuzzing 预算集中到高收益 harness 上。

与 PyTorch 版不同的是，本仓库针对 TensorFlow API 的调用方式、异常体系、raw op / high-level API 映射关系、dtype/shape 约束形式进行了适配。

---

## 架构总览

```text
TensorFlow API 列表
      │
      ▼
┌──────────────────────────────┐
│ build_yaml                    │
│ TensorFlow schema / docs      │
│ → 参数空间 YAML               │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ fuzz_api_one                  │
│ YAML + API txt + Atheris doc  │
│ → LLM 生成 TF Harness (.py)   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ screen                        │
│ 双通道 UCB 筛选执行            │
│ coverage / corpus / crash     │
└──────────────────────────────┘
```

通用的 PSR、LLM 生成、Screen 调度、Reward 设计、Profile 池刷新和软淘汰逻辑与 PyTorch 版一致。

---

## 目录结构

```text
.
├── build_yaml/                         # 模块一：TensorFlow YAML 参数空间生成
│   ├── pipeline.py                      # TF API → YAML 的统一 pipeline 入口
│   ├── export_tf_schema.py              # 导出 TensorFlow API schema
│   ├── export_tf_schema_unified.py      # 统一 schema 导出，支持 raw_ops 与 high-level API
│   ├── extract_tf_doc_hints.py          # 从 TensorFlow 文档中提取参数提示
│   ├── llm_doc_rank_extractor.py        # LLM 辅助提取 rank / shape hints
│   ├── tf_schema2yaml.py                # TensorFlow schema → YAML skeleton
│   ├── tf_schema2yaml_unified.py        # 统一 YAML skeleton 生成
│   ├── normalize_yaml_skeleton.py       # YAML 结构规范化
│   ├── param_role_enricher.py           # 参数角色补全
│   ├── param_role_enricher_unified.py   # 统一参数角色补全
│   ├── llm_patch_yaml_new.py            # LLM 补全 shape / rank / dtype 信息
│   ├── tf_patch_constraints.py          # LLM 补全 TensorFlow 约束
│   ├── tf_materialize_yaml.py           # 最终 YAML 物化
│   ├── tf_api_resolver.py               # high-level API 到 raw op 的辅助解析
│   ├── tf_highlevel_param_classifier.py # high-level API 参数分类
│   ├── tf_llm_prompts.py                # TensorFlow YAML 构建相关 prompt
│   ├── tf_schema_common.py              # schema 公共结构
│   ├── tf_schema_common_ext.py          # schema 扩展结构
│   ├── tf_raw_ops.txt                   # TensorFlow raw_ops 列表
│   └── api_list*.txt                    # TensorFlow API 列表样例
│
├── fuzz_api_one/                        # 模块二：TensorFlow Harness 生成
│   ├── llm_gen_harness.py               # LLM 生成 TensorFlow Atheris Harness
│   ├── gen_harness_unified.py           # 统一模板 / 辅助式 Harness 生成
│   ├── pepiline.py                      # 批量生成 pipeline
│   └── batch_gen_harness_summary.json   # 批量生成结果摘要
│
├── screen/                              # 模块三：Harness 筛选与调度执行
│   ├── cli/main.py                      # Screen 命令行入口
│   ├── bandit_audit_driver_hier.py      # 双层 UCB 主调度逻辑
│   ├── cov_global_union_audit.py        # 全局 union 覆盖率审计
│   ├── auto_harnesses.json              # harness 清单
│   ├── auto_harness_experiment.json     # 实验 harness 清单
│   ├── groups_map.json                  # harness → group 映射
│   ├── bandit/                          # UCB 策略、reward 计算
│   ├── pool/                            # profile 池管理
│   ├── prior/                           # 跨 harness 精英 profile 记忆
│   ├── runner/                          # fuzz / audit 执行器
│   ├── metrics/                         # coverage / libFuzzer 日志解析
│   └── config/                          # Screen 配置
│
├── atheris-doc/                         # Atheris 使用说明
│   └── atheris_readme.txt
│
├── fuzz_output/                         # 生成或运行中的 harness 输出目录
├── run_screen.sh                        # Screen 模块启动脚本
├── yaml2harness.json                    # YAML → harness 生成清单
└── atheris-doc/                         # Atheris 框架使用文档
```

---

## 与 PyTorch 版本的主要差异

| 项目 | PyTorch 版 | TensorFlow 版 |
|------|------------|---------------|
| 目标框架 | `torch.*` / `torch.nn.functional.*` | `tf.*` / `tf.raw_ops.*` / `tf.nn.*` / `tf.math.*` |
| YAML 构建入口 | `build_yaml/pipeline.py` | `build_yaml/pipeline.py`，但内部阶段为 TensorFlow 专用 A–E |
| schema 导出 | PyTorch runtime / aten schema | `export_tf_schema_unified.py` |
| high-level API 解析 | 主要围绕 PyTorch API / overload | 支持 high-level API 与 raw op 解析 |
| YAML skeleton | `schema2yaml.py` | `tf_schema2yaml_unified.py` |
| 参数角色补全 | PyTorch 参数语义 | TensorFlow 参数角色、attr、dtype、shape 语义 |
| LLM 约束补全 | PyTorch API 文档语义 | TensorFlow API 文档语义与 `tf.errors.*` 异常体系 |
| Harness 文件名 | `llm.torch.xxx.py` | `llm.tf.xxx.py` |
| 预期异常 | `RuntimeError`、`TypeError` 等 PyTorch/Python 异常 | `tf.errors.InvalidArgumentError`、`tf.errors.UnimplementedError`、`tf.errors.ResourceExhaustedError` 等 TensorFlow 异常 |

---

## 模块一：TensorFlow YAML 参数空间生成（`build_yaml`）

`build_yaml` 将 TensorFlow API 列表转换为结构化 YAML 参数空间表示。TensorFlow 版 pipeline 使用 A–E 阶段组织：

| Stage | 名称 | 主要功能 | 典型输出 |
|-------|------|----------|----------|
| A | schema / resolve / doc | 解析 API，导出统一 schema，提取文档 rank hints | `01_schema/`, `02_rank_hints/` |
| B | skeleton / normalize / enrich | 生成 YAML skeleton，规范化并补全参数角色 | `03_skeleton/`, `04_normalized/`, `05_enriched/` |
| C | shapes | LLM 补全 shape / rank / dtype 等结构信息 | `06_shaped/` |
| D | constraints | LLM 补全 TensorFlow API 约束 | `07_constrained/` |
| E | material | 物化最终 YAML | `08_final/` |

### API 列表格式

`api_list.txt` 中每行一个 TensorFlow API，例如：

```text
tf.raw_ops.Conv2D
tf.raw_ops.MatMul
tf.nn.conv2d
tf.nn.relu
tf.nn.softmax
tf.math.reduce_sum
tf.math.matmul
tf.reshape
tf.transpose
tf.concat
tf.gather
tf.image.resize
```

### 运行方式

建议在 `build_yaml/` 目录下运行，因为 pipeline 内部会以当前目录为基准调用同目录脚本：

```bash
cd build_yaml

python pipeline.py \
  --api_list api_list.txt \
  --doc_dir ../api_txt \
  --work_dir ../yaml_output_tf \
  --stages A,B,C,D,E \
  --llm_base_url "$OPENAI_BASE_URL" \
  --llm_model gpt-5-codex \
  --openai_base_url "$OPENAI_BASE_URL" \
  --openai_model gpt-5-codex
```

如果只希望运行不依赖 LLM 的启发式阶段，可以先执行：

```bash
cd build_yaml

python pipeline.py \
  --api_list api_list.txt \
  --work_dir ../yaml_output_tf \
  --stages A,B \
  --no_llm
```

最终 YAML 通常位于：

```text
yaml_output_tf/08_final/
```

---

## 模块二：TensorFlow Harness 生成（`fuzz_api_one`）

`fuzz_api_one` 根据 TensorFlow YAML、API 文档和 Atheris 说明，调用 LLM 生成可执行的 Atheris Fuzz Harness。

### 单个 API 生成

```bash
python fuzz_api_one/llm_gen_harness.py \
  --yaml yaml_output_tf/08_final/tf_math_reduce_sum.yaml \
  --api-txt api_txt/tf.math.reduce_sum.txt \
  --atheris-doc atheris-doc/atheris_readme.txt \
  --out fuzz_output/llm.tf.math.reduce_sum.py \
  --model gpt-5.4
```

说明：

- TensorFlow 版 `llm_gen_harness.py` 会从 YAML 中读取 API 信息；
- 输出 harness 通常命名为 `llm.tf.xxx.py`；
- 生成的 harness 会导入 `atheris`、`tensorflow as tf`，并定义 `TestOneInput(data: bytes)`；
- 生成逻辑偏向合法输入，但保留 dtype、rank、shape、attr、optional 参数和边界值多样性；
- 正常的 TensorFlow 输入校验异常应被吞掉，非预期崩溃应保留用于复现。

### 批量生成

如果已经准备好 `yaml2harness.json`，可批量生成：

```bash
python fuzz_api_one/pepiline.py \
  --json yaml2harness.json \
  --out-dir fuzz_output/ \
  --workers 2 \
  --skip-existing
```

`yaml2harness.json` 建议组织为类似形式：

```json
[
  {
    "api": "tf.math.reduce_sum",
    "yaml": ["yaml_output_tf/08_final/tf_math_reduce_sum.yaml"],
    "txt": "api_txt/tf.math.reduce_sum.txt"
  }
]
```

具体字段以当前脚本实际读取逻辑为准。

---

## 模块三：Screen 筛选执行（`screen`）

TensorFlow 版 Screen 模块沿用 PyTorch 版的双层 UCB 调度思想：

- harness 级 UCB：选择本轮执行哪个 TensorFlow harness；
- profile 级 UCB：为当前 harness 选择更优 fuzzing profile；
- 快速通道：基于 libFuzzer / Atheris 日志中的 coverage、feature、速度和新增 corpus；
- 慢速通道：基于全局 union 覆盖率审计；
- 软淘汰：低收益 harness 冷却一段时间后可重新参与调度；
- Profile 池：保留精英、淘汰低效 profile、变异生成新 profile；
- GroupPrior：同类 API 或同组 harness 之间共享有效 profile 经验。

### 启动方式

使用脚本启动：

```bash
bash run_screen.sh 24h
```

或直接调用 Screen CLI：

```bash
python3 -m screen.cli.main \
  --harnesses_json screen/auto_harnesses.json \
  --groups_map screen/groups_map.json \
  --root fuzz_output/ \
  --epoch 30 \
  --steps 0 \
  --audit_every 3 \
  --cov_audit_script screen/cov_global_union_audit.py
```

常用清单文件：

```text
screen/auto_harnesses.json
screen/auto_harness_experiment.json
screen/groups_map.json
```

---

## 环境要求

### 基础环境

```bash
python --version   # 建议 Python 3.10+
```

Python 依赖：

```bash
pip install tensorflow atheris pyyaml openai
```

如果需要覆盖率审计，还需要 LLVM 覆盖率工具链：

```bash
llvm-profdata --version
llvm-cov --version
```

普通 `pip install tensorflow` 可以用于 harness 可执行性验证和 crash 探测；如果要做 source-based coverage，需要使用带覆盖率插桩的 TensorFlow 构建版本，并正确配置 `LLVM_PROFILE_FILE`、`llvm-profdata`、`llvm-cov` 等工具链。

### LLM 配置

推荐通过环境变量配置 OpenAI 兼容接口：

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
```


---


## 关键文件说明

| 文件 / 目录 | 用途 |
|-------------|------|
| `build_yaml/pipeline.py` | TensorFlow API → YAML 参数空间的统一 pipeline |
| `build_yaml/export_tf_schema_unified.py` | 导出 TensorFlow raw_ops / high-level API schema |
| `build_yaml/tf_schema2yaml_unified.py` | schema → YAML skeleton |
| `build_yaml/param_role_enricher_unified.py` | 参数角色和语义补全 |
| `build_yaml/llm_patch_yaml_new.py` | LLM 补全 shape / dtype / rank 信息 |
| `build_yaml/tf_patch_constraints.py` | LLM 补全 TensorFlow 约束 |
| `build_yaml/tf_materialize_yaml.py` | 生成最终 YAML |
| `fuzz_api_one/llm_gen_harness.py` | 单个 TensorFlow harness 的 LLM 生成入口 |
| `fuzz_api_one/pepiline.py` | 批量 harness 生成入口 |
| `screen/auto_harnesses.json` | Screen 使用的 harness 清单 |
| `screen/groups_map.json` | harness 到 group 的映射 |
| `screen/cov_global_union_audit.py` | 全局覆盖率审计 |
| `run_screen.sh` | Screen 执行脚本 |
| `dl_fuzzer_crash/` | crash / PoC 示例 |
| `fuzz_output_experiment/` | 实验 harness 示例 |

---

## 常见问题

### 1. 为什么 TensorFlow 版 pipeline 是 A–E，而 PyTorch 版是 0–6？

TensorFlow 版需要同时处理 `tf.raw_ops.*` 和 high-level API，因此将 resolver、schema、doc hints、skeleton、enrich、LLM shape patch、constraint patch 和 materialization 重新组织为 A–E 阶段, 便于区分。

### 2. 普通 pip 安装的 TensorFlow 可以用吗？

可以用于验证 harness 可执行性、dry-run 和 crash 探测；如果要统计 TensorFlow 源码级覆盖率，则需要覆盖率插桩构建的 TensorFlow 版本以及 LLVM coverage 工具链。

