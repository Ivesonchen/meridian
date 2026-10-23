# 将 Meridian Notebook 工作流转为 Python 项目

[English](README.md) | [简体中文](README.zh-CN.md)

这是一个用于学习的项目，而不是新的建模库。它将 **`demo/` 中全部 9 个
Notebook** 转换为具名的 Python 函数和可独立执行的命令行（CLI）阶段。
项目仍然调用相同的 Meridian API，但不再依赖 Jupyter、Colab、Notebook
魔法命令或隐藏的单元格状态。原始 Notebook 和 Meridian 库代码均未修改。

## 1. 应该在哪里运行

**如果你使用的是配有 NVIDIA GPU 的 Windows 电脑，建议继续使用 WSL Ubuntu。**
WSL 是 Linux 运行环境，并不要求你使用 Notebook。在 VS Code 的 WSL 终端中，
普通 `.py` 程序可以使用与之前 Notebook 相同的 Python 环境和 GPU。

| 运行环境 | 建议的执行方式 |
| --- | --- |
| WSL Ubuntu + NVIDIA CUDA | `--device cuda` |
| 原生 Linux + NVIDIA CUDA | `--device cuda` |
| 没有 GPU 的 Linux/WSL | `--device cpu`，也是安全的默认设置 |
| 原生 Windows 或 macOS | 如果能够安装全部 Meridian 依赖，可能可以使用 CPU；本项目尚未验证 |

本项目会明确拒绝在原生 Windows 下使用 `--device cuda`。要使用你已经配置好的
GPU 环境，请使用 WSL 终端，而不是 PowerShell 中的 Windows Python。
原生 Windows 的 CPU 支持情况取决于所安装的 Meridian、JAX 和 TensorFlow 版本。
你**不需要** Jupyter，也不需要保持 Notebook 内核运行。

### 使用现有的 WSL 环境

在仓库根目录执行：

```bash
source ~/.venvs/meridian/bin/activate
python -m pip install -e './examples/meridian_workflows[all,dev]'
cd examples/meridian_workflows
python -m meridian_workflows list
```

本项目依赖当前的 Meridian 2.1 API。如果你在开发这个本地仓库，应将 Meridian
本身以可编辑模式安装，而不是使用与当前源码不匹配的旧版 PyPI 包：

```bash
# 在仓库根目录执行；CUDA 依赖适用于 Linux/WSL。
python -m pip install -e '.[and-cuda,schema]'
```

如果只使用 CPU，将上面的安装选项改为 `.[schema]`。示例项目还提供以下可选依赖组：
`calibration`（GeoX）、`tracking`（MLflow）、`scenario`（Scenario Planner），
以及包含全部可选功能的 `all`。基础 Getting Started 工作流不需要这些额外依赖。
不要在 Python 脚本内部执行安装依赖的命令。

## 2. 从这里开始：一个工作流，多个独立进程

在本项目目录下执行以下命令。`outputs/` 已加入 Git 忽略规则。

```bash
# 检查输入数据和模型结构，不进行采样。
python -m meridian_workflows getting-started prepare --device cpu

# 对整个流程进行短时测试；得到的模型不能用于业务决策。
python -m meridian_workflows getting-started train --profile smoke --device cuda

# 启动新的 Python 进程，读取上一步保存的模型，不会重新采样。
python -m meridian_workflows getting-started report --device cpu
```

在浏览器中打开 `outputs/getting-started/health_card.html`。同时查看
`summary.html`、`rhat.html`、`model_fit.html` 和 `parameter_diagnostics.csv`。
HTML 图表使用常见的 Vega/Altair JavaScript CDN，因此即使计算在本地完成，
浏览器渲染图表时仍可能需要联网。

`report` 阶段还会生成 **`residual_diagnostics.html`** 和
**`residual_diagnostics.csv`**，补充 **Durbin–Watson** 和
**Jarque–Bera 统计量及其 p-value**。这些指标复用模型拟合图中的后验预测均值，
不需要额外采样。残差定义为“实际值减去预测值”，与拟合图使用相同的尺度：
提供了每单位 KPI 收入时使用收入尺度，否则使用 KPI 尺度。

程序会按时间顺序对每个地区单独计算。如果包含多个地区，还会增加一行全国汇总结果：
在每个时间点对各地区的残差求和，而不是将不同地区拼接成一条时间序列。
这里使用的是 **All Data（全部数据）**，包括可能存在的留出观测值，
并非分别计算训练集和测试集。对于常量、全零或过短的残差序列，
无法计算的指标会明确标记并说明原因，不会被误判为通过。

Durbin–Watson 接近 2 表示一阶自相关可能较弱，但本报告不采用统一的通过/失败阈值。
Jarque–Bera p-value 使用渐近卡方分布近似；短序列、拟合后的残差、异方差和
时间自相关都会限制其解释。这个 p-value **不是** Meridian 的贝叶斯后验预测
p-value。补充报告会显示 Meridian 的健康状态，但不会改变原有评分或收敛检查。
它不是对贝叶斯似然的直接检验，也不能替代已收敛的模型。

训练完成后，程序会在**生成任何可选报告之前**保存 `model.binpb`。
其中包含输入数据、模型配置和推断样本。`model.json` 则记录实际的采样设置、
版本、耗时以及 `smoke`/`full` 标签。关闭 VS Code 或退出 Python 不会删除这些文件。

除非显式传入 `--overwrite`，否则训练命令不会覆盖已有模型。
建议为不同实验使用不同的输出目录：

```bash
python -m meridian_workflows getting-started train \
  --profile full --device cuda --output-dir outputs/getting-started-full

python -m meridian_workflows getting-started report \
  --output-dir outputs/getting-started-full

# 只有在检查诊断结果、确认模型充分收敛之后，才进行预算优化：
python -m meridian_workflows getting-started optimize \
  --output-dir outputs/getting-started-full
```

`full` 表示采用 Notebook 规模的采样数量，**不代表模型一定已经收敛**。
分析和优化命令会检查非常量参数是否存在过高或无法计算的 R-hat，以及是否出现了
采样发散（divergences）。这些检查不能替代完整的模型健康检查、有效样本量检查、
采样轨迹图或模型验证。

`--allow-unconverged` 可以绕过这道检查，并输出明确警告，但它**仅用于学习或调试**。
通过这种方式生成的结果不能用于决定广告预算。

### 你将学习到的完整流程

```text
CSV
  -> InputData：哪些列对应 KPI、控制变量、媒体、花费、触达人数等？
  -> ModelSpec + priors：模型采用了哪些假设和先验？
  -> Meridian：将数据与模型假设组合起来
  -> sample_prior / sample_posterior：计算成本较高的贝叶斯采样
  -> model.binpb：将训练结果持久化，分离训练与后续分析
  -> reviewer / visualizer / summarizer：检查模型质量并解释结果
  -> BudgetOptimizer：在模型有效的前提下，提出预算分配方案
```

`prepare` 只构建并描述输入数据和模型。常规工作流中，只有 `train` 阶段执行后验采样。
`report`、`analyze`、`optimize` 和 `export` 都会读取已经保存的样本。
如果模型文件不存在，程序会报错，而不是悄悄重新训练。

## 3. 项目结构

```text
meridian_workflows/
  __main__.py                 python -m 的程序入口
  cli.py                     解析命令，并显式分派到对应示例
  config.py                  CPU/CUDA 环境配置与采样设置
  data.py                    通用 CSV 加载器和模型构建函数
  pipeline.py                训练、原子保存模型、报告生成和质量检查
  residual_diagnostics.py    各地区和全国汇总的 Durbin–Watson、Jarque–Bera 诊断
  full_funnel_analysis.py    独立的两阶段模型数学计算辅助模块
  demos/                     全部 Notebook 对应的入口函数都在这里
    getting_started.py
    getting_started_tf.py
    reach_frequency.py
    response_curves.py
    prior_calibration.py
    full_funnel.py
    scenario_planner.py
    mlflow_demo.py
    rf_simulation.py
tests/                       针对真实数据和模块集成的测试
```

建议先阅读 `demos/getting_started.py`，然后依次阅读 `data.py`、`config.py`
和 `pipeline.py`。最后再阅读全漏斗模型的数学实现：它是从 Notebook
提取到普通 Python 模块中的高级实验性扩展，并不是理解基础工作流的前提。

每个示例都提供 `run(options)` 函数。需要构建模型的示例还会提供
`build_model(...)`，方便你直接查看该示例的具体模型假设。
程序不会动态执行 Notebook，也不会通过 `exec` 执行单元格文本，
运行时不依赖 Notebook 文件。

### 在自己的 `.py` 文件中调用示例

```python
from pathlib import Path
from meridian_workflows.config import Options, configure_runtime

configure_runtime(backend="jax", device="cpu")

# 必须先配置运行环境，再导入 Meridian 或依赖它的示例模块。
from meridian_workflows.demos import getting_started

getting_started.run(
    Options(output=Path("outputs/my-example"), action="prepare")
)
```

如果要切换计算后端或设备，请启动新的 Python 进程。
`run()` 是一个普通函数，不是依赖隐藏内核状态的 Notebook 单元格。

## 4. Notebook 与代码的对应关系及各个示例

| 原始 Notebook | CLI 示例名 / Python 模块 | 主要学习内容 |
| --- | --- | --- |
| [Getting Started](../../demo/Meridian_Getting_Started.ipynb) | `getting-started` / `getting_started.py` | 数据、先验、自动节点选择（AKS）、采样、健康检查、报告、优化和假设场景数据 |
| [Getting Started TF](../../demo/Meridian_Getting_Started_Tf.ipynb) | `getting-started-tf` / `getting_started_tf.py` | 使用旧版 TensorFlow 后端执行相同流程，并进行建模前探索性数据分析（EDA） |
| [RF Demo](../../demo/Meridian_RF_Demo.ipynb) | `reach-frequency` / `reach_frequency.py` | 展示次数与触达人数/频次的区别，以及 RF 先验 |
| [ROI/mROI](../../demo/ROI_mROI_Response_Curves.ipynb) | `response-curves` / `response_curves.py` | 可信区间、不确定性、盈利概率和响应曲线 |
| [Prior Calibration](../../demo/Meridian_Prior_Calibration_JAX.ipynb) | `prior-calibration` / `prior_calibration.py` | GeoX、通用增量实验，以及合并多种证据校准先验 |
| [Full Funnel](../../demo/Meridian_Full_Funnel.ipynb) | `full-funnel` / `full_funnel.py` | 品牌广告 -> 品牌资产 -> 转化 |
| [Scenario Planner](../../demo/Meridian_Scenario_Planner_Beta.ipynb) | `scenario-planner` / `scenario_planner.py` | 优化网格、本地数据表，以及可选的 Sheets/Looker 集成 |
| [MLflow](../../demo/Meridian_MLflow_Demo.ipynb) | `mlflow` / `mlflow_demo.py` | 可复现的本地实验跟踪 |
| [RF Simulation](../../demo/RF_Data_Simulation_for_Meridian.ipynb) | `rf-simulation` / `rf_simulation.py` | 生成具有已知真实效果和 ROI 的模拟数据 |

单模型示例共用 `prepare`、`train`、`report` 和 `optimize` 操作。
运行 `python -m meridian_workflows list` 可以查看完整操作列表。
下面的命令用于演示各项功能，**不是要求一次性全部执行的脚本**。

### Getting Started：假设预算场景

```bash
python -m meridian_workflows getting-started hypothetical \
  --output-dir outputs/getting-started-full
```

该操作使用 Notebook 中的假设场景 CSV、5,000 万预算和渠道分配比例。
程序会显式对齐地区与日期；如果地区/日期记录缺失或重复，会报错，
而不是悄悄用零填充。

### TensorFlow 版本

```bash
python -m meridian_workflows getting-started-tf prepare
python -m meridian_workflows getting-started-tf eda --prior 50
python -m meridian_workflows getting-started-tf train --device cuda --profile smoke
```

Meridian 的 TensorFlow 后端已被弃用。保留这个入口是为了帮助理解对应的
Notebook，而不是建议新项目优先使用 TensorFlow 而非 JAX。

### 触达人数与频次（Reach & Frequency）

```bash
python -m meridian_workflows reach-frequency prepare
python -m meridian_workflows reach-frequency train --device cuda --profile smoke
python -m meridian_workflows reach-frequency report
```

这个示例包含 3 个基于展示次数的渠道和 1 个 RF 渠道。
可以对比数据构建器的 `with_media()` 与 `with_reach()` 调用，
并查看 RF 渠道的 ROI 先验设置。

### ROI、边际 ROI 与响应曲线

```bash
# 复用已经拟合的模型，不再训练另一个模型。
python -m meridian_workflows response-curves analyze \
  --model outputs/getting-started-full/model.binpb
```

输出包括 `roi_credible_intervals.csv`、`mroi_credible_intervals.csv`、
`posterior_uncertainty.csv` 和 `response_curves.html`。
如果希望使用与该 Notebook 完全一致的输入结构，请执行它自己的 `prepare`/`train`
操作。与 Getting Started 不同，这个示例不包含自然媒体和促销输入，并使用默认先验。

### 先验校准

```bash
python -m meridian_workflows prior-calibration calibrate
```

Channel0 使用 GeoX 结果；Channel1 使用通用实验结果；Channel2 合并两种证据；
Channel3 和 Channel4 保持未校准状态。
Notebook 中的实验是**合成演示数据**，不是关于你实际业务的真实证据。
该命令会保存先验分布直方图和先验样本，不执行后验采样。
可选的 `train` 操作则会使用这些校准后的先验拟合模型。

### 全漏斗模型

```bash
python -m meridian_workflows full-funnel prepare
python -m meridian_workflows full-funnel train-stage1 --device cuda --profile smoke
python -m meridian_workflows full-funnel train-stage2 --device cuda --profile smoke
python -m meridian_workflows full-funnel report
# 获得质量合格的拟合结果后，再进行分析和优化：
python -m meridian_workflows full-funnel analyze
python -m meridian_workflows full-funnel optimize
```

阶段 1 通过品牌认知广告预测品牌搜索量。阶段 2 预测转化，并将品牌搜索量作为
不施加饱和变换的自然媒体中介变量。
每个阶段都单独保存模型，因此阶段 2 失败不会丢失阶段 1 的结果。
两个阶段应使用相同的链数和样本数，但采用独立的随机种子。
`train` 可以连续运行两个阶段，不过分阶段执行更便于学习。

独立的辅助模块保留了 Notebook 中考虑中介效应的分析器、
间接/剩余/总 ROI 分解，以及复合广告滞后效应图。
它依赖当前仓库版本的内部分析 API，因此升级 Meridian 时应同步更新并测试。
原 Notebook 中被注释掉、尚未完成的全漏斗云端上传草稿，
没有被当作可执行工作流。已支持的 Scenario Planner 功能由下面的独立入口提供。

### 场景规划器（Scenario Planner）

```bash
# 使用保存的样本在本地计算，不登录 Google，也不上传。
python -m meridian_workflows scenario-planner export \
  --model outputs/getting-started-full/model.binpb

# 可选：增加按季度划分的结果，计算成本更高。
python -m meridian_workflows scenario-planner export \
  --model outputs/getting-started-full/model.binpb --quarterly
```

该操作会生成 `scenario.binpb`，并在 `outputs/scenario-planner/tables`
下保存 CSV 文件。如果希望使用原 Notebook 的模型配置，请运行这个示例自己的
`prepare`/`train`，而不是复用另一个模型。

上传是一个独立且需要显式执行的操作：

```bash
python -m meridian_workflows scenario-planner upload \
  --spreadsheet-name "My Meridian demo"
```

上传需要 Google 应用默认凭据（Application Default Credentials，ADC），
凭据必须包含 Sheets 和 Drive 两种访问范围，并具备相应的 API 和项目权限。
请在项目之外配置凭据，不要将凭据提交到仓库。
上传操作会创建一个 Google 电子表格，并将其 URL 和 Scenario Planner 链接
写入文本文件。未经授权，不要上传私有数据。测试中的上传操作使用模拟实现，
不会真正上传。

### MLflow

```bash
python -m meridian_workflows mlflow train --device cuda --profile smoke
python -m meridian_workflows mlflow runs
mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db
```

该示例启用 Meridian 的自动日志记录，记录采样配置标签，并保存模型产物。
实验跟踪在本地进行，使用 SQLite 数据库；`runs` 操作不会训练模型。
启动 UI 的命令是可选的，运行后可以按 Ctrl+C 停止。

### RF 数据模拟

```bash
python -m meridian_workflows rf-simulation simulate --seed 1320

# 生成的数据结构可以直接用于 RF 示例。
python -m meridian_workflows reach-frequency prepare \
  --data outputs/rf-simulation/geo_rf_data.csv
```

输出包括地区级和全国级 CSV、`ground_truth.npz`、`raw_parameters.npz`，
以及时间效应、KPI 和花费的诊断 PNG 图。
这是数据生成过程，不是 MCMC 采样。
模拟器保留了 Notebook 中的数据生成方程，同时使用显式指定种子的随机数生成器，
以便兼容 JAX 和 TensorFlow；它不保证生成与原 TensorFlow 随机种子完全相同的样本。

这里有两项有意的修正：模拟的触达人数受到物理上合理的上下界约束；
全国级的每次转化收入按转化量加权，以保证汇总前后总收入一致。
全国级频次按触达人数加权。真实参数值以数值 NPZ 数组保存，
而不是使用 Python pickle。

## 5. 采样、性能与科学有效性检查

默认的 `smoke` 配置使用 2 条链、200 次适应迭代、100 次 burn-in、
每条链保留 200 个后验样本，以及 200 个先验样本。
它仅用于检查代码和流程是否能够正常连接并运行。

`full` 配置保留了 Notebook 规模的总链数和样本数。
为了降低 12 GB GPU 的显存峰值，链会被划分为每批最多 2 条，
而不是一次启动全部链。这可能增加实际运行时间，也可能需要更多适应迭代。
RF、响应曲线、TensorFlow 和 Scenario Planner 示例使用 2,000 次适应迭代；
Getting Started 使用 1,000 次。
全漏斗示例的每个阶段使用 4 条链，每条链保留 500 个样本；
MLflow 示例使用 7 条链。

你可以显式覆盖这些参数，例如：

```bash
python -m meridian_workflows getting-started train --device cuda \
  --chains 2,2 --adapt 1000 --burnin 500 --keep 1000 --prior 500 \
  --output-dir outputs/diagnostic-candidate
```

`--chains 4` 表示一批运行 4 条链；`--chains 2,2` 表示分两批运行，
总共仍然是 4 条链。没有任何一组参数能够保证收敛。
除非明确指定 `--profile full`，否则配置标签仍然是 `smoke`。

默认使用 CPU，以免意外启动较大的 GPU 任务。
如果请求 CUDA 但设备不可用，程序会明确报错，不会悄悄回退到 CPU。
默认关闭 JAX 显存预分配，保留项目的默认计算精度，并将编译缓存放在
用户主目录下的私有路径中。缓存减少的是重复编译，不是 MCMC 本身的计算量。

TensorFlow、ArviZ 和 Altair 仍可能输出上游库的警告。
特别是，ArviZ 关于自定义 `trace` 组的警告本身并不意味着拟合失败。
不要全局屏蔽警告，也不要把成功生成文件理解为模型已经良好收敛。

## 6. 测试

```bash
python -m pytest -q
python -m pyink --check meridian_workflows tests
```

测试强制使用 CPU。覆盖内容包括全部示例入口、真实示例数据加载、
CLI 参数校验、小规模真实后验采样与保存/加载/报告生成的完整流程、
响应曲线、校准后先验的持久化、全漏斗代数关系、RF 模拟中应保持的约束关系，
以及本地 MLflow 集成；还会检查残差统计量的数值、日期排序、地区汇总方式，
以及无法计算指标的特殊情况。云端上传使用模拟实现。
小规模采样模型和仅使用先验样本的全漏斗测试数据，
用于验证代码功能，不代表科学有效性验证或生产级模型拟合。

这些示例源于仓库中采用 Apache-2.0 许可证的 Notebook；
详见仓库的 [LICENSE](../../LICENSE)。全漏斗辅助函数保留了原始版权声明。
