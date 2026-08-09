# Football DC 本地足球预测系统

基于 Dixon-Coles 进球模型的本地足球比赛预测工具。项目目标是做一个可持续补数据、可回测、可解释的单场预测工作台，而不是把所有赛事混在一起做一个黑盒概率。

当前支持：

- 男足世界杯：正赛 + 预选赛周期
- 女足世界杯：正赛 + 预选赛周期
- 欧冠：正赛 + 预选赛
- 英超
- 中超
- 手动上传比赛 CSV、赔率 CSV、球队强度 CSV
- 页面补录比赛数据，补录后刷新即可参与训练
- 90 分钟胜平负、比分、大小球、双方进球、半全场、让球、赔率价值判断

## 项目特点

- **按赛事单独训练**：世界杯、女足世界杯、欧冠、英超、中超不会混训。
- **Dixon-Coles 基础模型**：拟合球队进攻、防守、主场优势和低比分相关性。
- **90 分钟口径**：淘汰赛点球和晋级信息只做记录，不进入进球模型。
- **中文 UI**：球队选择和页面文案尽量中文化，CSV 内部仍保存英文标准名。
- **动态数据源**：启动或刷新时自动扫描 `data/` 下实际存在的 CSV。
- **赔率不污染模型**：赔率只用于去水概率、融合概率和 EV，不反向写入训练。
- **人工修正可控**：赛前修正只有点击确认后才应用到本场预期进球。
- **本地优先**：Streamlit 本地网页，不依赖付费 API。

## 快速开始

```bash
cd /Users/imyb/Documents/Football
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

如果已经安装过依赖，平时只需要：

```bash
cd /Users/imyb/Documents/Football
source .venv/bin/activate
streamlit run app.py
```

默认访问地址通常是：

```text
http://localhost:8501
```

## 页面功能

### 单场预测

在页面中选择：

- 数据源
- 赛事
- 主队 / 队伍 A
- 客队 / 队伍 B
- 预测日期
- 是否中立场
- 是否启用训练快速模式

输出包括：

- 预期进球
- 90 分钟胜平负概率
- 最可能比分
- 比分概率矩阵
- 总进球数分布
- 大小球 1.5 / 2.5 / 3.5
- 双方进球（BTTS）
- 半全场概率
- 亚洲让球
- 欧洲让球胜平负
- 赔率价值表

### 赔率与价值

支持手动输入十进制赔率，也支持上传赔率 CSV。

赔率 CSV 字段：

```csv
event_date,competition,home_team,away_team,market,selection,line,odds_decimal,bookmaker,captured_at,source
```

页面会计算：

- 隐含概率
- 去水概率（No-vig）
- 纯模型概率
- 赔率融合概率
- 纯模型 EV
- 融合 EV

说明：赔率变化不会改变 Dixon-Coles 基础概率，只会改变融合概率和 EV。

### 补录比赛数据

补录入口在侧边栏。

补录时可以选择目标 CSV，例如：

- 世界杯 2026 正赛
- 世界杯 2026 预选赛周期
- 女足世界杯 2023 正赛
- 女足世界杯 2023 预选赛周期
- 欧冠 2025-2026 正赛
- 欧冠 2025-2026 预选赛
- 中超 2026
- 英超 2025-2026

补录字段包括：

- 日期
- 主队、客队
- 90 分钟比分
- 中立场
- 小组赛 / 淘汰赛 / 预选赛等阶段
- 轮次
- 是否点球
- 晋级球队
- 备注

补录写入前会自动备份目标 CSV 到同目录 `backups/`，并检查重复比赛、同队对阵、负数进球等问题。

## 数据目录

当前推荐结构：

```text
data/
  worldcup/
    finals_2026.csv
    qualifiers_2026_cycle_all.csv
    finals_2022.csv
    qualifiers_2022_cycle_all.csv
    team_strength_ratings.csv

  women_worldcup/
    finals_2023.csv
    qualifiers_2023_cycle_all.csv
    qualifiers_2027_cycle_all.csv

  epl/
    epl_2025_2026.csv

  csl/
    csl_2025.csv
    csl_2026.csv

  champions_league/
    main_2025_2026.csv
    qualifiers_2025_2026.csv

  odds/
    sample_odds.csv
```

数据源不需要写死到页面里。只要文件放进对应目录，并符合命名规则，页面刷新后会自动扫描显示。

命名规则：

- 男足世界杯正赛：`data/worldcup/finals_YYYY.csv`
- 男足世界杯预选赛周期：`data/worldcup/qualifiers_YYYY_cycle_all.csv`
- 女足世界杯正赛：`data/women_worldcup/finals_YYYY.csv`
- 女足世界杯预选赛周期：`data/women_worldcup/qualifiers_YYYY_cycle_all.csv`
- 英超：`data/epl/epl_YYYY_YYYY.csv`
- 中超：`data/csl/csl_YYYY.csv`
- 欧冠正赛：`data/champions_league/main_YYYY_YYYY.csv`
- 欧冠预选赛：`data/champions_league/qualifiers_YYYY_YYYY.csv`

## 比赛 CSV 格式

最小字段：

```csv
competition,season,date,home_team,away_team,home_goals,away_goals,neutral_site
```

推荐完整字段：

```csv
competition,season,date,home_team,away_team,home_goals,away_goals,neutral_site,stage,round,score_basis,decided_by_penalties,winner,notes
```

示例：

```csv
competition,season,date,home_team,away_team,home_goals,away_goals,neutral_site,stage,round,score_basis,decided_by_penalties,winner,notes
WorldCup,2026,2026-06-11,Mexico,South Africa,2,0,true,Group,Group Stage,FT90,false,,
WomenWorldCup,2023,2023-08-20,Spain,England,1,0,true,Knockout,Final,FT90,false,Spain,
CSL,2026,2026-03-06,Chengdu Rongcheng,Shenzhen Peng City,5,1,false,League,Round 1,FT90,false,,
```

字段说明：

- `competition`：赛事代码，例如 `WorldCup`、`WorldCupQualifiers`、`WomenWorldCup`、`WomenWorldCupQualifiers`、`ChampionsLeague`、`ChampionsLeagueQualifiers`、`EPL`、`CSL`
- `season`：赛季或赛事年份
- `date`：比赛日期，建议 `YYYY-MM-DD`
- `home_team` / `away_team`：英文标准队名
- `home_goals` / `away_goals`：90 分钟比分
- `neutral_site`：是否中立场
- `stage`：阶段，例如 `Group`、`Knockout`、`Qualification`、`League`
- `round`：轮次，例如 `Group Stage`、`Round of 16`、`Final`
- `score_basis`：比分口径，训练只使用 `FT90`
- `decided_by_penalties`：是否点球决胜
- `winner`：晋级或获胜球队，仅记录展示
- `notes`：备注

### 点球、加时、退赛怎么处理

- 90 分钟比分写入 `home_goals / away_goals`
- 点球大战不写入进球字段
- 点球晋级写入 `decided_by_penalties=true`、`winner`、`notes`
- 加时后比分如果没有 90 分钟比分，应标记为非 `FT90`，避免进入 Dixon-Coles 训练
- 退赛、取消、判负等记录可以保留，但应使用非 `FT90` 口径，例如 `VOID`

## 当前内置数据

### 男足

- 世界杯 2026 正赛
- 世界杯 2026 预选赛周期
- 世界杯 2022 正赛
- 世界杯 2022 预选赛周期
- 英超 2025-2026 样例
- 中超 2025 / 2026

### 女足

- 女足世界杯 2023 正赛：64 场
- 女足世界杯 2023 预选赛周期：87 条记录，150 场训练样本中会排除 1 条非 FT90 记录
- 女足世界杯 2027 预选赛周期：231 场

女足 2023 预选赛当前纳入了 AFC、OFC、CONMEBOL、Concacaf、洲际附加赛的关键资格赛阶段。UEFA / CAF 2023 周期全量小组赛暂未强行补齐，后续可继续追加。

### 欧冠

- 欧冠 2025-2026 正赛：189 场
- 欧冠 2025-2026 预选赛：92 场

欧冠上一届数据包含联赛阶段、淘汰赛附加赛、16 强、1/4 决赛、半决赛、决赛，以及第一轮资格赛至附加赛。标记为 `AET` 的加时比分会保留展示，但默认不进入 Dixon-Coles 90 分钟训练。

## 模型说明

当前主页面使用 Dixon-Coles 作为基础模型。

训练参数包括：

- 每队进攻强度
- 每队防守强度
- 主场优势
- Dixon-Coles 低比分相关参数 `rho`

训练规则：

- 按 `competition` 分开训练
- `WorldCup` 会合并 `WorldCupQualifiers`
- `WomenWorldCup` 会合并 `WomenWorldCupQualifiers`
- `ChampionsLeague` 会合并 `ChampionsLeagueQualifiers`
- `EPL`、`CSL` 单独训练
- 只训练 `score_basis=FT90` 的比赛
- 预选赛默认低于正赛权重
- 淘汰赛可按阶段略微加权
- 可启用时间衰减，默认半衰期 365 天

重要限制：

- 当前输出是 90 分钟赛果概率，不是晋级概率
- 点球和晋级结果不进入进球模型
- 赔率不进入历史训练
- 伤病、天气、战意等只通过赛前人工修正近似
- xG/xGA 特征接口已预留，但没有真实 xG 数据时不会伪造

## 训练快速模式

当预选赛数据很多时，完整训练可能变慢或不收敛。

训练快速模式会：

- 世界杯 / 女足世界杯 / 欧冠：只保留当前双方相关的预选赛，同时保留正赛
- 英超 / 中超：数据过多时保留最近比赛

这个模式用于提高本地交互速度，不改变 CSV 原始数据。

## 数据质量检查

新增或补录 CSV 后建议运行：

```bash
.venv/bin/python scripts/data_quality_report.py
```

检查内容包括：

- 缺少字段
- 重复比赛
- 缺少比分
- 负数进球
- 主客队相同
- 非 `FT90` 比分口径
- `prediction_available_at` 是否晚于比赛日期

## 命令行预测

可以不用网页，直接跑脚本：

```bash
.venv/bin/python scripts/train_and_predict.py \
  --matches data/worldcup/finals_2026.csv \
  --home "Spain" \
  --away "Austria"
```

## 回测

运行 walk-forward 回测：

```bash
.venv/bin/python scripts/run_backtest.py data/worldcup/finals_2026.csv WorldCup 30
```

回测指标：

- Accuracy
- Log Loss
- Brier Score
- Ranked Probability Score（RPS）
- Calibration Error

结果会写入 `reports/`。

## 测试

```bash
.venv/bin/python -m unittest discover -s tests
```

当前测试覆盖：

- 数据导入与标准化
- catalog 自动扫描
- 中文队名映射
- Dixon-Coles 训练
- 市场概率和 EV
- 赛前人工修正
- 女足世界杯数据源
- 欧冠数据源
- 回测与特征层

## Git 使用

当前远程仓库：

```text
https://github.com/yangbo-f/Football_DC.git
```

常用提交流程：

```bash
git status --short
git add -A
git commit -m "Describe your change"
git push
```

`.gitignore` 已排除：

- `.venv/`
- Python 缓存
- 数据备份目录
- 本地 Streamlit 选择缓存
- 根目录原始 logo 副本

## 后续精度提升路线

详细路线见：

- [docs/model_accuracy_roadmap_zh.md](docs/model_accuracy_roadmap_zh.md)
- [docs/requirements_zh.md](docs/requirements_zh.md)

优先级建议：

1. 补齐历史数据，尤其是各赛事最近 3-5 年。
2. 做数据质量检查，统一队名和比分口径。
3. 用 walk-forward 回测决定时间衰减、预选赛权重、阶段权重。
4. 引入真实 xG/xGA、射门质量、休息天数、旅途距离。
5. 加入 FIFA/Elo/联赛强度作为跨赛区锚点。
6. 单独做晋级概率模块，不用 90 分钟胜平负替代。
7. 接入真实赔率源，但只用于市场对比和融合概率。
