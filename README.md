# Football Dixon-Coles Predictor

这是一个用于足球比分预测的 Dixon-Coles 本地小工具。第一版重点支持英超和世界杯，并且按赛事分开训练模型：英超使用俱乐部联赛历史赛果，世界杯使用国家队赛果和中立场逻辑，不把不同比赛体系硬混成一个模型。

## 当前模型范围

- 输入历史比赛：`competition, season, date, home_team, away_team, home_goals, away_goals, neutral_site`
- 数据层会统一补齐：`stage, round, score_basis, decided_by_penalties, winner, notes, source_file, match_importance, prediction_available_at`
- 拟合参数：
  - 球队进攻强度
  - 球队防守强度
  - 主场优势
  - Dixon-Coles 低比分相关参数 `rho`
- 支持时间衰减：越新的比赛权重越高
- 输出预测：
  - 比分概率矩阵
  - 主胜 / 平局 / 客胜
  - 大小球概率，例如 `Over 2.5`
  - BTTS 双方进球概率
  - 亚洲让球 / 欧洲让球概率
  - 十进制赔率隐含概率、去水概率、概率差和 EV

## 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 跑样例

```bash
python scripts/train_and_predict.py \
  --matches data/sample_matches.csv \
  --home "Arsenal" \
  --away "Chelsea"
```

## 启动本地网页

```bash
streamlit run app.py
```

网页里可以：

- 按世界杯、英超、中超目录加载数据
- 上传自己的比赛 CSV
- 上传赔率 CSV，或手动输入赔率
- 选择赛事后只训练该赛事模型
- 使用中文队名选择球队，模型内部仍保留英文标准名
- 手动选择或输入球队并输入赔率
- 查看胜平负、比分、半全场、进球数、大小球、BTTS、让球和价值判断
- 对比纯模型概率和赔率融合概率

默认数据源是 `data/worldcup/finals_2026.csv`。模型的基础数据完全来自你加载的 CSV，训练时按 `competition` 分开，且只使用 `score_basis=FT90` 的比赛。

## 数据质量检查

每次补录或新增 CSV 后，建议先运行：

```bash
.venv/bin/python scripts/data_quality_report.py
```

检查内容包括：

- 是否缺少统一 schema 字段
- 是否有重复比赛
- 是否缺少比分
- 是否有负数进球
- 主队和客队是否相同
- 是否存在非 `FT90` 比分口径
- `prediction_available_at` 是否晚于比赛日期

`match_importance` 是后续回测调权的基础字段。当前只生成初始规则，不直接替换现有 Dixon-Coles 训练权重。

## 赛前特征层

项目已提供严格按时间顺序生成的赛前特征：

```python
from football_dc.data import load_matches
from football_dc.features import build_pre_match_features

matches = load_matches("data/worldcup/finals_2026.csv")
features = build_pre_match_features(matches)
```

当前会生成：

- 赛前 Elo：`home_elo, away_elo, elo_diff`
- Elo 趋势：`elo_30d_change, elo_90d_change, elo_180d_change, elo_peak_1y, elo_std_1y`
- 近期状态：`form_3, form_5, form_10`
- 对手强度修正状态：`weighted_form_5, weighted_form_10`
- 对手强度：`avg_opponent_elo_5, avg_opponent_elo_10, strength_of_schedule`
- xG 预留滚动特征：`rolling_xG_3/5/10, rolling_xGA_3/5/10, rolling_xGD_5/10`

这些特征只使用每场比赛之前已经发生的数据。当前 CSV 没有真实 xG 时，xG 特征保持缺失，不会用比分或射门数据伪造 xG。

## 模型层与回测

当前保留 Dixon-Coles 作为 Baseline，并新增轻量模型接口：

- `EloModel`：基于赛前 Elo 输出胜平负概率
- `LogisticBaselineModel`：基于 Elo、近期状态、对手强度等特征输出胜平负概率
- `MarketModel`：将赔率去水后转成市场概率
- `Ensemble`：按配置权重融合 Dixon-Coles、Elo、Logistic 和 Market

这些接口不会自动替换当前页面的基础预测，后续应通过回测决定默认权重。

可运行 Baseline walk-forward 回测：

```bash
.venv/bin/python scripts/run_backtest.py data/worldcup/finals_2026.csv WorldCup 30
```

回测指标包括：

- Accuracy
- Log Loss
- Brier Score
- Ranked Probability Score（RPS）
- Calibration Error

回测结果会写入 `reports/`。当前校准层提供 Temperature Scaling、Platt-style Scaling 和 Isotonic 接口，其中 Isotonic 先保留兼容入口，等样本量足够后再做真实分桶拟合。

## 赛前预测层

页面会明确展示：

- 当前预测口径：90 分钟常规时间，不等于晋级概率
- 模型版本：Dixon-Coles Baseline，Elo/xG 特征层已就绪
- 风险评级：低风险 / 中风险 / 高风险
- 数据完整度、模型置信度、市场分歧
- 哪些信息进入训练，哪些只用于赔率比较或人工修正

人工备注不会直接改变预测；只有确认后的数值修正会改变本场预期进球。

## 推荐数据目录

```text
data/
  worldcup/
    finals_2026.csv
    qualifiers_2026.csv
    finals_2022.csv
    finals_2018.csv
  epl/
    epl_2025_2026.csv
  csl/
    csl_2026.csv
  odds/
    sample_odds.csv
```

赔率 CSV 字段：

```csv
event_date,competition,home_team,away_team,market,selection,line,odds_decimal,bookmaker,captured_at,source
```

赔率不会改变基础 Dixon-Coles 模型；它只用于去水市场概率、EV 和“赔率融合概率”。融合权重可在页面侧栏调整。

## 你的真实数据需要长这样

```csv
competition,season,date,home_team,away_team,home_goals,away_goals,neutral_site
EPL,2025-2026,2025-08-16,Liverpool,Bournemouth,4,2,false
WorldCup,2026,2026-06-11,Mexico,South Africa,2,0,true
```

日期建议使用 `YYYY-MM-DD`。球队名称必须前后一致，比如不要同时出现 `Man United` 和 `Manchester United`，除非先做映射清洗。

也可以导入 Football-Data 风格 CSV，常用字段会自动映射：

- `Div` -> `competition`
- `Date` -> `date`
- `HomeTeam` -> `home_team`
- `AwayTeam` -> `away_team`
- `FTHG` -> `home_goals`
- `FTAG` -> `away_goals`

## 推荐迭代路线

1. 先用英超最近 2-4 个赛季做基线。
2. 加入时间衰减，比较是否提升预测准度。
3. 单独补世界杯/国家队数据，不和俱乐部联赛混训。
4. 建立回测：按日期滚动训练，只预测未来比赛，避免数据泄漏。
5. 加入赛前数据特征，例如休息天数、伤停、赔率市场概率。
6. 用 log loss、Brier score、校准曲线评估，而不是只看命中率。

详细需求草案见 [docs/requirements_zh.md](docs/requirements_zh.md)。
