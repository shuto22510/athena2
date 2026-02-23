# 変圧器オイル温度予測 PoC

ETTh1データセットを用いた時系列予測モデルの段階的検証。
ベースライン→ML→DLの3段階で「MLは本当に必要か？」を定量的に評価。

## 成果物
- **最終レポート**: [poc_report_v4.pdf](slides/poc_report_v4.pdf)
- **分析ノートブック**: [poc_analysis.ipynb](notebooks/poc_analysis.ipynb)

## 主要な結果（Operational Scenario A: Lag 0利用可能）
実運用を想定し「時刻tの観測値を得た直後に未来を予測する」条件で統一評価。

| モデル | 1h | 6h | 24h | 168h |
|---|---|---|---|---|
| Persistence (BL) | 0.44 | 1.16 | 1.63 | 2.76 |
| **Ridge (推薦)** | **0.44** | **1.04** | **1.56** | **2.29** |

## 結論
- **短期(1h)**: 直前値(Persistence)が最強。
- **長期(168h)**: Ridge(α=2000またはVal調律版)がベースライン比 **-17.2%** の改善を達成。
- **実用性**: 計画業務（中長期予測）においてMLモデルは極めて高い付加価値を提供。

## セットアップ
pip install -r requirements.txt

## 実行手順
1. `scripts/model_01_baseline.py` — ベースライン評価
2. `scripts/model_02_ml.py` — ML評価（Ridge/LightGBM）
3. `scripts/model_05_informer.py` — Informer評価
4. `notebooks/poc_analysis.ipynb` — 統合分析

## 修正履歴
- SeasonalNaiveのリーク修正（h>24で未来値を参照するバグを発見・修正）
- Ridge α最適化（α=1→2000で168h MAE 16%改善）

※ スライド・Notebookの数値はlag0なし（lag1始まり）の初期検証結果です。
その後、運用シナリオを「観測直後に予測する」と明確化し、
lag0（現在の観測値）を特徴量に追加して再検証しました。
scripts/内のコードはlag0あり版（最新）のため、一部数値が異なります。
