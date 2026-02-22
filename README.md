# 変圧器オイル温度予測 PoC

ETTh1データセットを用いた時系列予測モデルの段階的検証。
BL→ML→DLの3段階で「MLは本当に必要か？」を定量的に評価。

## 成果物
- **最終レポート**: [poc_report_v4.pdf](slides/poc_report_v4.pdf)
- **分析ノートブック**: [poc_analysis.ipynb](notebooks/poc_analysis.ipynb)

## 主要な結果（168h予測 MAE ℃）
| モデル | ETTh1 | ETTh2 | ETTm1 | ETTm2 |
|---|---|---|---|---|
| Persistence | 2.76 | 5.70 | 2.77 | 5.70 |
| Ridge(α=2000) | **2.31** | **4.71** | **2.31** | **4.41** |
| Informer | 3.66 | 4.85 | 4.23 | 5.21 |

## 結論
- 短期(1-6h): Persistenceが最強
- 長期(168h): Ridge(α=2000)が全4データセットでBL比-16～23%改善
- DL(Informer): 論文値超を達成したがRidgeに及ばず

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
