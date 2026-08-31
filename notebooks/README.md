# V2 notebooks

这些 Notebook 是统一线性与非线性比较的正式入口。历史 Notebook 保留在根目录和 `machinelearning_final/`，不追溯改变其结果解释。

当前已完成：

| 顺序 | Notebook | 状态 | Test |
|---:|---|---|---|
| 00 | [`00_统一数据探查.ipynb`](./00_统一数据探查.ipynb) | 已全量源文件执行；只输出 Train/Selection 统计 | 已在最终评分时读取 |
| 10 | [`10_共同输入_Logistic.ipynb`](./10_共同输入_Logistic.ipynb) | 已收敛并登记锁模 | 已评分 |
| 11 | [`11_WOE_Logistic.ipynb`](./11_WOE_Logistic.ipynb) | 已收敛并登记锁模 | 已评分 |
| 20 | [`20_随机森林.ipynb`](./20_随机森林.ipynb) | 已完成 OOB 预筛、全量决赛、配对比较并登记锁模 | 已评分 |
| 30 | [`30_CatBoost.ipynb`](./30_CatBoost.ipynb) | 已完成原生类别处理、受控搜索、边界延长、SHAP、配对比较并登记锁模 | 已评分 |
| 40 | [`40_LightGBM.ipynb`](./40_LightGBM.ipynb) | 已完成原生类别处理、受控搜索、Tree SHAP、配对比较并登记锁模 | 已评分 |
| 50 | [`50_XGBoost.ipynb`](./50_XGBoost.ipynb) | 已完成 Train-fitted One-Hot、受控搜索、Tree SHAP、配对比较并登记锁模 | 已评分 |
| 60 | [`60_MLP.ipynb`](./60_MLP.ipynb) | 已完成 Selection-AUC early stopping、受控搜索、permutation importance 并登记锁模 | 已评分 |
| 70 | [`70_最终Test揭盲.ipynb`](./70_最终Test揭盲.ipynb) | 已在全部模型锁定后一次性评分并写入可审计产物 | 已完成 |

八个 Notebook 均调用 `src/credit_risk/`，不复制公共数据和指标函数。运行时产物写入被 Git 忽略的 `artifacts/`。

建议从仓库根目录执行：

```bash
jupyter nbconvert --to notebook --execute --inplace \
  "notebooks/00_统一数据探查.ipynb" \
  --ExecutePreprocessor.timeout=1200

jupyter nbconvert --to notebook --execute --inplace \
  "notebooks/10_共同输入_Logistic.ipynb" \
  --ExecutePreprocessor.timeout=2400

jupyter nbconvert --to notebook --execute --inplace \
  "notebooks/11_WOE_Logistic.ipynb" \
  --ExecutePreprocessor.timeout=3600

jupyter nbconvert --to notebook --execute --inplace \
  "notebooks/20_随机森林.ipynb" \
  --ExecutePreprocessor.timeout=7200

jupyter nbconvert --to notebook --execute --inplace \
  "notebooks/30_CatBoost.ipynb" \
  --ExecutePreprocessor.timeout=10800

jupyter nbconvert --to notebook --execute --inplace \
  "notebooks/40_LightGBM.ipynb" \
  --ExecutePreprocessor.timeout=7200

jupyter nbconvert --to notebook --execute --inplace \
  "notebooks/50_XGBoost.ipynb" \
  --ExecutePreprocessor.timeout=10800

jupyter nbconvert --to notebook --execute --inplace \
  "notebooks/60_MLP.ipynb" \
  --ExecutePreprocessor.timeout=10800
```

七个模型均已登记，`artifacts/locked_models.json` 的 `suite_locked` 为 `true`。最终 Test
评分只运行过一次，产物位于 `artifacts/final_test/20260831T073511_991324Z/`。该随机 Test
划分曾在历史项目中被检查过，因此结果可用于本次受控比较，但报告不得将它描述成从未接触过的
独立泛化估计；如需该强度的结论，应另行建立新的封存留出集或时间外样本。
