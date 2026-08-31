# Repository workflow: unified model benchmark V2

## Purpose

V2 在不破坏历史五模型项目的前提下，逐步建立共同输入 Logistic、WOE Logistic、Random Forest、CatBoost、LightGBM、XGBoost 和 MLP 的统一比较流程。

## Sources of truth

| Contract | File |
|---|---|
| 数据、切分、指标和 Test 政策 | `configs/experiment.yaml` |
| 公共字段与固定特征工程 | `configs/feature_policy.yaml` |
| 各模型预先声明的搜索范围 | `configs/model_search_spaces.yaml` |
| 详细研究方案 | `machinelearning_final/README.md` |
| 共享实现 | `src/credit_risk/` |
| 自动验证 | `tests/` |

Notebook 中的局部配置若与版本化配置冲突，正式 V2 实验以版本化配置为准；历史运行保持原解释，不追溯修改。

## Compatibility strategy

整合分三步进行：

1. **并存**：保留根目录历史 Notebook、输出和验证模块；新增配置与共享包。
2. **迁移**：每个正式模型 Notebook 逐一改为读取共享切分、特征政策和指标。
3. **收敛**：所有模型锁定后由单独的最终比较入口统一访问 Test。

在迁移完成前，不批量移动 Notebook，不改变历史报告的相对路径。

## Final-Test gate

`configs/experiment.yaml` 列出的每个模型必须在 `artifacts/locked_models.json` 中登记：

- 数据 SHA256。
- 切分 SHA256。
- 特征政策版本。
- 最终超参数和训练轮数。
- Selection 指标。
- 模型与预处理产物路径。

只有登记表中所有模型的 `locked=true` 且指纹一致，`assert_test_access_allowed` 才允许最终比较入口读取 Test。

该门槛是流程保护，不是安全边界。Notebook 作者仍需避免在门槛之外直接读取 Test。

当前 V2 固定切分指纹为 `4ae4ceee661ba8c210fa0a1cf950448799a0aef468659e6d2590c1bb4e8a3835`，其计算同时包含三组有序行索引和 `random_state=42`。

## Generated artifacts

运行期产物统一进入 `artifacts/`：

```text
artifacts/
├── locked_models.json
├── common_logistic/<run_id>/
├── woe_logistic/<run_id>/
├── random_forest/<run_id>/
├── catboost/<run_id>/
├── lightgbm/<run_id>/
├── xgboost/<run_id>/
├── mlp/<run_id>/
└── final_comparison/<run_id>/
```

每个模型目录至少保存锁定配置、数据/切分指纹、预处理规则、特征清单、搜索结果、模型文件和 Selection 预测。Test 预测只在最终比较阶段生成。

## Recommended development sequence

```text
shared contract
→ common-input Logistic
→ WOE Logistic migration
→ formal Random Forest
→ CatBoost alignment
→ LightGBM/XGBoost alignment
→ formal MLP
→ model-suite lock
→ one-time Test comparison
```

## Local checks

```bash
make check-config
make test
git diff --check
```

自动测试不依赖未提交的 Kaggle 数据，使用合成样本验证公共契约。
