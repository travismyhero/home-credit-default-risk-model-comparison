# 仓库协作与修改约定

本仓库同时保留历史五模型比较和新的统一七模型实验。修改时优先保证可复现、无数据泄漏和旧入口兼容。

## 开始一次修改

```bash
git status --short --branch
git switch -c <type>/<short-description>
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest
```

不要在存在不明未提交修改时执行 `git reset --hard`、强制 checkout 或清理整个工作区。

## 分支命名

建议使用：

- `feat/<name>`：新增模型或功能。
- `fix/<name>`：修复错误。
- `refactor/<name>`：结构调整但不改变预期结果。
- `docs/<name>`：文档修改。
- `experiment/<name>`：预先声明的新实验。

## 提交信息

使用小而明确的提交：

```text
docs: define final-test policy
refactor: extract shared feature engineering
feat: add common-input logistic baseline
test: reject overlapping split indices
```

不要把源代码、全量模型产物、缓存和临时 Notebook 混在同一提交中。

## 数据与产物

- Kaggle 原始数据放在 `data/`，由 `.gitignore` 排除。
- 运行期模型、检查点和大体积报告放在 `artifacts/` 或模型自己的 `model_outputs/`，默认不提交。
- 需要长期保存的小型最终表格和图表放在 `reports/`，并在 README 中说明来源。
- 不提交 `catboost_info/`、缓存、debug executed Notebook 或包含秘密/令牌的文件。

## Notebook 规则

- Notebook 只负责实验编排、参数声明和结果展示。
- 公共数据、切分、特征、指标和锁模规则优先放在 `src/credit_risk/`。
- 预处理统计量只在 Train 上拟合。
- Selection 用于 Early Stopping、参数选择和辅助诊断。
- 所有规定模型登记锁定前，不得运行最终 Test 单元。
- 正式比较必须保存按 `SK_ID_CURR` 对齐的预测。

## 合并前检查

```bash
make check-config
make test
git diff --check
git status --short
```

合并请求或提交说明应写清：

- 改动目的。
- 使用的数据和切分指纹。
- 是否改变特征政策或 Test 政策。
- 运行了哪些测试。
- 新增或更新了哪些模型产物。
