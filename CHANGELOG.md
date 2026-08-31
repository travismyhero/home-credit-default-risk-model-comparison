# Changelog

本文件记录影响实验口径、共享代码、模型范围和复现方式的变化。自动生成的指标波动不单独记录，除非对应正式锁定运行。

## Unreleased

### Added

- 新的统一线性与非线性模型比较协议。
- 版本化的实验、特征政策和模型搜索空间配置。
- `src/credit_risk/` 共享包：数据契约、固定切分、特征工程、统一指标、配对 Bootstrap、候选选择和 Test 访问门槛。
- Random Forest、CatBoost、LightGBM、XGBoost、MLP 及两类 Logistic 基准的统一登记要求。
- 防止样本重叠、配置缺失、Schema 错误、Test 提前访问和指纹不一致的自动测试。
- Git 协作约定、Makefile 和持续集成配置。
- 无 Test 物化的 V2 统一数据探查 Notebook。
- 已正式执行并登记的共同输入 Logistic 与 WOE Logistic Notebook。

### Changed

- 将新的七模型研究定位为 V2 受控实验；保留根目录历史五模型比较作为已执行基线。
- 明确区分“共同输入的算法对照”和“每个模型的最佳实践”两条实验线。
- 明确当前 Test 在早期开发中已经暴露，最终报告必须披露该限制或改用新的封存 Holdout。

### Compatibility

- 没有移动或重命名现有 Notebook。
- 没有删除或覆盖根目录历史输出。
- 现有 `credit_risk_validation.py` 和旧 Notebook 入口继续可用。
