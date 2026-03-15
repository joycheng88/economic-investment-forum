# Quick Reference: Real-Time GLP-1 Sentiment Analysis Pipeline

## 🎯 核心概念速记

### VADER情感分析
```
VADER (Valence Aware Dictionary and sEntiment Reasoner)
├─ 输入: 文本字符串
├─ 输出: 4个分数
│   ├─ positive: 0.0-1.0 (正面强度)
│   ├─ negative: 0.0-1.0 (负面强度)
│   ├─ neutral: 0.0-1.0 (中性强度)
│   └─ compound: -1.0-1.0 (综合情感 ★ 最重要)
└─ 标签规则:
    ├─ compound > 0.05  → 正面 (label=1)
    ├─ compound < -0.05 → 负面 (label=0)
    └─ 其他 → 中性 (label=2)

示例：
  compound=0.72 → 强正面
  compound=-0.65 → 强负面
  compound=0.0 → 中性
```

### TF-IDF特征
```
TF-IDF(term, doc) = TF(term, doc) × IDF(term)

TF = 词在文档中出现次数 / 文档总词数
     → 衡量该词在文档中的重要性

IDF = log(总文档数 / 包含该词的文档数)
      → 衡量该词在整个语料中的稀有性

TF-IDF高 = 词频高 + 稀有
TF-IDF低 = 常见词汇（如"the", "is"等）

参数调优：
  max_features=5000     → 保留最常见的5000个词
  min_df=2              → 至少出现2次的词
  max_df=0.8            → 去掉80%以上文档都有的词
  ngram_range=(1,2)     → 使用单词和词对
```

### 情感指数计算
```
SI(t) = 100 + 10 × [α × P_pos(t) - (1-α) × P_neg(t)]

其中：
  SI(t): 时刻t的情感指数
  P_pos(t): 时刻t的正面概率（0-1）
  P_neg(t): 时刻t的负面概率（0-1）
  α = 0.6: 正面权重（可调）

示例计算：
  假设某日：P_pos=0.7, P_neg=0.2
  SI = 100 + 10×[0.6×0.7 - 0.4×0.2]
     = 100 + 10×[0.42 - 0.08]
     = 100 + 3.4
     = 103.4

解释：
  SI > 105 → 强烈看涨 ⬆️
  95-105  → 中性 ➡️
  SI < 95 → 强烈看跌 ⬇️
```

### 模型性能指标
```
Accuracy  = (TP+TN)/(TP+TN+FP+FN)
           = 整体预测正确比例

Precision = TP/(TP+FP)
           = 预测为正面的准确性
           = "有多可信"

Recall    = TP/(TP+FN)
           = 真实正面的发现率
           = "有多全面"

F1-Score  = 2×(Precision×Recall)/(Precision+Recall)
           = Precision和Recall的调和平均
           = ★ 最平衡的综合指标

ROC-AUC   = ROC曲线下的面积
           = 衡量模型的判别能力
           = 0.5(随机) ~ 1.0(完美)

关键点：
- 使用F1-Score作为主要评估指标
- ROC-AUC反映模型对阈值变化的鲁棒性
- 在类别不平衡时，F1比准确率更重要
```

---

## 🔧 常用代码片段

### 1. 加载和使用模型
```python
# 加载所有模型
models = load_models('./models')

# 单条预测
text = "GLP-1治疗显示出色的临床效果"
result = predict_sentiment_ensemble(text, models)
print(f"情感: {result['sentiment']}")
print(f"概率: {result['probability']:.2%}")
print(f"置信度: {result['confidence']:.2%}")

# 批量预测
texts = ["文本1", "文本2", "文本3"]
results = [predict_sentiment_ensemble(t, models) for t in texts]

# 提取结果
sentiments = [r['sentiment'] for r in results]
probabilities = [r['probability'] for r in results]
```

### 2. 生成VADER标签
```python
# 单条文本
result = generate_weak_labels_vader("某个文本")
print(f"标签: {result['label']}")
print(f"Compound分数: {result['compound']:.4f}")

# 批量标注
df['labels'] = df['text'].apply(lambda x: 
    generate_weak_labels_vader(x)['label']
)
```

### 3. 计算情感指数
```python
# 按日聚合
daily_sentiment = df_all.groupby(df_all['date'].dt.date).agg({
    'prob_positive_ensemble': ['mean', 'std', 'count']
})

# 计算指数
alpha = 0.6
SI = 100 + 10 * (
    alpha * daily_sentiment['mean_prob'] - 
    (1-alpha) * (1 - daily_sentiment['mean_prob'])
)
```

### 4. 评估模型
```python
from sklearn.metrics import f1_score, roc_auc_score

# F1分数
f1 = f1_score(y_true, y_pred)

# ROC-AUC
auc = roc_auc_score(y_true, y_pred_proba)

# 完整评估
metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred),
    'recall': recall_score(y_true, y_pred),
    'f1': f1_score(y_true, y_pred),
    'auc': roc_auc_score(y_true, y_pred_proba)
}
```

### 5. 保存/加载模型
```python
import joblib
from tensorflow.keras.models import load_model

# 保存逻辑回归
joblib.dump(lr_model, 'lr_model.joblib')

# 加载逻辑回归
lr_model = joblib.load('lr_model.joblib')

# 保存神经网络
nn_model.save('nn_model.h5')

# 加载神经网络
nn_model = load_model('nn_model.h5')
```

---

## 📊 数据预处理

### 文本清洗流程
```
原始文本
  ↓
转小写 → "GLP-1 works Great!" → "glp-1 works great!"
  ↓
去URL → "Check https://example.com" → "Check"
  ↓
去特殊字符 → "Cost: $100!" → "Cost 100"
  ↓
分词 → "Cost 100" → ["cost", "100"]
  ↓
去停用词 → ["cost"] (去掉"100")
  ↓
词形还原 → ["cost"] (已是原形)
  ↓
重新组合 → "cost"
```

### 特征编码方式

**TF-IDF编码：**
```
文本: "GLP-1 is effective"
处理后: ["glp", "effective"]

TF-IDF矩阵（示例）:
词汇       TF-IDF值
glp        0.45
effective  0.89
diabetes   0.00
...
(共5000维)
```

**Embedding编码：**
```
文本: "GLP-1 is effective"
处理后: ["glp", "is", "effective"]

Embedding矩阵（示例）:
glp        → [0.2, -0.5, 0.3, ..., 0.1] (100维向量)
is         → [0.8, 0.1, -0.2, ..., 0.6]
effective  → [0.6, 0.4, 0.5, ..., -0.3]

语义关系：
  vec("glp") + vec("effective") ≈ vec("有效的glp")
```

---

## ⚙️ 模型参数调优指南

### 逻辑回归
```python
LogisticRegression(
    max_iter=1000,              # 最大迭代次数
    random_state=42,            # 随机种子
    class_weight='balanced',    # 处理类别不平衡
    C=1.0,                      # 正则化强度（小=强）
    solver='lbfgs'              # 求解器
)

调优建议：
- 如果准确率低：增加max_iter或调整C
- 如果过拟合：减小C（增加正则化）
- 如果欠拟合：增大C（减少正则化）
```

### 神经网络
```python
Sequential([
    Embedding(input_dim=5000, output_dim=100),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

调优建议：
- 如果过拟合：增加Dropout比例（如0.3或0.4）
- 如果欠拟合：增加Dense层的单元数
- 如果收敛慢：调整学习率（如0.01或0.0001）
- 如果震荡：使用BatchNormalization
```

---

## 🚨 常见问题排查

### 问题1: "模型F1分数低（<0.70）"
```
排查步骤：
1. 检查数据质量
   ├─ 是否有大量空值？
   ├─ 文本长度分布是否合理？
   └─ 类别是否严重不平衡？

2. 检查标签质量
   ├─ VADER标签阈值是否合理？
   ├─ 是否需要人工审核调整？
   └─ 是否有标签噪声？

3. 调整模型
   ├─ 增加TF-IDF特征维度
   ├─ 调整正则化参数
   └─ 尝试更复杂的模型
```

### 问题2: "模型在新数据上性能下降"
```
原因和解决：
1. 数据分布变化（Dataset Shift）
   → 重新收集训练数据，混合新旧数据重训

2. 语言/表达方式变化
   → 更新停用词列表，调整文本预处理

3. 模型漂移（Model Drift）
   → 定期评估，每周或每月重训一次

4. 新出现的关键词
   → 更新关键词列表，可能需要扩展TF-IDF词表
```

### 问题3: "预测速度太慢"
```
优化方案：
1. 使用更轻量的模型
   ├─ 逻辑回归比神经网络快100倍
   └─ TF-IDF比Embedding快

2. 减少特征维度
   ├─ 降低max_features（如3000而非5000）
   └─ 增加min_df阈值

3. 硬件加速
   ├─ 使用GPU运行神经网络
   └─ 使用量化模型（如INT8）

4. 批处理和缓存
   ├─ 批量预测而非单条
   └─ 缓存常见查询结果
```

---

## 📅 维护检查清单

### 每日检查
- [ ] 数据是否正常收集？
- [ ] 是否有异常错误日志？
- [ ] 预测延迟是否正常？

### 每周检查
- [ ] 模型在新数据上的F1分数
- [ ] 情感指数的变化趋势
- [ ] 是否需要模型更新？

### 每月检查
- [ ] 整体性能报告
- [ ] 类别分布是否变化？
- [ ] 是否出现新的关键词或表达？
- [ ] 计划下一次的重新训练？

### 季度检查
- [ ] 对标竞争对手的系统
- [ ] 评估新模型架构的必要性
- [ ] 数据和标签质量审计
- [ ] 文档更新

---

## 📈 指标对标参考

### 不同任务的F1分数参考
```
任务难度    相关领域          一般F1分数范围
───────────────────────────────────────
简单      新闻分类          0.85-0.95
中等      情感分析          0.75-0.85
困难      细粒度分类        0.60-0.75
超困难     实体关系抽取      0.40-0.60

我们的系统：F1 ≈ 0.88 → 中等偏上水平
```

### 部署前的准备清单
```
模型方面：
  [ ] F1分数 > 0.80
  [ ] ROC-AUC > 0.85
  [ ] 在测试集上的性能稳定
  [ ] 推理延迟 < 1秒/条

数据方面：
  [ ] 训练数据 > 1000条
  [ ] 类别分布相对均衡
  [ ] 验证集和测试集无泄露
  [ ] 数据质量审核完成

系统方面：
  [ ] 模型已保存
  [ ] 版本控制完善
  [ ] 监控告警配置
  [ ] 文档齐全
```

---

## 🎓 学习资源

### 理论基础
- VADER论文: Hutto & Gilbert (2014)
- TF-IDF: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- 情感分析综述: SemEval任务论文

### 实践工具
- NLTK: https://www.nltk.org/
- Scikit-learn: https://scikit-learn.org/
- TensorFlow: https://tensorflow.org/
- Transformers: https://huggingface.co/transformers/

### 数据集
- Twitter sentiment (SemEval)
- IMDB电影评论
- 中文新浪微博 (NLPCC)

---

**最后更新**: 2026年1月  
**适用版本**: GLP-1 Sentiment Analysis v1.0
