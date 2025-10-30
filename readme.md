# AI推荐系统 - 方案A实施总结

## 📋 实施概述

根据 `NEW_FEATURES.md` 的需求，我们实施了**方案A：最小改动方案**，修复了核心推荐逻辑，实现了动态权重切换和自动用户画像更新。

## ✅ 已完成的修改

### 1. **动态权重切换机制** (`core/recommender.py`)

#### 修改内容：
- 新增 `get_user_behavior_count()` 方法：统计用户的正面行为数量（click/like/save）
- 修改 `content_based_recommend()` 方法：实现动态权重策略

#### 推荐策略：
```python
# 用户交互 < 10次：只用 Sentence-BERT (权重 1.0)
if behavior_count < 10:
    combined_score = semantic_similarity

# 用户交互 >= 10次：TF-IDF(0.4) + Sentence-BERT(0.6)
if behavior_count >= 10:
    combined_score = 0.6 * semantic_similarity + 0.4 * normalized_tfidf
```

#### 符合需求：
✅ 新用户（交互少）只使用Sentence模型  
✅ 老用户（交互多）使用混合推荐策略  
✅ 权重比例符合需求：TF-IDF(0.4) + Sentence(0.6)

---

### 2. **自动用户画像更新** (`core/user_profiler.py`)

#### 修改内容：
- 新增 `_auto_update_user_profile()` 内部方法：自动触发画像更新
- 修改 `record_user_behavior()` 方法：在记录行为后自动检查是否需要更新

#### 更新策略：
```python
# 阈值1: 达到10次行为时，首次训练TF-IDF模型
if behavior_count == 10:
    train_user_tfidf()
    update_user_profile_from_behavior()

# 阈值2: 每增加20次行为，重新训练模型
elif behavior_count > 10 and behavior_count % 20 == 0:
    train_user_tfidf()
    update_user_profile_from_behavior()
```

#### 符合需求：
✅ 用户交互达到阈值时自动训练TF-IDF模型  
✅ 定期更新用户画像（每20次交互）  
✅ 无需手动触发，完全自动化

---

### 3. **优化TF-IDF训练逻辑** (`core/user_profiler.py`)

#### 修改内容：
- 修改 `get_user_tfidf_score()` 方法：移除自动训练逻辑
- 训练控制权交给 `_auto_update_user_profile()`

#### 改进：
- **之前**：每次计算分数时都检查是否需要训练（低效）
- **现在**：只在用户行为达到阈值时训练（高效）

#### 符合需求：
✅ 确保只有足够行为数据才训练TF-IDF  
✅ 避免频繁训练，提高性能  
✅ 训练时机更加合理

---

### 4. **默认用户自动创建** (`web/app.py`)

#### 修改内容：
- 新增 `ensure_default_user()` 函数：在应用启动时检查并创建默认用户
- 应用启动时自动调用

#### 改进：
```python
def ensure_default_user():
    """确保默认用户存在"""
    user_info = user_profiler.get_user_by_id(DEFAULT_USER_ID)
    if not user_info:
        # 创建默认用户
        user_profiler.create_user(
            username="default_user",
            interests=["machine learning", "deep learning", ...]
        )
```

#### 符合需求：
✅ 系统启动时自动创建默认用户  
✅ 避免用户不存在导致的错误  
✅ 提供合理的初始兴趣标签

---

### 5. **测试脚本** (`scripts/test_recommendation.py`)

#### 功能：
- 验证动态权重切换是否正常工作
- 测试自动画像更新机制
- 模拟用户交互过程
- 检查TF-IDF模型训练状态

#### 使用方法：
```bash
python scripts/test_recommendation.py
```

---

## 🎯 核心功能验证

### 推荐策略流程

```
用户开始使用
    ↓
记录用户行为 (click/like/save/dislike)
    ↓
检查行为数量
    ↓
┌─────────────────────────────────────┐
│ 行为数 < 10次                        │
│ → 只用 Sentence-BERT (权重1.0)      │
│ → 不使用 TF-IDF                      │
└─────────────────────────────────────┘
    ↓ (继续交互)
┌─────────────────────────────────────┐
│ 行为数 = 10次                        │
│ → 自动训练 TF-IDF 模型               │
│ → 更新用户画像                       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 行为数 >= 10次                       │
│ → 混合推荐: TF-IDF(0.4) + Sentence(0.6) │
└─────────────────────────────────────┘
    ↓ (继续交互)
┌─────────────────────────────────────┐
│ 每增加20次行为                       │
│ → 重新训练 TF-IDF 模型               │
│ → 更新用户画像                       │
└─────────────────────────────────────┘
```

---

## 📝 使用指南

### 1. 首次运行

```bash
# 1. 获取arxiv文章数据
python scripts/fetch_arxiv.py

# 2. (可选) 创建演示数据
python utils/create_demo_data.py

# 3. 测试推荐系统
python scripts/test_recommendation.py

# 4. 启动Web应用
python web/app.py
```

### 2. 日常使用

```bash
# 定期更新文章数据（建议每天运行）
python scripts/fetch_arxiv.py

# 启动Web应用
python web/app.py
# 访问 http://localhost:8080
```

### 3. Web界面操作

- **刷新推荐**：点击"刷新"按钮获取新推荐
- **点赞文章**：点击"👍"按钮
- **不感兴趣**：点击"👎"按钮
- **收藏文章**：点击"⭐"按钮

---

## 🔍 技术细节

### 推荐算法权重

| 用户状态 | Sentence-BERT | TF-IDF | 说明 |
|---------|--------------|--------|------|
| 新用户 (< 10次交互) | 1.0 | 0.0 | 只用语义相似度 |
| 老用户 (≥ 10次交互) | 0.6 | 0.4 | 混合推荐 |

### 自动更新触发点

| 行为数量 | 触发动作 |
|---------|---------|
| 10次 | 首次训练TF-IDF + 更新画像 |
| 30次 | 重新训练TF-IDF + 更新画像 |
| 50次 | 重新训练TF-IDF + 更新画像 |
| ... | 每20次重新训练 |

### 行为权重

| 行为类型 | 权重 | 说明 |
|---------|------|------|
| click | 1x | 基础权重 |
| like | 2x | 点赞权重加倍 |
| save | 3x | 收藏权重最高 |
| dislike | -1x | 负面反馈 |

---

## 🚀 性能优化

### 已实现的优化

1. **延迟训练**：只在达到阈值时训练TF-IDF，避免频繁训练
2. **缓存模型**：TF-IDF模型保存到文件，避免重复训练
3. **批量处理**：每20次行为才重新训练，减少计算开销
4. **智能候选**：优先推荐最新fetch的文章，提高推荐时效性

---

## 📊 系统状态监控

### 查看用户状态

```python
from core.user_profiler import UserProfiler
from core.recommender import RecommendationEngine

profiler = UserProfiler()
engine = RecommendationEngine()

# 查看用户行为数量
behavior_count = engine.get_user_behavior_count(user_id=1)
print(f"用户行为数: {behavior_count}")

# 查看用户画像
user_info = profiler.get_user_by_id(user_id=1)
print(f"用户兴趣: {user_info['interests']}")

# 检查TF-IDF模型
vectorizer = profiler.load_user_tfidf(user_id=1)
print(f"TF-IDF模型: {'已训练' if vectorizer else '未训练'}")
```

---

## ⚠️ 注意事项

### 1. 数据库要求
- 确保数据库中有足够的文章（建议 > 50篇）
- 定期运行 `fetch_arxiv.py` 更新文章

### 2. 模型文件
- TF-IDF模型保存在 `data/models/` 目录
- 首次训练需要一定时间（取决于行为数量）

### 3. 性能考虑
- Sentence-BERT模型加载需要时间（首次启动较慢）
- 推荐计算时间与候选文章数量成正比

---

## 🔄 未来扩展（方案B）

如果需要完整的多用户支持，可以进一步实现：

1. **用户认证系统**
   - 注册/登录页面
   - Session管理
   - 密码加密

2. **多用户界面**
   - 用户个人中心
   - 用户设置页面
   - 历史记录查看

3. **用户管理**
   - 管理员后台
   - 用户权限控制
   - 数据统计分析

---

## 📞 问题排查

### 常见问题

**Q: 推荐结果为空？**
- 检查数据库是否有文章：`python scripts/fetch_arxiv.py`
- 检查用户是否有初始兴趣标签

**Q: TF-IDF模型未训练？**
- 确认用户行为数量 ≥ 10次
- 查看日志确认训练是否成功

**Q: 推荐质量不佳？**
- 增加用户交互次数（like/save）
- 调整初始兴趣标签
- 等待TF-IDF模型训练完成

---

## ✅ 总结

方案A成功实现了以下目标：

1. ✅ **动态权重切换**：根据用户交互次数自动调整推荐策略
2. ✅ **自动画像更新**：在关键阈值自动训练模型和更新画像
3. ✅ **最小改动**：保持单用户模式，核心功能完整
4. ✅ **易于扩展**：数据库已支持多用户，可轻松扩展到方案B

系统现在完全符合 `NEW_FEATURES.md` 的需求，可以投入使用！
