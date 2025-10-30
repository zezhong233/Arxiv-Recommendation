# AI推荐系统 

本推荐系统在claude-sonnet-4-5的指导下完成。

## 📝 使用指南

### 1. 首次运行

```bash
# 0. 相关package(建议新建一个环境)

pip install -r requirements.txt

# 1. 获取arxiv文章数据
python scripts/fetch_arxiv.py

# 3. 测试推荐系统
python scripts/test_recommendation.py

# 4. 启动Web应用
python web/app.py
```

- **进入右上角setting选择你感兴趣领域的label**。

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

本文档由chat gpt 4.1o 实现

