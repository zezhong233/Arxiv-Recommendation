#!/usr/bin/env python3
"""
生成推荐脚本
专门用于为用户生成个性化论文推荐
"""
import os
max_core = os.cpu_count()
used = str(max_core/2)
os.environ['NUMEXPR_MAX_THREADS'] = used

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_collector import ArxivDataCollector
from core.recommender import RecommendationEngine
import sqlite3
import json

def create_or_update_user(db_path, username, interests):
    """创建或更新用户"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 检查用户是否存在
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    
    interests_json = json.dumps(interests)
    
    if result:
        user_id = result[0]
        print(f"✅ 找到用户 '{username}' (ID: {user_id})")
        # 更新用户兴趣
        cursor.execute(
            "UPDATE users SET interests = ? WHERE id = ?",
            (interests_json, user_id)
        )
        print(f"📝 更新用户兴趣")
    else:
        # 创建新用户
        cursor.execute(
            "INSERT INTO users (username, interests) VALUES (?, ?)",
            (username, interests_json)
        )
        user_id = cursor.lastrowid
        print(f"✅ 创建新用户 '{username}' (ID: {user_id})")
    
    conn.commit()
    conn.close()
    
    return user_id

def main():
    print("=" * 60)
    print("论文推荐系统")
    print("=" * 60)
    
    # 1. 检查数据库
    collector = ArxivDataCollector()
    articles = collector.get_articles_from_db(limit=1000)
    
    if len(articles) == 0:
        print("❌ 数据库为空！")
        print("💡 请先运行: python scripts/fetch_arxiv.py")
        return
    
    print(f"📚 数据库中有 {len(articles)} 篇论文\n")
    
    # 2. 创建或获取用户
    print("设置用户兴趣...")
    test_interests = [
        "high redshift quasars",
        "galaxy formation",
        "dark matter",
        "active galactic nuclei",
        "AGN",
        "supermassive black holes",
        "LRDs",
        "obscured-AGN",
        "galaxy dust"
    ]
    
    user_id = create_or_update_user(
        collector.db_path,
        'arxiv_test_user',
        test_interests
    )
    
    print(f"🎯 用户兴趣: {', '.join(test_interests)}\n")
    
    # 3. 生成推荐
    print("=" * 60)
    print("正在生成推荐...")
    print("=" * 60)
    
    recommender = RecommendationEngine()
    recommendations = recommender.hybrid_recommend(user_id=user_id, limit=10)
    
    # 4. 显示结果
    if not recommendations:
        print("\n⚠️  没有生成推荐。可能的原因：")
        print("   - 数据库中论文数量较少")
        print("   - 论文内容与用户兴趣不匹配")
        print("   - 需要更多时间让系统学习")
    else:
        print(f"\n✅ 成功生成 {len(recommendations)} 个推荐：\n")
        print("=" * 60)
        for i, rec in enumerate(recommendations, 1):
            print(f"\n📄 推荐 {i}")
            print(f"标题: {rec['title']}")
            print(f"作者: {rec['authors']}")
            print(f"摘要: {rec['abstract'][:200]}...")
            print(f"PDF: {rec.get('pdf_url', 'N/A')}")
            print(f"推荐分数: {rec['recommendation_score']:.4f}")
            print("-" * 60)

if __name__ == "__main__":
    main()
