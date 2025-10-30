"""
测试推荐系统功能
验证动态权重切换和自动画像更新
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.user_profiler import UserProfiler
from core.recommender import RecommendationEngine
from core.data_collector import ArxivDataCollector
from config.settings import get_config

def test_recommendation_system():
    """测试推荐系统"""
    print("=" * 60)
    print("测试推荐系统 - 动态权重切换")
    print("=" * 60)
    
    config = get_config()
    profiler = UserProfiler(config)
    engine = RecommendationEngine(config)
    collector = ArxivDataCollector(config)
    
    # 1. 确保有测试用户
    print("\n 检查测试用户...")
    user_info = profiler.get_user_by_id(1)
    if not user_info:
        print("   创建测试用户...")
        try:
            user_id = profiler.create_user(
                username="test_user",
                interests=["machine learning", "deep learning", "computer vision"]
            )
            print(f"   ✅ 用户创建成功 (ID: {user_id})")
        except ValueError:
            user_info = profiler.get_user("test_user")
            user_id = user_info['id']
            print(f"   ✅ 使用现有用户 (ID: {user_id})")
    else:
        user_id = user_info['id']
        print(f"   ✅ 用户已存在 (ID: {user_id})")
    
    # 2. 检查数据库中的文章数量
    print("\n2️⃣ 检查文章数据...")
    articles = collector.get_articles_from_db(limit=100)
    print(f"   📚 数据库中有 {len(articles)} 篇文章")
    
    if len(articles) == 0:
        print("   ⚠️  数据库为空，请先运行: python scripts/fetch_arxiv.py")
        return
    
    # 3. 测试初始推荐（行为数 < 10）
    print("\n3️⃣ 测试初始推荐（用户交互 < 10次）...")
    behavior_count = engine.get_user_behavior_count(user_id)
    print(f"   当前用户行为数: {behavior_count}")
    
    if behavior_count < 10:
        print("   🎯 应该只使用 Sentence-BERT 推荐")
        recommendations = engine.hybrid_recommend(user_id, limit=5)
        print(f"   ✅ 生成了 {len(recommendations)} 个推荐")
        
        if recommendations:
            print("\n   推荐文章示例:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec['title'][:60]}...")
                print(f"      分数: {rec['recommendation_score']:.3f}")
    
    # 4. 模拟用户交互
    print("\n4️⃣ 模拟用户交互...")
    if behavior_count < 10:
        print(f"   需要添加 {10 - behavior_count} 次交互以触发TF-IDF训练")
        
        # 随机选择一些文章进行交互
        import random
        sample_articles = random.sample(articles, min(10 - behavior_count, len(articles)))
        
        for article in sample_articles:
            action = random.choice(['click', 'like', 'save'])
            profiler.record_user_behavior(user_id, article['id'], action)
            print(f"   ✅ 记录行为: {action} - {article['title'][:50]}...")
        
        # 重新检查行为数
        behavior_count = engine.get_user_behavior_count(user_id)
        print(f"\n   当前用户行为数: {behavior_count}")
    
    # 5. 测试混合推荐（行为数 >= 10）
    if behavior_count >= 10:
        print("\n5️⃣ 测试混合推荐（用户交互 >= 10次）...")
        print("   🎯 应该使用 TF-IDF(0.4) + Sentence(0.6) 混合推荐")
        
        # 检查TF-IDF模型是否存在
        vectorizer = profiler.load_user_tfidf(user_id)
        if vectorizer:
            print("   ✅ TF-IDF模型已加载")
        else:
            print("   ⚠️  TF-IDF模型不存在（可能正在训练中）")
        
        recommendations = engine.hybrid_recommend(user_id, limit=5)
        print(f"   ✅ 生成了 {len(recommendations)} 个推荐")
        
        if recommendations:
            print("\n   推荐文章示例:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec['title'][:60]}...")
                print(f"      分数: {rec['recommendation_score']:.3f}")
    
    # 6. 测试用户画像
    print("\n6️⃣ 检查用户画像...")
    user_info = profiler.get_user_by_id(user_id)
    if user_info:
        interests = user_info.get('interests', [])
        print(f"   用户兴趣: {', '.join(interests[:10])}")
        if len(interests) > 10:
            print(f"   ... 共 {len(interests)} 个兴趣标签")
    
    # 7. 总结
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print(f"✅ 用户ID: {user_id}")
    print(f"✅ 行为记录数: {behavior_count}")
    print(f"✅ 推荐策略: {'混合推荐 (TF-IDF + Sentence)' if behavior_count >= 10 else '纯Sentence推荐'}")
    print(f"✅ TF-IDF模型: {'已训练' if profiler.load_user_tfidf(user_id) else '未训练'}")
    print("\n💡 提示:")
    print("   - 继续与文章交互，系统会自动更新用户画像")
    print("   - 每20次交互会重新训练TF-IDF模型")
    print("   - 启动Web界面: python web/app.py")

if __name__ == "__main__":
    test_recommendation_system()
