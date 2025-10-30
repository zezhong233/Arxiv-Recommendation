import sys
import os
# 设置NumExpr环境变量以优化性能
max_core = os.cpu_count()
used = str(max_core/2)
os.environ['NUMEXPR_MAX_THREADS'] = used
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_collector import ArxivDataCollector
from core.text_analyzer import TextAnalyzer
from core.recommender import RecommendationEngine
from datetime import datetime, timedelta

def ensure_directories():
    """确保必要的目录存在"""
    import os
    from config.settings import get_config
    
    config = get_config()
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

def main():
    # 0. 确保目录存在
    ensure_directories()
    
    # 1. 初始化收集器
    collector = ArxivDataCollector()
    
    # 2. 设置搜索参数
    from config.settings import get_config
    config = get_config()
    

    categories  = [config.YOUR_CATEGORIES]
    MAX_FETCH_DAYS = 7
    INITIA_FETCH_DAYS = 30
    # 3. 检查数据库状态和上次更新时间
    print("=" * 60)
    print("检查本地数据库状态...")
    print("=" * 60)
    
    existing_articles = collector.get_articles_from_db(limit=1000)
    days_since_update = collector.get_days_since_last_update()
    
    print(f"📊 数据库中已有 {len(existing_articles)} 篇文章")
    
    # 4. 智能决定更新策略
    if len(existing_articles) == 0:
        # 情况1: 数据库为空，首次运行
        print("📭 数据库为空，执行首次数据获取")
        print(f"📥 目标获取 {INITIA_FETCH_DAYS} 天的论文")
        # 首次运行获取更长时间范围以确保获得足够文章
        days_to_fetch = INITIA_FETCH_DAYS  
        
    elif days_since_update is None:
        # 情况2: 无法获取上次更新时间（异常情况）
        print("⚠️  无法确定上次更新时间，执行完整更新")
        days_to_fetch = MAX_FETCH_DAYS
        
    else:
        # 情况3: 根据距离上次更新的天数决定策略
        print(f"⏰ 距离上次更新已过 {days_since_update} 天")
        
        if days_since_update == 0:
            # 今天已更新过
            print("✅ 今天已更新过，使用现有数据")
            days_to_fetch = 0
        elif days_since_update < MAX_FETCH_DAYS:
            # 间隔较短，增量更新
            print(f"🔄 执行增量更新：获取最近 {days_since_update + 1} 天的论文")
            days_to_fetch = days_since_update + 1  # +1确保覆盖到今天
        else:
            # 间隔较长，获取最近3天的数据
            print(f"🔄 间隔较长，重新获取最近 {MAX_FETCH_DAYS} 天的论文")
            days_to_fetch = MAX_FETCH_DAYS
    
    # 5. 执行数据获取（如果需要）
    if days_to_fetch > 0:
        print(f"\n{'=' * 60}")
        print(f"开始从 arXiv 获取论文...")
        print(f"{'=' * 60}")
        print(f"🎯 目标类别: {', '.join(categories)}")
        print(f"📅 时间范围: 最近 {days_to_fetch} 天")
        
        articles = collector.fetch_recent_articles(
            days=days_to_fetch,
            categories=categories,
        )
        

        if articles:
            saved_count = collector.save_articles(articles)
            print(f"\n成功保存 {saved_count} 篇论文")
            
            # 检查去重效果
            new_count = saved_count
            fetched_count = len(articles)
            duplicate_count = fetched_count - new_count
            
            if duplicate_count > 0:
                print(f" 检测到 {duplicate_count} 篇重复文章（已自动跳过）")
        else:
            print("  未获取到新文章")
    else:
        print("无需更新，使用现有数据")
    
    # 6. 从数据库加载文章用于推荐
    print(f"\n{'=' * 60}")
    print("数据同步完成！")
    print(f"{'=' * 60}")
    
    all_articles = collector.get_articles_from_db(limit=1000)
    print(f"当前数据库共有 {len(all_articles)} 篇论文")
if __name__ == "__main__":
    main()