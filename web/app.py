"""
Flask Web应用主文件 - 简化版
单用户文章推荐系统
"""
import os
max_core = os.cpu_count()
used = str(max_core/2)
os.environ['NUMEXPR_MAX_THREADS'] = used

from flask import Flask, render_template, request, jsonify
import logging

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_config
from core.user_profiler import UserProfiler
from core.recommender import RecommendationEngine

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化Flask应用
app = Flask(__name__)
config = get_config()

# 初始化核心组件
user_profiler = UserProfiler(config)
recommendation_engine = RecommendationEngine(config)

# 固定用户ID（单用户系统）
DEFAULT_USER_ID = 1

def ensure_default_user():
    """确保默认用户存在"""
    user_info = user_profiler.get_user_by_id(DEFAULT_USER_ID)
    if not user_info:
        logger.info("默认用户不存在，正在创建...")
        try:
            # 创建默认用户
            user_id = user_profiler.create_user(
                username="default_user",
                interests=["machine learning", "deep learning", "artificial intelligence", "computer vision"]
            )
            logger.info(f"✅ 默认用户创建成功 (ID: {user_id})")
        except ValueError as e:
            # 用户名已存在，获取该用户
            logger.warning(f"用户名已存在: {e}")
            existing_user = user_profiler.get_user("default_user")
            if existing_user:
                logger.info(f"使用现有用户 (ID: {existing_user['id']})")
    else:
        logger.info(f"默认用户已存在 (ID: {DEFAULT_USER_ID})")

# 在应用启动时确保默认用户存在
ensure_default_user()

@app.route('/')
def index():
    """首页 - 文章推荐"""
    # 获取推荐文章
    recommendations = recommendation_engine.hybrid_recommend(DEFAULT_USER_ID, limit=20)
    
    return render_template('index.html', recommendations=recommendations)

@app.route('/api/like', methods=['POST'])
def like_article():
    """点赞文章"""
    data = request.get_json()
    article_id = data.get('article_id')
    
    if not article_id:
        return jsonify({'error': '缺少文章ID'}), 400
    
    user_profiler.record_user_behavior(DEFAULT_USER_ID, article_id, 'like')
    
    return jsonify({'message': '点赞成功'})

@app.route('/api/dislike', methods=['POST'])
def dislike_article():
    """不喜欢文章"""
    data = request.get_json()
    article_id = data.get('article_id')
    
    if not article_id:
        return jsonify({'error': '缺少文章ID'}), 400
    
    user_profiler.record_user_behavior(DEFAULT_USER_ID, article_id, 'dislike')
    
    return jsonify({'message': '已标记为不感兴趣'})

@app.route('/api/save', methods=['POST'])
def save_article():
    """收藏文章"""
    data = request.get_json()
    article_id = data.get('article_id')
    
    if not article_id:
        return jsonify({'error': '缺少文章ID'}), 400
    
    user_profiler.record_user_behavior(DEFAULT_USER_ID, article_id, 'save')
    
    return jsonify({'message': '收藏成功'})

@app.route('/api/refresh', methods=['POST'])
def refresh_recommendations():
    """刷新推荐"""
    try:
        # 生成新的推荐
        recommendations = recommendation_engine.hybrid_recommend(DEFAULT_USER_ID, limit=20)
        
        return jsonify({
            'message': '刷新成功',
            'count': len(recommendations)
        })
    except Exception as e:
        logger.error(f"刷新推荐失败: {e}")
        return jsonify({'error': '刷新失败'}), 500

@app.route('/settings')
def settings():
    """用户设置页面"""
    # 获取用户信息
    user_info = user_profiler.get_user_by_id(DEFAULT_USER_ID)
    
    # 获取用户行为统计
    behavior_count = recommendation_engine.get_user_behavior_count(DEFAULT_USER_ID)
    
    # 检查TF-IDF模型状态
    has_tfidf_model = user_profiler.load_user_tfidf(DEFAULT_USER_ID) is not None
    
    return render_template('settings.html', 
                         user=user_info, 
                         behavior_count=behavior_count,
                         has_tfidf_model=has_tfidf_model)

@app.route('/api/update_interests', methods=['POST'])

def update_interests():
    """更新用户兴趣"""
    data = request.get_json()
    interests_text = data.get('interests', '')
    
    if not interests_text:
        return jsonify({'error': '兴趣标签不能为空'}), 400
    
    # 解析兴趣标签（支持逗号、分号、换行分隔）
    import re
    interests = re.split(r'[,，;；\n]+', interests_text)
    interests = [interest.strip() for interest in interests if interest.strip()]
    
    if not interests:
        return jsonify({'error': '请至少输入一个兴趣标签'}), 400
    
    if len(interests) > 50:
        return jsonify({'error': '兴趣标签最多50个'}), 400
    
    try:
        # 更新用户兴趣
        user_profiler.update_user_interests(DEFAULT_USER_ID, interests)
        
        logger.info(f"用户兴趣已更新: {interests[:10]}...")
        
        return jsonify({
            'message': '兴趣标签更新成功',
            'interests': interests
        })
    except Exception as e:
        logger.error(f"更新兴趣失败: {e}")
        return jsonify({'error': '更新失败，请重试'}), 500

if __name__ == '__main__':
    # 启动应用
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )
