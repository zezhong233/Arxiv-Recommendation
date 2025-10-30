"""
Arxiv推荐系统配置文件
"""

import os

class Config:
    """基础配置类"""
    
    # 基础路径
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'mine_data')
    MODEL_DIR = os.path.join(DATA_DIR, 'models')
    
    # 数据库配置
    DATABASE_URL = os.path.join(DATA_DIR, 'articles.db')
    
    # Arxiv API配置
    ARXIV_API_URL = "http://export.arxiv.org/api/query"
    MAX_RESULTS_PER_QUERY = 100  # 每次查询最多获取50篇（提高限制）
    RETRAIN_THRESHOLD = 20  # TF-IDF重新训练阈值（新增文章数）
    DEFAULT_CATEGORIES = [
        'cs.AI',  # Artificial Intelligence
        'cs.LG',  # Machine Learning  
        'cs.CL',  # Computation and Language
        'cs.CV',  # Computer Vision
        'cs.IR',  # Information Retrieval
        'stat.ML', # Machine Learning (Statistics)
    ]
    
    # 推荐系统配置
    RECOMMENDATION_COUNT = 10  # 每次推荐的文章数量
    MIN_SIMILARITY_THRESHOLD = 0.3  # 最小相似度阈值
    USER_FEEDBACK_WEIGHT = 0.7  # 用户反馈权重
    CONTENT_WEIGHT = 0.3  # 内容相似度权重
    
    # 文本处理配置
    MAX_TEXT_LENGTH = 512  # BERT模型最大输入长度
    SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'  # 句子向量模型
    
    # Flask配置
    SECRET_KEY = 'your-secret-key-here'
    DEBUG = False
    HOST = '0.0.0.0'
    PORT = 5000
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    
class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    
# 根据环境变量选择配置
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """获取配置对象"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'production')
    return config[config_name]