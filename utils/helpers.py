"""
工具函数模块
提供各种通用的辅助功能
"""

import logging
import re
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import sqlite3

from config.settings import get_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseUtils:
    """数据库工具类"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.db_path = self.config.DATABASE_URL
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """
        执行查询并返回结果
        
        Args:
            query: SQL查询语句
            params: 查询参数
        
        Returns:
            查询结果列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # 获取列名
            columns = [description[0] for description in cursor.description]
            
            # 获取数据
            rows = cursor.fetchall()
            
            # 转换为字典列表
            results = []
            for row in rows:
                results.append(dict(zip(columns, row)))
            
            return results
            
        except Exception as e:
            logger.error(f"数据库查询失败: {e}")
            return []
        finally:
            conn.close()
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """
        执行更新操作
        
        Args:
            query: SQL更新语句
            params: 更新参数
        
        Returns:
            影响的行数
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            conn.commit()
            return cursor.rowcount
            
        except Exception as e:
            logger.error(f"数据库更新失败: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()
    
    def get_database_stats(self) -> Dict[str, int]:
        """
        获取数据库统计信息
        
        Returns:
            统计信息字典
        """
        stats = {}
        
        # 文章统计
        articles = self.execute_query("SELECT COUNT(*) as count FROM articles")
        stats['articles'] = articles[0]['count'] if articles else 0
        
        # 用户统计
        users = self.execute_query("SELECT COUNT(*) as count FROM users")
        stats['users'] = users[0]['count'] if users else 0
        
        # 行为统计
        behaviors = self.execute_query("SELECT COUNT(*) as count FROM user_behaviors")
        stats['behaviors'] = behaviors[0]['count'] if behaviors else 0
        
        # 推荐统计
        recommendations = self.execute_query("SELECT COUNT(*) as count FROM recommendations")
        stats['recommendations'] = recommendations[0]['count'] if recommendations else 0
        
        return stats
    
    def cleanup_old_data(self, days: int = 20):
        """
        清理旧数据
        
        Args:
            days: 保留最近几天的数据
        """
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # 清理旧的推荐记录
        deleted_recs = self.execute_update(
            "DELETE FROM recommendations WHERE created_at < ?", 
            (cutoff_date,)
        )
        
        # 清理旧的行为记录（保留更长时间）
        behavior_cutoff = (datetime.now() - timedelta(days=days*2)).strftime('%Y-%m-%d')
        deleted_behaviors = self.execute_update(
            "DELETE FROM user_behaviors WHERE timestamp < ?", 
            (behavior_cutoff,)
        )
        
        logger.info(f"清理完成：删除 {deleted_recs} 条推荐记录，{deleted_behaviors} 条行为记录")

class TextUtils:
    """文本处理工具类"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        清理文本
        
        Args:
            text: 原始文本
        
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除特殊字符但保留基本标点
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]]', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_arxiv_id(text: str) -> Optional[str]:
        """
        从文本中提取arxiv ID
        
        Args:
            text: 包含arxiv ID的文本
        
        Returns:
            arxiv ID 或 None
        """
        # 匹配arxiv ID格式
        patterns = [
            r'arxiv:(\d{4}\.\d{4,5})',
            r'(\d{4}\.\d{4,5})',
            r'arxiv\.org/abs/(\d{4}\.\d{4,5})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """
        截断文本
        
        Args:
            text: 原始文本
            max_length: 最大长度
            suffix: 后缀
        
        Returns:
            截断后的文本
        """
        if not text or len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def highlight_keywords(text: str, keywords: List[str]) -> str:
        """
        高亮关键词
        
        Args:
            text: 原始文本
            keywords: 关键词列表
        
        Returns:
            高亮后的HTML文本
        """
        highlighted_text = text
        
        for keyword in keywords:
            if keyword:
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                highlighted_text = pattern.sub(
                    f'<mark class="highlight">{keyword}</mark>', 
                    highlighted_text
                )
        
        return highlighted_text

class ValidationUtils:
    """验证工具类"""
    
    @staticmethod
    def validate_username(username: str) -> tuple[bool, str]:
        """
        验证用户名
        
        Args:
            username: 用户名
        
        Returns:
            (是否有效, 错误信息)
        """
        if not username:
            return False, "用户名不能为空"
        
        if len(username) < 2:
            return False, "用户名至少需要2个字符"
        
        if len(username) > 50:
            return False, "用户名不能超过50个字符"
        
        # 检查字符
        if not re.match(r'^[a-zA-Z0-9\u4e00-\u9fa5_-]+$', username):
            return False, "用户名只能包含字母、数字、中文、下划线和连字符"
        
        return True, ""
    
    @staticmethod
    def validate_interests(interests: List[str]) -> tuple[bool, str]:
        """
        验证兴趣列表
        
        Args:
            interests: 兴趣列表
        
        Returns:
            (是否有效, 错误信息)
        """
        if not interests:
            return False, "至少需要设置一个兴趣"
        
        if len(interests) > 20:
            return False, "兴趣不能超过20个"
        
        for interest in interests:
            if not interest.strip():
                return False, "兴趣不能为空"
            
            if len(interest) > 100:
                return False, "单个兴趣不能超过100个字符"
        
        return True, ""

class CacheUtils:
    """缓存工具类"""
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or os.path.join(get_config().DATA_DIR, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_cache_path(self, key: str) -> str:
        """获取缓存文件路径"""
        safe_key = re.sub(r'[^\w\-_]', '_', key)
        return os.path.join(self.cache_dir, f"{safe_key}.cache")
    
    def set_cache(self, key: str, data: Any, ttl: int = 3600) -> bool:
        """
        设置缓存
        
        Args:
            key: 缓存键
            data: 缓存数据
            ttl: 生存时间（秒）
        
        Returns:
            是否成功
        """
        try:
            cache_data = {
                'data': data,
                'timestamp': datetime.now().timestamp(),
                'ttl': ttl
            }
            
            cache_path = self.get_cache_path(key)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, default=str)
            
            return True
        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
            return False
    
    def get_cache(self, key: str) -> Optional[Any]:
        """
        获取缓存
        
        Args:
            key: 缓存键
        
        Returns:
            缓存数据或None
        """
        try:
            cache_path = self.get_cache_path(key)
            
            if not os.path.exists(cache_path):
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 检查是否过期
            timestamp = cache_data['timestamp']
            ttl = cache_data['ttl']
            
            if datetime.now().timestamp() - timestamp > ttl:
                os.remove(cache_path)
                return None
            
            return cache_data['data']
            
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
            return None
    
    def clear_cache(self, pattern: str = None):
        """
        清理缓存
        
        Args:
            pattern: 匹配模式，如果为None则清理所有
        """
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    if pattern is None or pattern in filename:
                        os.remove(os.path.join(self.cache_dir, filename))
            
            logger.info("缓存清理完成")
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")

class DateUtils:
    """日期工具类"""
    
    @staticmethod
    def format_relative_time(date_str: str) -> str:
        """
        格式化相对时间
        
        Args:
            date_str: 日期字符串
        
        Returns:
            相对时间描述
        """
        try:
            date = datetime.strptime(date_str[:19], '%Y-%m-%d %H:%M:%S')
            now = datetime.now()
            diff = now - date
            
            if diff.days > 30:
                return f"{diff.days // 30}个月前"
            elif diff.days > 0:
                return f"{diff.days}天前"
            elif diff.seconds > 3600:
                return f"{diff.seconds // 3600}小时前"
            elif diff.seconds > 60:
                return f"{diff.seconds // 60}分钟前"
            else:
                return "刚刚"
        except:
            return date_str[:10]
    
    @staticmethod
    def get_date_range(days: int) -> tuple[str, str]:
        """
        获取日期范围
        
        Args:
            days: 天数
        
        Returns:
            (开始日期, 结束日期)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return (
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

# 单例实例
db_utils = DatabaseUtils()
cache_utils = CacheUtils()

# 使用示例
if __name__ == "__main__":
    # 测试数据库工具
    stats = db_utils.get_database_stats()
    print(f"数据库统计: {stats}")
    
    # 测试文本工具
    text = "  这是一个测试文本<script>alert('test')</script>  "
    cleaned = TextUtils.clean_text(text)
    print(f"清理后的文本: {cleaned}")
    
    # 测试验证工具
    valid, msg = ValidationUtils.validate_username("test_user")
    print(f"用户名验证: {valid}, {msg}")
    
    # 测试缓存工具
    cache_utils.set_cache("test_key", {"data": "test"}, ttl=60)
    cached_data = cache_utils.get_cache("test_key")
    print(f"缓存数据: {cached_data}")