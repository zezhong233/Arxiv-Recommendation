"""
Arxiv数据收集模块
负责从arxiv API获取论文数据并存储到数据库
"""

import arxiv
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import re

from config.settings import get_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivDataCollector:
    """Arxiv数据收集器"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.db_path = self.config.DATABASE_URL
        self._db_initialized = False  # 数据库初始化标记
    
    def _ensure_database_initialized(self):
        """确保数据库已初始化（延迟初始化，只执行一次）"""
        if not self._db_initialized:
            self.init_database()
            self._db_initialized = True
        
    def init_database(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path) ## connetion 对象
        cursor = conn.cursor() ## cursor对象，connection的执行者
        
        # 创建文章表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT NOT NULL,
                abstract TEXT NOT NULL,
                categories TEXT NOT NULL,
                published_date TEXT NOT NULL,
                updated_date TEXT NOT NULL,
                pdf_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding_vector TEXT  -- 存储文章的向量表示
            )
        ''')
        
        # 创建用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                interests TEXT,  -- JSON格式存储用户兴趣
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建用户行为表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_behaviors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                article_id TEXT,
                action_type TEXT,  -- 'click', 'like', 'dislike', 'save'
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (article_id) REFERENCES articles (id)
            )
        ''')
        
        # 创建推荐记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                article_id TEXT,
                score REAL,
                algorithm_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (article_id) REFERENCES articles (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("数据库初始化完成")
    
    def clean_text(self, text: str) -> str:
        """清理文本数据"""
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符但保留基本标点
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        return text.strip()
    
    def fetch_recent_articles(self, days: int = 3, categories: List[str] = None, max_articles: int = None) -> List[Dict]:
        """
        获取最近几天的文章
        
        Args:
            days: 获取最近几天的文章
            categories: 指定类别,如果为None则使用默认类别
            max_articles: 最多获取的文章数量,None表示不限制
        
        Returns:
            文章列表
        """
        self._ensure_database_initialized()  # 确保数据库已初始化
        
        if categories is None:
            categories = self.config.DEFAULT_CATEGORIES
        
        # 如果没有指定max_articles，使用配置中的MAX_RESULTS_PER_QUERY
        if max_articles is None:
            max_articles = self.config.MAX_RESULTS_PER_QUERY
        
        articles = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for category in categories:
            logger.info(f"正在获取类别 {category} 的文章...")
            
            # 如果已经获取足够的文章，跳出循环
            if len(articles) >= max_articles:
                break
            
            try:
                # 构建查询
                # 注意：arxiv库会自动分页，每页100条
                search = arxiv.Search(
                    query=f"cat:{category}",
                    max_results=max_articles,  # 设置总共要获取的文章数
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                count = 0
                for paper in search.results():
                    # 检查是否已达到目标数量
                    if len(articles) >= max_articles:
                        break
                    
                    # 检查日期范围
                    if paper.published.replace(tzinfo=None) < start_date:
                        break
                    
                    article_data = {
                        'id': paper.entry_id.split('/')[-1],  # 提取arxiv ID
                        'title': self.clean_text(paper.title),
                        'authors': ', '.join([author.name for author in paper.authors]),
                        'abstract': self.clean_text(paper.summary),
                        'categories': ', '.join(paper.categories),
                        'published_date': paper.published.strftime('%Y-%m-%d %H:%M:%S'),
                        'updated_date': paper.updated.strftime('%Y-%m-%d %H:%M:%S'),
                        'pdf_url': paper.pdf_url
                    }
                    
                    articles.append(article_data)
                    count += 1

                
                logger.info(f"  从类别 {category} 获取了 {count} 篇文章")
                
                # 避免过于频繁的API调用
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"获取类别 {category} 的文章时出错: {e}")
                continue
        
        logger.info(f"总共获取到 {len(articles)} 篇文章")
        return articles
    
    def save_articles(self, articles: List[Dict]) -> int:
        """
        保存文章到数据库
        
        Args:
            articles: 文章列表
        
        Returns:
            成功保存的文章数量
        """
        self._ensure_database_initialized()  # 确保数据库已初始化
        
        if not articles:
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取当前时间作为 fetched_at
        from datetime import datetime
        fetched_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        saved_count = 0
        for article in articles:
            try:
                # 检查文章是否已存在
                cursor.execute('SELECT created_at FROM articles WHERE id = ?', (article['id'],))
                existing = cursor.fetchone()
                
                if existing:
                    # 文章已存在，只更新内容和 fetched_at，保留原 created_at
                    cursor.execute('''
                        UPDATE articles 
                        SET title = ?, authors = ?, abstract = ?, categories = ?,
                            published_date = ?, updated_date = ?, pdf_url = ?, fetched_at = ?
                        WHERE id = ?
                    ''', (
                        article['title'],
                        article['authors'],
                        article['abstract'],
                        article['categories'],
                        article['published_date'],
                        article['updated_date'],
                        article['pdf_url'],
                        fetched_time,
                        article['id']
                    ))
                else:
                    # 新文章，插入时 created_at 和 fetched_at 都设为当前时间
                    cursor.execute('''
                        INSERT INTO articles 
                        (id, title, authors, abstract, categories, published_date, updated_date, pdf_url, created_at, fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        article['id'],
                        article['title'],
                        article['authors'],
                        article['abstract'],
                        article['categories'],
                        article['published_date'],
                        article['updated_date'],
                        article['pdf_url'],
                        fetched_time,  # created_at
                        fetched_time   # fetched_at
                    ))
                
                saved_count += 1
            except Exception as e:
                logger.error(f"保存文章 {article['id']} 时出错: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"成功保存 {saved_count} 篇文章到数据库 (fetched_at: {fetched_time})")
        return saved_count
    
    def get_articles_by_keywords(self, keywords: List[str], max_results: int = 50) -> List[Dict]:
        """
        根据关键词搜索文章
        
        Args:
            keywords: 关键词列表
            max_results: 最大结果数
        
        Returns:
            文章列表
        """
        self._ensure_database_initialized()  # 确保数据库已初始化
        
        # 构建查询字符串
        query_parts = []
        for keyword in keywords:
            query_parts.append(f'all:"{keyword}"')
        
        query = ' OR '.join(query_parts)
        
        logger.info(f"搜索关键词: {keywords}")
        
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            articles = []
            for paper in search.results():
                article_data = {
                    'id': paper.entry_id.split('/')[-1],
                    'title': self.clean_text(paper.title),
                    'authors': ', '.join([author.name for author in paper.authors]),
                    'abstract': self.clean_text(paper.summary),
                    'categories': ', '.join(paper.categories),
                    'published_date': paper.published.strftime('%Y-%m-%d %H:%M:%S'),
                    'updated_date': paper.updated.strftime('%Y-%m-%d %H:%M:%S'),
                    'pdf_url': paper.pdf_url
                }
                articles.append(article_data)
            
            logger.info(f"根据关键词搜索到 {len(articles)} 篇文章")
            return articles
            
        except Exception as e:
            logger.error(f"搜索文章时出错: {e}")
            return []
    
    def get_articles_from_db(self, limit: int = 100, category: str = None) -> List[Dict]:
        """
        从数据库获取文章
        
        Args:
            limit: 限制数量
            category: 指定类别
        
        Returns:
            文章列表
        """
        self._ensure_database_initialized()  # 确保数据库已初始化
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if category:
            cursor.execute('''
                SELECT * FROM articles 
                WHERE categories LIKE ? 
                ORDER BY published_date DESC 
                LIMIT ?
            ''', (f'%{category}%', limit))
        else:
            cursor.execute('''
                SELECT * FROM articles 
                ORDER BY published_date DESC 
                LIMIT ?
            ''', (limit,))
        
        columns = [description[0] for description in cursor.description]
        articles = []
        
        for row in cursor.fetchall():
            article = dict(zip(columns, row))
            articles.append(article)
        
        conn.close()
        return articles
    
    def get_last_update_time(self) -> Optional[datetime]:
        """
        获取数据库中最新文章的入库时间（即上次更新时间）
        
        Returns:
            最后更新时间,如果数据库为空则返回None
        """
        self._ensure_database_initialized()  # 确保数据库已初始化
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT created_at FROM articles 
                ORDER BY created_at DESC 
                LIMIT 1
            ''')
            result = cursor.fetchone()
            
            if result:
                # 解析时间字符串
                time_str = result[0]
                return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            return None
            
        except Exception as e:
            logger.error(f"获取最后更新时间出错: {e}")
            return None
        finally:
            conn.close()
    
    def get_days_since_last_update(self) -> Optional[int]:
        """
        计算距离上次更新的天数
        
        Returns:
            天数,如果数据库为空则返回None
        """
        last_update = self.get_last_update_time()
        if last_update is None:
            return None
        
        days_diff = (datetime.now() - last_update).days
        return days_diff
    
    def update_database(self, days: int = 1):
        """
        更新数据库（获取最新文章）
        
        Args:
            days: 获取最近几天的文章
        """
        logger.info(f"开始更新数据库，获取最近 {days} 天的文章...")
        
        # 获取新文章
        new_articles = self.fetch_recent_articles(days=days)
        
        # 保存到数据库
        saved_count = self.save_articles(new_articles)
        
        logger.info(f"数据库更新完成，新增 {saved_count} 篇文章")
        return saved_count

# 使用示例
if __name__ == "__main__":
    collector = ArxivDataCollector()
    
    # 更新数据库
    collector.update_database(days=3)
    
    # 根据关键词搜索
    articles = collector.get_articles_by_keywords(['machine learning', 'deep learning'])
    print(f"搜索到 {len(articles)} 篇相关文章")
    
    # 从数据库获取文章
    db_articles = collector.get_articles_from_db(limit=10)
    print(f"数据库中有 {len(db_articles)} 篇文章")