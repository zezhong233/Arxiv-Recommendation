"""
推荐算法模块
实现多种推荐策略的混合推荐系统
"""

import numpy as np
import sqlite3
import json
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import random

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from config.settings import get_config
from core.text_analyzer import TextAnalyzer
from core.user_profiler import UserProfiler

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationEngine:
    """推荐引擎"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.db_path = self.config.DATABASE_URL
        self.text_analyzer = TextAnalyzer(config)
        self.user_profiler = UserProfiler(config)
    
    def get_user_behavior_count(self, user_id: int) -> int:
        """
        获取用户的行为记录数量
        
        Args:
            user_id: 用户ID
        
        Returns:
            行为记录数量
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM user_behaviors
            WHERE user_id = ? AND action_type IN ('click', 'like', 'save')
        ''', (user_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def content_based_recommend(self, user_id: int, limit: int = 20) -> List[Tuple[str, float]]:
        """
        基于内容的推荐
        动态权重策略：
        - 用户交互 < 10次,只用 Sentence-BERT (权重1.0)
        - 用户交互 >= 10次,TF-IDF(0.4) + Sentence-BERT(0.6)
        
        Args:
            user_id: 用户ID
            limit: 推荐数量限制
        
        Returns:
            (文章ID, 综合分数) 的列表
        """
        logger.info(f"为用户 {user_id} 生成基于内容的推荐...")
        
        # 检查用户行为数量，决定推荐策略
        behavior_count = self.get_user_behavior_count(user_id)
        use_tfidf = behavior_count >= 10
        
        if use_tfidf:
            logger.info(f"用户有 {behavior_count} 次交互记录，使用混合推荐策略 (TF-IDF 0.4 + Sentence 0.6)")
        else:
            logger.info(f"用户只有 {behavior_count} 次交互记录，仅使用 Sentence-BERT 推荐")
        
        # 获取用户兴趣向量 (Sentence-BERT)
        user_vector = self.user_profiler.build_user_interest_vector(user_id)
        
        if len(user_vector) == 0:
            logger.warning(f"用户 {user_id} 没有有效的兴趣向量")
            return []
        
        # 获取候选文章
        candidate_articles = self.get_candidate_articles(user_id)
        
        if not candidate_articles:
            logger.warning("没有找到候选文章")
            return []
        
        logger.info(f"开始为 {len(candidate_articles)} 篇候选文章打分...")
        
        recommendations = []
        
        for article in candidate_articles:
            article_text = f"{article['title']} {article['abstract']}"
            
            # 1. Sentence-BERT 语义相似度
            article_vector = self.text_analyzer.get_sentence_embedding(article_text)
            
            if len(article_vector) == 0:
                continue
            
            semantic_similarity = cosine_similarity(
                user_vector.reshape(1, -1), 
                article_vector.reshape(1, -1)
            )[0][0]
            
            # 2. 根据用户交互数量决定是否使用 TF-IDF
            if use_tfidf:
                # 用户个性化 TF-IDF 分数
                tfidf_score = self.user_profiler.get_user_tfidf_score(user_id, article_text)
                
                # 归一化 TF-IDF 分数到 [0, 1] 范围
                normalized_tfidf = min(tfidf_score / 5.0, 1.0)
                
                # 混合评分: TF-IDF(0.4) + Sentence(0.6)
                combined_score = 0.6 * semantic_similarity + 0.4 * normalized_tfidf
            else:
                # 只使用 Sentence-BERT: 权重1.0
                combined_score = semantic_similarity
            
            # 只推荐分数超过阈值的文章
            if combined_score > self.config.MIN_SIMILARITY_THRESHOLD:
                recommendations.append((article['id'], float(combined_score)))
        
        # 按综合分数排序
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"生成了 {len(recommendations)} 个推荐（过滤前: {len(candidate_articles)}")
        
        return recommendations[:limit]
    
    def hybrid_recommend(self, user_id: int, limit: int = None) -> List[Dict]:
        """
        混合推荐算法
        当前只使用基于内容的推荐（单用户模式）
        
        Args:
            user_id: 用户ID
            limit: 推荐数量限制
        
        Returns:
            推荐文章列表
        """
        if limit is None:
            limit = self.config.RECOMMENDATION_COUNT
        
        logger.info(f"为用户 {user_id} 生成推荐")
        
        # 只使用内容推荐（单用户模式，不需要协同过滤）
        content_recs = self.content_based_recommend(user_id, limit)
        
        # 获取文章详情
        final_recommendations = []
        for article_id, score in content_recs:
            article_info = self.get_article_info(article_id)
            if article_info:
                article_info['recommendation_score'] = score
                article_info['algorithm_type'] = 'content_based'
                final_recommendations.append(article_info)
        
        # 记录推荐结果
        self.save_recommendations(user_id, final_recommendations)
        
        logger.info(f"为用户 {user_id} 生成了 {len(final_recommendations)} 个推荐")
        
        return final_recommendations
    
    def user_has_interacted(self, user_id: int, article_id: str) -> bool:
        """
        检查用户是否已经与文章交互过
        
        Args:
            user_id: 用户ID
            article_id: 文章ID
        
        Returns:
            是否已交互
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM user_behaviors
            WHERE user_id = ? AND article_id = ?
        ''', (user_id, article_id))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def get_candidate_articles(self, user_id: int, days: int = 7) -> List[Dict]:
        """
        获取候选文章（排除用户已交互过的）
        策略：优先推荐最近一次fetch的文章，用7天内fetch的文章补充
        
        Args:
            user_id: 用户ID
            days: 补充文章的时间范围（天）
        
        Returns:
            候选文章列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取用户已交互的文章ID
        cursor.execute('''
            SELECT DISTINCT article_id FROM user_behaviors
            WHERE user_id = ?
        ''', (user_id,))
        
        interacted_articles = {row[0] for row in cursor.fetchall()}
        
        # 第一步：获取最近一次fetch的文章（优先推荐）
        cursor.execute('''
            SELECT MAX(fetched_at) FROM articles
        ''')
        latest_fetch = cursor.fetchone()[0]
        
        articles = []
        
        if latest_fetch:
            logger.info(f"最近一次fetch时间: {latest_fetch}")
            
            # 获取最近一次fetch的所有文章
            cursor.execute('''
                SELECT id, title, abstract, categories, published_date, fetched_at
                FROM articles
                WHERE fetched_at = ?
                ORDER BY published_date DESC
            ''', (latest_fetch,))
            
            for row in cursor.fetchall():
                article_id = row[0]
                if article_id not in interacted_articles:
                    articles.append({
                        'id': article_id,
                        'title': row[1],
                        'abstract': row[2],
                        'categories': row[3],
                        'published_date': row[4],
                        'fetched_at': row[5],
                        'priority': 'latest'  # 标记为最新fetch的文章
                    })
            
            logger.info(f"最近一次fetch获得 {len(articles)} 篇候选文章")
        
        # 第二步：如果最近一次fetch的文章不够，用7天内的文章补充
        if len(articles) < 50:  # 如果候选文章少于50篇
            logger.info(f"候选文章不足，用最近{days}天内fetch的文章补充...")
            
            # 计算7天前的时间
            seven_days_ago = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
            
            # 获取7天内fetch的文章（排除最近一次的）
            cursor.execute('''
                SELECT id, title, abstract, categories, published_date, fetched_at
                FROM articles
                WHERE fetched_at >= ? AND fetched_at != ?
                ORDER BY fetched_at DESC, published_date DESC
                LIMIT 200
            ''', (seven_days_ago, latest_fetch if latest_fetch else ''))
            
            supplementary_count = 0
            for row in cursor.fetchall():
                article_id = row[0]
                if article_id not in interacted_articles:
                    articles.append({
                        'id': article_id,
                        'title': row[1],
                        'abstract': row[2],
                        'categories': row[3],
                        'published_date': row[4],
                        'fetched_at': row[5],
                        'priority': 'supplementary'  # 标记为补充文章
                    })
                    supplementary_count += 1
            
            logger.info(f"补充了 {supplementary_count} 篇文章，总候选数: {len(articles)}")
        
        conn.close()
        
        # 按优先级和时间排序：最新fetch的在前
        articles.sort(key=lambda x: (
            0 if x.get('priority') == 'latest' else 1,  # latest优先
            x.get('fetched_at', ''),  # 然后按fetch时间
            x.get('published_date', '')  # 最后按发表时间
        ), reverse=True)
        
        logger.info(f"最终返回 {len(articles)} 篇候选文章")
        return articles
    
    def get_article_info(self, article_id: str) -> Optional[Dict]:
        """
        获取文章详细信息
        
        Args:
            article_id: 文章ID
        
        Returns:
            文章信息字典
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, title, authors, abstract, categories, 
                   published_date, updated_date, pdf_url
            FROM articles
            WHERE id = ?
        ''', (article_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'title': row[1],
                'authors': row[2],
                'abstract': row[3],
                'categories': row[4],
                'published_date': row[5],
                'updated_date': row[6],
                'pdf_url': row[7]
            }
        
        return None
    
    def save_recommendations(self, user_id: int, recommendations: List[Dict]):
        """
        保存推荐结果到数据库
        
        Args:
            user_id: 用户ID
            recommendations: 推荐结果列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for rec in recommendations:
            cursor.execute('''
                INSERT INTO recommendations (user_id, article_id, score, algorithm_type)
                VALUES (?, ?, ?, ?)
            ''', (
                user_id,
                rec['id'],
                rec['recommendation_score'],
                rec['algorithm_type']
            ))
        
        conn.commit()
        conn.close()

# 使用示例
if __name__ == "__main__":
    engine = RecommendationEngine()
    
    # 假设有用户ID为1
    user_id = 1
    
    # 生成推荐
    recommendations = engine.content_based_recommend(user_id, limit=10)
    
    print(f"为用户 {user_id} 生成了 {len(recommendations)} 个推荐:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title'][:60]}... (分数: {rec['recommendation_score']:.3f})")
