"""
用户画像模块
构建和维护用户兴趣模型，基于用户设置和行为数据
"""

import numpy as np
import sqlite3
import json
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter

from config.settings import get_config
from core.text_analyzer import TextAnalyzer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserProfiler:
    """用户画像构建器"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.db_path = self.config.DATABASE_URL
        self.text_analyzer = TextAnalyzer(config)
    
    def create_user(self, username: str, interests: List[str]) -> int:
        """
        创建新用户
        
        Args:
            username: 用户名
            interests: 初始兴趣列表
        
        Returns:
            用户ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 将兴趣列表转换为JSON
        interests_json = json.dumps(interests)
        
        try:
            cursor.execute('''
                INSERT INTO users (username, interests)
                VALUES (?, ?)
            ''', (username, interests_json))
            
            user_id = cursor.lastrowid
            conn.commit()
            
            if user_id is None:
                raise ValueError("无法创建用户")
            
            logger.info(f"创建用户成功: {username} (ID: {user_id})")
            return user_id
            
        except sqlite3.IntegrityError:
            logger.error(f"用户名 {username} 已存在")
            raise ValueError(f"用户名 {username} 已存在")
        finally:
            conn.close()
    
    def get_user(self, username: str) -> Optional[Dict]:
        """
        获取用户信息
        
        Args:
            username: 用户名
        
        Returns:
            用户信息字典
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, interests, created_at, last_active
            FROM users
            WHERE username = ?
        ''', (username,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'username': row[1],
                'interests': json.loads(row[2]) if row[2] else [],
                'created_at': row[3],
                'last_active': row[4]
            }
        return None
    
    def update_user_interests(self, user_id: int, new_interests: List[str]):
        """
        更新用户兴趣
        
        Args:
            user_id: 用户ID
            new_interests: 新的兴趣列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        interests_json = json.dumps(new_interests)
        
        cursor.execute('''
            UPDATE users 
            SET interests = ?, last_active = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (interests_json, user_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"更新用户 {user_id} 的兴趣")
    
    def record_user_behavior(self, user_id: int, article_id: str, action_type: str):
        """
        记录用户行为，并在达到阈值时自动更新用户画像
        
        Args:
            user_id: 用户ID
            article_id: 文章ID
            action_type: 行为类型 ('click', 'like', 'dislike', 'save')
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_behaviors (user_id, article_id, action_type)
            VALUES (?, ?, ?)
        ''', (user_id, article_id, action_type))
        
        # 更新用户最后活跃时间
        cursor.execute('''
            UPDATE users 
            SET last_active = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"记录用户行为: 用户{user_id} {action_type} 文章{article_id}")
        
        # 检查是否需要更新用户画像和TF-IDF模型
        self._auto_update_user_profile(user_id)
    
    def get_user_behaviors(self, user_id: int, days: int = 30) -> List[Dict]:
        """
        获取用户行为历史
        
        Args:
            user_id: 用户ID
            days: 获取最近几天的行为
        
        Returns:
            行为记录列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 计算时间范围
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            SELECT ub.article_id, ub.action_type, ub.timestamp,
                   a.title, a.abstract, a.categories
            FROM user_behaviors ub
            JOIN articles a ON ub.article_id = a.id
            WHERE ub.user_id = ? AND ub.timestamp >= ?
            ORDER BY ub.timestamp DESC
        ''', (user_id, start_date))
        
        behaviors = []
        for row in cursor.fetchall():
            behaviors.append({
                'article_id': row[0],
                'action_type': row[1],
                'timestamp': row[2],
                'title': row[3],
                'abstract': row[4],
                'categories': row[5]
            })
        
        conn.close()
        return behaviors
    
    def build_user_interest_vector(self, user_id: int) -> np.ndarray:
        """
        构建用户兴趣向量
        
        Args:
            user_id: 用户ID
        
        Returns:
            用户兴趣向量
        """
        # 获取用户基础兴趣
        user_info = self.get_user_by_id(user_id)
        if not user_info:
            return np.array([])
        
        # 获取用户行为历史
        behaviors = self.get_user_behaviors(user_id, days=60)
        
        # 收集正面和负面行为的文章
        positive_texts = []
        negative_texts = []
        
        for behavior in behaviors:
            article_text = f"{behavior['title']} {behavior['abstract']}"
            
            if behavior['action_type'] in ['click', 'like', 'save']:
                positive_texts.append(article_text)
            elif behavior['action_type'] in ['dislike']:
                negative_texts.append(article_text)
        
        # 如果没有行为数据，使用初始兴趣
        if not positive_texts:
            initial_interests = user_info.get('interests', [])
            if initial_interests:
                positive_texts = [' '.join(initial_interests)]
        
        if not positive_texts:
            return np.array([])
        
        # 计算正面兴趣向量
        positive_embeddings = []
        for text in positive_texts:
            embedding = self.text_analyzer.get_sentence_embedding(text)
            if len(embedding) > 0:
                positive_embeddings.append(embedding)
        
        if not positive_embeddings:
            return np.array([])
        
        # 计算平均向量作为用户兴趣向量
        user_vector = np.mean(positive_embeddings, axis=0)
        
        # 如果有负面行为，减少负面内容的影响
        if negative_texts:
            negative_embeddings = []
            for text in negative_texts:
                embedding = self.text_analyzer.get_sentence_embedding(text)
                if len(embedding) > 0:
                    negative_embeddings.append(embedding)
            
            if negative_embeddings:
                negative_vector = np.mean(negative_embeddings, axis=0)
                # 减少负面影响（权重可调）
                user_vector = user_vector - 0.3 * negative_vector
        
        return user_vector
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """
        根据ID获取用户信息
        
        Args:
            user_id: 用户ID
        
        Returns:
            用户信息字典
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, interests, created_at, last_active
            FROM users
            WHERE id = ?
        ''', (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'username': row[1],
                'interests': json.loads(row[2]) if row[2] else [],
                'created_at': row[3],
                'last_active': row[4]
            }
        return None
    
    def get_user_preferred_categories(self, user_id: int) -> List[Tuple[str, float]]:
        """
        获取用户偏好的文章类别
        
        Args:
            user_id: 用户ID
        
        Returns:
            (类别, 偏好分数) 的列表
        """
        behaviors = self.get_user_behaviors(user_id, days=90)
        
        # 统计各类别的行为
        category_scores = defaultdict(float)
        
        for behavior in behaviors:
            categories = behavior['categories'].split(', ')
            
            # 不同行为的权重
            weight = 1.0
            if behavior['action_type'] == 'like':
                weight = 2.0
            elif behavior['action_type'] == 'save':
                weight = 3.0
            elif behavior['action_type'] == 'dislike':
                weight = -1.0
            
            for category in categories:
                category = category.strip()
                category_scores[category] += weight
        
        # 排序并返回
        sorted_categories = sorted(category_scores.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        return [(cat, score) for cat, score in sorted_categories if score > 0]
    
    def get_user_keywords(self, user_id: int, top_k: int = 20) -> List[str]:
        """
        提取用户感兴趣的关键词
        
        Args:
            user_id: 用户ID
            top_k: 返回前k个关键词
        
        Returns:
            关键词列表
        """
        behaviors = self.get_user_behaviors(user_id, days=60)
        
        # 收集正面行为的文章文本
        positive_texts = []
        for behavior in behaviors:
            if behavior['action_type'] in ['click', 'like', 'save']:
                article_text = f"{behavior['title']} {behavior['abstract']}"
                positive_texts.append(article_text)
        
        if not positive_texts:
            # 如果没有行为数据，返回初始兴趣
            user_info = self.get_user_by_id(user_id)
            if user_info:
                return user_info.get('interests', [])[:top_k]
            return []
        
        # 合并所有文本
        combined_text = ' '.join(positive_texts)
        
        # 提取关键词
        keywords = self.text_analyzer.extract_keywords(combined_text, top_k=top_k)
        
        return keywords
    
    def update_user_profile_from_behavior(self, user_id: int):
        """
        根据用户行为更新用户画像
        
        Args:
            user_id: 用户ID
        """
        # 获取用户关键词
        keywords = self.get_user_keywords(user_id, top_k=15)
        
        # 获取用户偏好类别
        preferred_categories = self.get_user_preferred_categories(user_id)
        top_categories = [cat for cat, score in preferred_categories[:5]]
        
        # 合并关键词和类别作为新的兴趣
        new_interests = keywords + top_categories
        
        # 去重并限制长度
        new_interests = list(dict.fromkeys(new_interests))[:20]
        
        # 更新用户兴趣
        self.update_user_interests(user_id, new_interests)
        
        logger.info(f"基于行为数据更新用户 {user_id} 的画像")
    
    def train_user_tfidf(self, user_id: int, min_behaviors: int = 10) -> bool:
        """
        为用户训练个性化TF-IDF模型
        基于用户喜欢的文章（点击/点赞/保存）
        
        Args:
            user_id: 用户ID
            min_behaviors: 最少需要的行为数量
        
        Returns:
            是否成功训练
        """
        import pickle
        import os
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        logger.info(f"开始为用户 {user_id} 训练个性化TF-IDF模型...")
        
        # 获取用户的正面行为
        behaviors = self.get_user_behaviors(user_id, days=90)
        
        # 收集用户喜欢的文章文本
        positive_texts = []
        for behavior in behaviors:
            if behavior['action_type'] in ['click', 'like', 'save']:
                article_text = f"{behavior['title']} {behavior['abstract']}"
                # 根据行为类型增加权重（通过重复文本）
                if behavior['action_type'] == 'like':
                    positive_texts.extend([article_text] * 2)  # 点赞权重x2
                elif behavior['action_type'] == 'save':
                    positive_texts.extend([article_text] * 3)  # 保存权重x3
                else:
                    positive_texts.append(article_text)  # 点击权重x1
        
        if len(positive_texts) < min_behaviors:
            logger.warning(f"用户 {user_id} 的行为数据不足（{len(positive_texts)} < {min_behaviors}），跳过训练")
            return False
        
        try:
            # 预处理文本
            processed_texts = [self.text_analyzer.preprocess_text(text) for text in positive_texts]
            
            # 训练TF-IDF向量化器
            vectorizer = TfidfVectorizer(
                max_features=1000,  # 用户模型可以小一些
                ngram_range=(1, 2),
                min_df=1,  # 用户数据较少，降低最小文档频率
                max_df=0.9
            )
            
            vectorizer.fit(processed_texts)
            
            # 保存模型
            model_dir = self.config.MODEL_DIR
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, f'user_{user_id}_tfidf.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            
            # 保存元数据
            meta_path = os.path.join(model_dir, f'user_{user_id}_tfidf_meta.json')
            meta = {
                'user_id': user_id,
                'behavior_count': len(behaviors),
                'training_samples': len(positive_texts),
                'training_date': datetime.now().isoformat(),
                'vocabulary_size': len(vectorizer.vocabulary_)
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            
            logger.info(f"✅ 用户 {user_id} 的TF-IDF模型训练成功")
            logger.info(f"   - 训练样本: {len(positive_texts)}")
            logger.info(f"   - 词汇量: {len(vectorizer.vocabulary_)}")
            logger.info(f"   - 保存位置: {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"训练用户 {user_id} 的TF-IDF模型失败: {e}")
            return False
    
    def load_user_tfidf(self, user_id: int):
        """
        加载用户的个性化TF-IDF模型
        
        Args:
            user_id: 用户ID
        
        Returns:
            TF-IDF向量化器，如果不存在则返回None
        """
        import pickle
        import os
        
        model_path = os.path.join(self.config.MODEL_DIR, f'user_{user_id}_tfidf.pkl')
        
        if not os.path.exists(model_path):
            return None
        
        try:
            with open(model_path, 'rb') as f:
                vectorizer = pickle.load(f)
            return vectorizer
        except Exception as e:
            logger.error(f"加载用户 {user_id} 的TF-IDF模型失败: {e}")
            return None
    
    def should_retrain_user_tfidf(self, user_id: int, threshold: int = 20) -> bool:
        """
        判断是否需要重新训练用户的TF-IDF模型
        
        Args:
            user_id: 用户ID
            threshold: 新增行为的阈值
        
        Returns:
            是否需要重新训练
        """
        import os
        
        meta_path = os.path.join(self.config.MODEL_DIR, f'user_{user_id}_tfidf_meta.json')
        
        # 如果模型不存在，需要训练
        if not os.path.exists(meta_path):
            return True
        
        try:
            # 读取元数据
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            trained_behavior_count = meta.get('behavior_count', 0)
            
            # 获取当前行为数量
            current_behaviors = self.get_user_behaviors(user_id, days=90)
            current_behavior_count = len(current_behaviors)
            
            # 判断是否增加了足够多的行为
            new_behaviors = current_behavior_count - trained_behavior_count
            
            if new_behaviors >= threshold:
                logger.info(f"用户 {user_id} 新增了 {new_behaviors} 次行为，需要重新训练TF-IDF模型")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"检查用户 {user_id} 的TF-IDF模型状态失败: {e}")
            return True  # 出错时重新训练
    
    def _auto_update_user_profile(self, user_id: int):
        """
        自动更新用户画像（内部方法）
        在用户行为达到特定阈值时触发
        
        Args:
            user_id: 用户ID
        """
        # 获取用户的正面行为数量
        behaviors = self.get_user_behaviors(user_id, days=90)
        positive_behaviors = [b for b in behaviors if b['action_type'] in ['click', 'like', 'save']]
        behavior_count = len(positive_behaviors)
        
        # 阈值1: 达到10次行为时，首次训练TF-IDF模型
        if behavior_count == 10:
            logger.info(f"用户 {user_id} 达到10次交互，开始训练TF-IDF模型")
            self.train_user_tfidf(user_id, min_behaviors=10)
            self.update_user_profile_from_behavior(user_id)
        
        # 阈值2: 每增加20次行为，重新训练模型和更新画像
        elif behavior_count > 10 and behavior_count % 20 == 0:
            logger.info(f"用户 {user_id} 达到{behavior_count}次交互，更新TF-IDF模型和用户画像")
            self.train_user_tfidf(user_id, min_behaviors=10)
            self.update_user_profile_from_behavior(user_id)
    
    def get_user_tfidf_score(self, user_id: int, article_text: str) -> float:
        """
        使用用户的个性化TF-IDF模型为文章打分
        
        Args:
            user_id: 用户ID
            article_text: 文章文本（标题+摘要）
        
        Returns:
            TF-IDF分数
        """
        # 加载模型（不再自动训练，由_auto_update_user_profile控制）
        vectorizer = self.load_user_tfidf(user_id)
        
        if vectorizer is None:
            # 如果没有模型，返回0分（说明用户交互次数不足）
            return 0.0
        
        try:
            # 预处理文本
            processed_text = self.text_analyzer.preprocess_text(article_text)
            
            # 计算TF-IDF向量
            tfidf_vector = vectorizer.transform([processed_text])
            
            # 使用向量的L2范数作为分数（表示与用户兴趣的匹配度）
            score = float(np.linalg.norm(tfidf_vector.toarray()))
            
            return score
            
        except Exception as e:
            logger.error(f"计算TF-IDF分数失败: {e}")
            return 0.0

# 使用示例
if __name__ == "__main__":
    profiler = UserProfiler()
    
    # 创建测试用户
    try:
        user_id = profiler.create_user("test_user", 
                                     ["machine learning", "deep learning", "computer vision"])
        print(f"创建用户成功，ID: {user_id}")
    except ValueError:
        # 用户已存在
        user_info = profiler.get_user("test_user")
        if user_info:
            user_id = user_info['id']
            print(f"用户已存在，ID: {user_id}")
        else:
            print("获取用户信息失败")
            exit(1)
    
    # 模拟用户行为
    profiler.record_user_behavior(user_id, "2310.12345", "click")
    profiler.record_user_behavior(user_id, "2310.12346", "like")
    
    # 获取用户兴趣向量
    interest_vector = profiler.build_user_interest_vector(user_id)
    print(f"用户兴趣向量维度: {len(interest_vector)}")
    
    # 获取用户关键词
    keywords = profiler.get_user_keywords(user_id)
    print(f"用户关键词: {keywords}")
