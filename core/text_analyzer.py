"""
文本分析模块
使用NLP技术对论文文本进行特征提取和语义分析
"""
import numpy as np
import pandas as pd
import sqlite3
import pickle
import os
import logging
from typing import List, Dict, Tuple, Optional
import re
import json

# NLP相关库
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import jieba

from config.settings import get_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextAnalyzer:
    """文本分析器"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.db_path = self.config.DATABASE_URL
        self.model_dir = self.config.MODEL_DIR
        
        # 初始化模型
        self.sentence_model = None
        self.tfidf_vectorizer = None
        self.load_or_create_models()
        
        # 下载NLTK数据
        self._download_nltk_data()
    
    def _download_nltk_data(self):
        """下载必要的NLTK数据"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def load_or_create_models(self):
        """加载或创建文本分析模型"""
        # 尝试加载Sentence-BERT模型
        try:
            logger.info("正在加载Sentence-BERT模型...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence-BERT模型加载成功!")
        except Exception as e:
            logger.warning(f"Sentence-BERT模型加载失败: {e}")
            logger.info("使用离线模式,跳过Sentence-BERT模型加载")
            self.sentence_model = None
        
        # 加载或创建TF-IDF向量化器（智能更新）
        self._load_or_update_tfidf_model()
    
    def _load_or_update_tfidf_model(self):
        """智能加载或更新TF-IDF模型"""
        tfidf_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
        tfidf_meta_path = os.path.join(self.model_dir, 'tfidf_meta.json')
        
        # 获取当前数据库中的文章数量
        current_article_count = self._get_article_count()
        
        # 检查是否存在已训练的模型
        if os.path.exists(tfidf_path) and os.path.exists(tfidf_meta_path):
            try:
                # 加载模型
                with open(tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                
                # 加载元数据
                with open(tfidf_meta_path, 'r') as f:
                    meta = json.load(f)
                    trained_count = meta.get('article_count', 0)
                
                # 判断是否需要重新训练
                article_increase = current_article_count - trained_count
                
                if article_increase >= 20:  # 新增20篇以上，重新训练
                    logger.info(f"检测到新增 {article_increase} 篇文章,重新训练TF-IDF模型...")
                    self.create_tfidf_model()
                elif article_increase > 0:
                    logger.info(f"TF-IDF模型加载成功(训练时: {trained_count}篇，当前: {current_article_count}篇，差距较小，暂不更新）")
                else:
                    logger.info("TF-IDF向量化器加载成功")
                
            except Exception as e:
                logger.error(f"加载TF-IDF向量化器失败: {e}")
                self.tfidf_vectorizer = None
                self.create_tfidf_model()
        else:
            # 模型不存在，创建新模型
            if current_article_count > 0:
                logger.info("TF-IDF模型不存在,正在创建...")
                self.create_tfidf_model()
            else:
                logger.warning("数据库为空,跳过TF-IDF模型创建")
                self.tfidf_vectorizer = None
    
    def _get_article_count(self) -> int:
        """获取数据库中的文章数量"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM articles")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            logger.error(f"获取文章数量失败: {e}")
            return 0
    
    def preprocess_text(self, text: str) -> str:
        """
        文本预处理
        
        Args:
            text: 原始文本
        
        Returns:
            预处理后的文本
        """
        if not text:
            return ""
        
        # 转换为小写
        text = text.lower()
        
        # 移除特殊字符，保留字母、数字和空格
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        
        # 移除停用词（英文）
        try:
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            words = text.split()
            text = ' '.join([word for word in words if word not in stop_words])
        except Exception as e:
            logger.warning(f"移除停用词时出错: {e}")
        
        return text.strip()
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        提取关键词
        
        Args:
            text: 文本内容
            top_k: 返回前k个关键词
        
        Returns:
            关键词列表
        """
        if not text:
            return []
        
        # 预处理文本
        processed_text = self.preprocess_text(text)
        
        # 使用TF-IDF提取关键词
        if self.tfidf_vectorizer:
            try:
                tfidf_matrix = self.tfidf_vectorizer.transform([processed_text])
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                
                # 获取TF-IDF分数
                scores = tfidf_matrix.toarray()[0]
                
                # 创建词汇-分数对
                word_scores = list(zip(feature_names, scores))
                
                # 按分数排序并取前k个
                word_scores.sort(key=lambda x: x[1], reverse=True)
                keywords = [word for word, score in word_scores[:top_k] if score > 0]
                
                return keywords
            except Exception as e:
                logger.error(f"TF-IDF关键词提取失败: {e}")
        
        # 简单的关键词提取（备用方案）
        words = processed_text.split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # 过滤短词
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]
    
    def create_tfidf_model(self):
        """创建TF-IDF模型"""
        logger.info("正在创建TF-IDF模型...")
        
        # 从数据库获取所有文章
        articles = self.get_all_articles_text()
        
        if not articles:
            logger.warning("没有找到文章数据,无法创建TF-IDF模型")
            return
        
        # 预处理所有文本
        processed_texts = [self.preprocess_text(article) for article in articles]
        
        # 创建TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # 训练模型
        self.tfidf_vectorizer.fit(processed_texts)
        
        # 保存模型
        os.makedirs(self.model_dir, exist_ok=True)
        tfidf_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
        with open(tfidf_path, 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        # 保存元数据
        tfidf_meta_path = os.path.join(self.model_dir, 'tfidf_meta.json')
        meta = {
            'article_count': len(articles),
            'training_date': pd.Timestamp.now().isoformat(),
            'vocabulary_size': len(self.tfidf_vectorizer.vocabulary_)
        }
        with open(tfidf_meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f" TF-IDF模型训练完成")
        logger.info(f"   - 训练样本: {len(articles)} 篇文章")
        logger.info(f"   - 词汇量: {len(self.tfidf_vectorizer.vocabulary_)} 个词")
        logger.info(f"   - 保存位置: {tfidf_path}")
    
    def get_all_articles_text(self) -> List[str]:
        """从数据库获取所有文章文本"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT title, abstract FROM articles")
        rows = cursor.fetchall()
        conn.close()
        
        # 合并标题和摘要
        texts = []
        for title, abstract in rows:
            text = f"{title} {abstract}"
            texts.append(text)
        
        return texts
    
    def get_sentence_embedding(self, text: str) -> np.ndarray:
        """
        获取文本的句子嵌入向量
        
        Args:
            text: 输入文本
        
        Returns:
            向量表示
        """
        if not text:
            return np.array([])
        
        if not self.sentence_model:
            # 如果模型未加载，返回随机向量作为占位符
            logger.warning("Sentence模型未加载,使用随机向量")
            return np.random.rand(384)  # all-MiniLM-L6-v2的向量维度
        
        try:
            # 限制文本长度
            if len(text) > self.config.MAX_TEXT_LENGTH:
                text = text[:self.config.MAX_TEXT_LENGTH]
            
            embedding = self.sentence_model.encode(text, show_progress_bar=False)
            return embedding
        except Exception as e:
            logger.error(f"获取句子嵌入失败: {e}")
            return np.array([])
    
    def get_tfidf_vector(self, text: str) -> np.ndarray:
        """
        获取文本的TF-IDF向量
        
        Args:
            text: 输入文本
        
        Returns:
            TF-IDF向量
        """
        if not text or not self.tfidf_vectorizer:
            return np.array([])
        
        try:
            processed_text = self.preprocess_text(text)
            tfidf_vector = self.tfidf_vectorizer.transform([processed_text])
            return tfidf_vector.toarray()[0]
        except Exception as e:
            logger.error(f"获取TF-IDF向量失败: {e}")
            return np.array([])
    

    def analyze_article(self, article: Dict) -> Dict:
        """
        分析单篇文章，提取特征
        
        Args:
            article: 文章字典
        
        Returns:
            分析结果
        """
        # 合并标题和摘要
        full_text = f"{article.get('title', '')} {article.get('abstract', '')}"
        
        # 提取关键词
        keywords = self.extract_keywords(full_text)
        
        # 获取嵌入向量
        sentence_embedding = self.get_sentence_embedding(full_text)
        tfidf_vector = self.get_tfidf_vector(full_text)
        
        # 分析结果
        analysis_result = {
            'article_id': article.get('id'),
            'keywords': keywords,
            'sentence_embedding': sentence_embedding.tolist() if len(sentence_embedding) > 0 else [],
            'tfidf_vector': tfidf_vector.tolist() if len(tfidf_vector) > 0 else [],
            'text_length': len(full_text),
            'title_length': len(article.get('title', '')),
            'abstract_length': len(article.get('abstract', ''))
        }
        
        return analysis_result
    
    def batch_analyze_articles(self, limit: int = None) -> int:
        """
        批量分析文章并更新数据库
        
        Args:
            limit: 限制处理数量
        
        Returns:
            处理的文章数量
        """
        logger.info("开始批量分析文章...")
        
        # 获取需要分析的文章
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM articles WHERE embedding_vector IS NULL"
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        
        processed_count = 0
        
        for row in cursor.fetchall():
            article = dict(zip(columns, row))
            
            try:
                # 分析文章
                analysis = self.analyze_article(article)
                
                # 更新数据库
                embedding_json = json.dumps({
                    'sentence_embedding': analysis['sentence_embedding'],
                    'keywords': analysis['keywords']
                })
                
                update_cursor = conn.cursor()
                update_cursor.execute(
                    "UPDATE articles SET embedding_vector = ? WHERE id = ?",
                    (embedding_json, article['id'])
                )
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    logger.info(f"已处理 {processed_count} 篇文章")
                    conn.commit()
                
            except Exception as e:
                logger.error(f"分析文章 {article['id']} 失败: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"批量分析完成，共处理 {processed_count} 篇文章")
        return processed_count

# 使用示例
if __name__ == "__main__":
    analyzer = TextAnalyzer()
    
    # 创建TF-IDF模型
    analyzer.create_tfidf_model()
    
    # 批量分析文章
    count = analyzer.batch_analyze_articles(limit=50)
    print(f"分析了 {count} 篇文章")
    