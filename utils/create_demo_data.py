"""
创建演示数据脚本
为了在没有网络连接的情况下演示系统功能
"""

import sys
import os
import sqlite3
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_config

def create_demo_data():
    """创建演示数据"""
    config = get_config()
    
    # 确保数据目录存在
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    conn = sqlite3.connect(config.DATABASE_URL)
    cursor = conn.cursor()
    
    # 创建表（如果不存在）
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
            embedding_vector TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            interests TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_behaviors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            article_id TEXT,
            action_type TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id),
            FOREIGN KEY (article_id) REFERENCES articles (id)
        )
    ''')
    
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
    
    # 插入演示文章数据
    demo_articles = [
        {
            'id': '2310.12345',
            'title': 'Attention Is All You Need: A Comprehensive Survey of Transformer Models',
            'authors': 'John Smith, Jane Doe, Bob Wilson',
            'abstract': 'This paper provides a comprehensive survey of Transformer architectures and their applications in natural language processing. We review the evolution from the original Transformer to modern variants including BERT, GPT, and T5. Our analysis covers architectural innovations, training methodologies, and performance benchmarks across various NLP tasks.',
            'categories': 'cs.LG, cs.CL',
            'published_date': '2023-10-12 14:30:00',
            'updated_date': '2023-10-12 14:30:00',
            'pdf_url': 'https://arxiv.org/pdf/2310.12345.pdf'
        },
        {
            'id': '2310.12346',
            'title': 'Deep Reinforcement Learning for Autonomous Navigation: Recent Advances and Challenges',
            'authors': 'Alice Johnson, Charlie Brown, David Lee',
            'abstract': 'This work presents a comprehensive overview of deep reinforcement learning approaches for autonomous navigation systems. We discuss state-of-the-art algorithms including Deep Q-Networks (DQN), Policy Gradient methods, and Actor-Critic architectures. The paper also addresses current challenges in sample efficiency, sim-to-real transfer, and safety considerations.',
            'categories': 'cs.AI, cs.RO',
            'published_date': '2023-10-11 09:15:00',
            'updated_date': '2023-10-11 09:15:00',
            'pdf_url': 'https://arxiv.org/pdf/2310.12346.pdf'
        },
        {
            'id': '2310.12347',
            'title': 'Computer Vision in Medical Imaging: A Deep Learning Perspective',
            'authors': 'Maria Garcia, Robert Taylor, Sarah Chen',
            'abstract': 'Medical imaging has been revolutionized by deep learning techniques. This survey covers recent advances in convolutional neural networks for medical image analysis, including diagnostic applications in radiology, pathology, and ophthalmology. We discuss dataset challenges, model interpretability, and regulatory considerations for clinical deployment.',
            'categories': 'cs.CV, cs.LG',
            'published_date': '2023-10-10 16:45:00',
            'updated_date': '2023-10-10 16:45:00',
            'pdf_url': 'https://arxiv.org/pdf/2310.12347.pdf'
        },
        {
            'id': '2310.12348',
            'title': 'Federated Learning: Privacy-Preserving Machine Learning in Distributed Settings',
            'authors': 'Thomas Anderson, Emily White, Michael Zhang',
            'abstract': 'Federated learning enables training machine learning models across decentralized data sources while preserving privacy. This paper reviews algorithmic approaches including FedAvg, FedProx, and differential privacy techniques. We analyze communication efficiency, convergence properties, and security implications in various federated scenarios.',
            'categories': 'cs.LG, cs.CR',
            'published_date': '2023-10-09 11:20:00',
            'updated_date': '2023-10-09 11:20:00',
            'pdf_url': 'https://arxiv.org/pdf/2310.12348.pdf'
        },
        {
            'id': '2310.12349',
            'title': 'Graph Neural Networks for Social Network Analysis: Methods and Applications',
            'authors': 'Lisa Wang, Kevin Miller, Anna Rodriguez',
            'abstract': 'Graph Neural Networks (GNNs) have emerged as powerful tools for analyzing social networks and graph-structured data. This comprehensive review covers GNN architectures including Graph Convolutional Networks, GraphSAGE, and Graph Attention Networks. Applications span community detection, link prediction, and influence maximization in social platforms.',
            'categories': 'cs.SI, cs.LG',
            'published_date': '2023-10-08 13:55:00',
            'updated_date': '2023-10-08 13:55:00',
            'pdf_url': 'https://arxiv.org/pdf/2310.12349.pdf'
        },
        {
            'id': '2310.12350',
            'title': 'Quantum Machine Learning: Algorithms and Hardware Implementations',
            'authors': 'James Wilson, Rachel Green, Peter Kim',
            'abstract': 'Quantum computing promises exponential speedups for certain machine learning tasks. This paper surveys quantum algorithms for optimization, classification, and dimensionality reduction. We discuss current hardware limitations, noise mitigation strategies, and near-term applications on NISQ devices.',
            'categories': 'quant-ph, cs.LG',
            'published_date': '2023-10-07 08:30:00',
            'updated_date': '2023-10-07 08:30:00',
            'pdf_url': 'https://arxiv.org/pdf/2310.12350.pdf'
        }
    ]
    
    for article in demo_articles:
        cursor.execute('''
            INSERT OR REPLACE INTO articles 
            (id, title, authors, abstract, categories, published_date, updated_date, pdf_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            article['id'],
            article['title'],
            article['authors'],
            article['abstract'],
            article['categories'],
            article['published_date'],
            article['updated_date'],
            article['pdf_url']
        ))
    
    # 创建演示用户
    cursor.execute('''
        INSERT OR REPLACE INTO users (id, username, interests)
        VALUES (1, 'demo_user', ?)
    ''', (json.dumps(['machine learning', 'deep learning', 'computer vision', 'natural language processing']),))
    
    # 创建一些用户行为数据
    behaviors = [
        (1, '2310.12345', 'click'),
        (1, '2310.12345', 'like'),
        (1, '2310.12347', 'click'),
        (1, '2310.12347', 'save'),
        (1, '2310.12349', 'click'),
    ]
    
    for user_id, article_id, action_type in behaviors:
        cursor.execute('''
            INSERT INTO user_behaviors (user_id, article_id, action_type)
            VALUES (?, ?, ?)
        ''', (user_id, article_id, action_type))
    
    conn.commit()
    conn.close()
    
    print(f"✅ 演示数据创建成功！")
    print(f"📊 创建了 {len(demo_articles)} 篇文章")
    print("👤 创建了演示用户: demo_user")
    print("🎯 创建了用户行为数据")
    print("\n🚀 现在可以启动应用了:")
    print("python web/app.py")

if __name__ == "__main__":
    create_demo_data()