#!/usr/bin/env python3
"""
清空数据库脚本
删除数据库文件和TF-IDF模型，用于重新开始
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_config

def clear_database(confirm=True):
    """
    清空数据库和模型文件
    
    Args:
        confirm: 是否需要用户确认
    
    Returns:
        True if successful, False otherwise
    """
    config = get_config()
    
    # 要删除的文件列表
    files_to_remove = []
    
    # 数据库文件
    db_path = config.DATABASE_URL
    if os.path.exists(db_path):
        files_to_remove.append(('数据库', db_path))
    
    # TF-IDF模型文件
    tfidf_path = os.path.join(config.MODEL_DIR, 'tfidf_vectorizer.pkl')
    if os.path.exists(tfidf_path):
        files_to_remove.append(('TF-IDF模型', tfidf_path))
    
    # TF-IDF元数据文件
    tfidf_meta_path = os.path.join(config.MODEL_DIR, 'tfidf_meta.json')
    if os.path.exists(tfidf_meta_path):
        files_to_remove.append(('TF-IDF元数据', tfidf_meta_path))
    
    # 如果没有文件需要删除
    if not files_to_remove:
        print("✅ 数据库和模型文件不存在，无需清空")
        return True
    
    # 显示要删除的文件
    print("=" * 60)
    print("以下文件将被删除：")
    print("=" * 60)
    for name, path in files_to_remove:
        size = os.path.getsize(path) / 1024  # KB
        print(f"  - {name}: {path}")
        print(f"    大小: {size:.2f} KB")
    print("=" * 60)
    
    # 用户确认
    if confirm:
        response = input("\n⚠️  确认删除以上文件？(yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("❌ 操作已取消")
            return False
    
    # 执行删除
    deleted_count = 0
    for name, path in files_to_remove:
        try:
            os.remove(path)
            print(f"✅ 已删除 {name}")
            deleted_count += 1
        except Exception as e:
            print(f"❌ 删除 {name} 失败: {e}")
    
    print(f"\n🎉 完成！成功删除 {deleted_count}/{len(files_to_remove)} 个文件")
    
    if deleted_count == len(files_to_remove):
        print("\n💡 提示：")
        print("   - 运行 'python scripts/fetch_arxiv.py' 重新获取数据")
        print("   - 数据库和模型将自动重新创建")
        return True
    else:
        return False

def main():
    """主函数"""
    print("🗑️  数据库清空工具")
    print("=" * 60)
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] in ['-y', '--yes']:
        # 不需要确认
        clear_database(confirm=False)
    else:
        # 需要用户确认
        clear_database(confirm=True)

if __name__ == "__main__":
    main()
