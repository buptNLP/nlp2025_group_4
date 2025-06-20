import chromadb
from chromadb.utils import embedding_functions
import requests
import json
import re
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import asdict
import os

# 导入后端工具
from backend_utils import backend_logger as st
from backend_config import HAS_BM25, HAS_RERANKER
from models import AncientTextSegment
from text_processing import SmartTextChunker, AncientTextAnalyzer
from backend_retrieval import HybridRetriever  # 使用修复版本
from api_client import get_api_client, APIProvider

if HAS_RERANKER:
    import torch

class UniversalAncientRAG:
    """通用古文RAG系统 - 修复版本，解决重排序重复加载问题"""
    
    # def __init__(self, embedding_model: str = "BAAI/bge-large-zh-v1.5",
    def __init__(self, embedding_model: str = "BAAI/bge-large-zh-v1.5",
             max_chunk_size: int = 150,
             min_chunk_size: int = 20,
             context_window: int = 80,
             use_reranker: bool = None,
             reranker_model: str = None):
        self.client = chromadb.Client()
        self.collection_name = "ancient_texts_collection"
        self.embedding_model = embedding_model
        self.segments: List[AncientTextSegment] = []
        
        # 🔧 修复：正确配置重排序选项
        self.use_reranker = use_reranker if use_reranker is not None else (
            HAS_RERANKER and os.getenv("RAG_USE_RERANKER", "true").lower() == "true"
        )
        self.reranker_model = reranker_model or os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-large")
        
        # 使用配置参数初始化分块器
        self.chunker = SmartTextChunker(
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            context_window=context_window
        )
        
        self.analyzer = AncientTextAnalyzer()
        self.retriever: Optional[HybridRetriever] = None
        self.api_client = None
            
        # 配置embedding函数
        self._setup_embedding_function()
        
        # 初始化向量数据库
        self._initialize_collection()

    def set_api_config(self, config: Dict[str, Any]):
        """设置API配置并创建相应的客户端"""
        try:
            # 使用工厂方法创建 API 客户端
            self.api_client = get_api_client(config)
    
            # 存储当前配置（用于显示状态）
            self.current_api_config = config
            st.info(f"✅ 已配置 {config.get('provider', 'unknown').upper()} API")
            
        except Exception as e:
            st.error(f"设置 API 客户端失败: {str(e)}")
            self.api_client = None

    def update_chunker_params(self, max_chunk_size: int = None, 
                         min_chunk_size: int = None, 
                         context_window: int = None):
        """更新分块器参数"""
        if max_chunk_size is not None:
            self.chunker.max_chunk_size = max_chunk_size
        if min_chunk_size is not None:
            self.chunker.min_chunk_size = min_chunk_size
        if context_window is not None:
            self.chunker.context_window = context_window
        
        # 返回当前参数，用于显示
        return {
            'max_chunk_size': self.chunker.max_chunk_size,
            'min_chunk_size': self.chunker.min_chunk_size,
            'context_window': self.chunker.context_window
        }
    
    def update_reranker_config(self, use_reranker: bool = None, reranker_model: str = None):
        """更新重排序配置"""
        if use_reranker is not None:
            self.use_reranker = use_reranker and HAS_RERANKER
        
        if reranker_model is not None:
            self.reranker_model = reranker_model
        
        # 如果检索器已初始化，更新其配置
        if self.retriever:
            self.retriever.update_config(
                use_reranker=self.use_reranker,
                reranker_model=self.reranker_model
            )
        
        st.info(f"🔄 重排序配置已更新: 启用={self.use_reranker}, 模型={self.reranker_model}")
    
    def _setup_embedding_function(self):
        """启用GPU加速的embedding"""
        try:
            if HAS_RERANKER:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                st.info(f"🔧 使用设备: {device}")
                
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model,
                    device=device
                )
            else:
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.embedding_model
                )
        except Exception as e:
            st.warning(f"Embedding 函数初始化失败，使用默认配置: {e}")
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
    
    def _initialize_collection(self):
        """初始化collection"""
        try:
            # 尝试获取现有collection
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
        except:
            # 创建新collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": f"古文向量数据库 - {self.embedding_model}"}
            )

    def change_embedding_model(self, new_model: str):
        """更换embedding模型并重建索引"""
        st.info(f"🔄 切换embedding模型: {self.embedding_model} → {new_model}")
        
        # 保存当前数据
        old_segments = self.segments.copy()
        
        # 删除旧collection
        try:
            self.client.delete_collection(self.collection_name)
        except:
            pass
        
        # 更新模型
        self.embedding_model = new_model
        self._setup_embedding_function()
        self._initialize_collection()
        
        # 重建索引
        if old_segments:
            self.segments = old_segments
            self._build_vector_database()
            st.success(f"✅ 已切换到 {new_model} 并重建索引")

    def load_from_directory(self, root_dir: str, file_extensions: List[str] = None) -> int:
        """递归加载目录下的所有古文数据"""
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.text']
        
        root_path = Path(root_dir)
        total_segments = 0
        
        if not root_path.exists():
            st.error(f"目录不存在: {root_dir}")
            return 0
        
        # 递归查找所有指定格式的文件
        st.info("正在扫描文件...")
        
        all_text_files = []
        for ext in file_extensions:
            pattern = f"**/*{ext}"
            files = list(root_path.glob(pattern))
            all_text_files.extend(files)
        
        # 去重并排序
        all_text_files = sorted(list(set(all_text_files)))
        
        if not all_text_files:
            st.warning(f"在指定目录下没有找到任何支持的文件格式: {', '.join(file_extensions)}")
            return 0
        
        st.info(f"发现 {len(all_text_files)} 个文本文件，开始处理...")
        
        # 创建处理统计
        processing_stats = {
            'processed_files': 0,
            'empty_files': 0,
            'error_files': 0,
            'total_segments': 0
        }
        
        # 处理每个文本文件
        for file_idx, text_file in enumerate(all_text_files):
            if file_idx % 10 == 0:  # 每10个文件显示一次进度
                st.progress(
                    file_idx / len(all_text_files), 
                    f"正在处理: {text_file.name} ({file_idx + 1}/{len(all_text_files)})"
                )
            
            try:
                # 解析文件路径以获取书名和篇章信息
                book_name, chapter_name = self._parse_file_path(text_file, root_path)
                
                # 读取文件内容
                content = self._read_text_file(text_file)
                
                if content and len(content.strip()) > 10:  # 忽略过短的文件
                    segments = self._process_chapter(book_name, chapter_name, content, str(text_file))
                    self.segments.extend(segments)
                    total_segments += len(segments)
                    processing_stats['total_segments'] += len(segments)
                    processing_stats['processed_files'] += 1
                    
                    # 记录处理结果
                    if len(segments) > 0:
                        st.info(f"✅ {book_name}/{chapter_name}: {len(segments)} 段")
                else:
                    processing_stats['empty_files'] += 1
                    st.warning(f"⚠️ 空文件: {text_file.name}")
                    
            except Exception as e:
                processing_stats['error_files'] += 1
                st.error(f"❌ 处理失败: {text_file.name} - {str(e)}")
        
        # 显示处理统计
        st.info("📊 处理统计")
        st.metric("成功处理", processing_stats['processed_files'])
        st.metric("空文件", processing_stats['empty_files'])
        st.metric("错误文件", processing_stats['error_files'])
        st.metric("总文本段", processing_stats['total_segments'])
        
        # 构建向量数据库
        if self.segments:
            with st.spinner("构建向量数据库..."):
                self._build_vector_database()
            st.success(f"加载完成！共处理 {total_segments} 个文本片段")
        else:
            st.warning("未找到有效的文本数据")
        
        return total_segments
    
    def load_single_directory(self, root_dir: str, file_extensions: List[str] = None) -> int:
        """加载单个目录（非递归）的古文数据"""
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.text']
        
        root_path = Path(root_dir)
        total_segments = 0
        
        if not root_path.exists():
            st.error(f"目录不存在: {root_dir}")
            return 0
        
        # 查找当前目录下的文件
        st.info("正在扫描文件...")
        
        all_text_files = []
        for ext in file_extensions:
            files = list(root_path.glob(f"*{ext}"))
            all_text_files.extend(files)
        
        # 也检查直接子目录中的文件
        for subdir in root_path.iterdir():
            if subdir.is_dir():
                for ext in file_extensions:
                    files = list(subdir.glob(f"*{ext}"))
                    all_text_files.extend(files)
        
        all_text_files = sorted(list(set(all_text_files)))
        
        if not all_text_files:
            st.warning(f"在指定目录下没有找到任何支持的文件格式: {', '.join(file_extensions)}")
            return 0
        
        st.info(f"发现 {len(all_text_files)} 个文本文件，开始处理...")
        
        # 处理文件的逻辑与递归版本相同
        for file_idx, text_file in enumerate(all_text_files):
            if file_idx % 10 == 0:
                st.progress(
                    file_idx / len(all_text_files), 
                    f"正在处理: {text_file.name} ({file_idx + 1}/{len(all_text_files)})"
                )
            
            try:
                book_name, chapter_name = self._parse_file_path(text_file, root_path)
                content = self._read_text_file(text_file)
                
                if content and len(content.strip()) > 10:
                    segments = self._process_chapter(book_name, chapter_name, content, str(text_file))
                    self.segments.extend(segments)
                    total_segments += len(segments)
                    
                    if len(segments) > 0:
                        st.success(f"✅ {book_name}/{chapter_name}: {len(segments)} 段")
                        
            except Exception as e:
                st.error(f"❌ 处理失败: {text_file.name} - {str(e)}")
        
        # 构建向量数据库
        if self.segments:
            with st.spinner("构建向量数据库..."):
                self._build_vector_database()
        
        st.success(f"加载完成！共处理 {total_segments} 个文本片段")
        return total_segments
    
    def _parse_file_path(self, file_path: Path, root_path: Path) -> Tuple[str, str]:
        """智能解析文件路径以获取书名和篇章信息"""
        try:
            # 获取相对于根目录的路径
            relative_path = file_path.relative_to(root_path)
            path_parts = list(relative_path.parts[:-1])  # 排除文件名
            
            # 根据目录层次智能解析
            if len(path_parts) == 0:
                # 直接在根目录：text.txt
                book_name = root_path.name
                chapter_name = file_path.stem
            elif len(path_parts) == 1:
                # 一层目录：书名/text.txt 或 篇章/text.txt
                book_name = path_parts[0]
                chapter_name = file_path.stem if file_path.stem != 'text' else path_parts[0]
            elif len(path_parts) == 2:
                # 标准结构：书名/篇章/text.txt
                book_name = path_parts[0]
                chapter_name = path_parts[1]
            else:
                # 深层结构：书名/卷/篇章/子章节/text.txt
                book_name = path_parts[0]
                # 将中间层级用层次分隔符连接
                chapter_parts = path_parts[1:]
                chapter_name = " > ".join(chapter_parts)
            
            # 清理名称中的特殊字符
            book_name = self._clean_name(book_name)
            chapter_name = self._clean_name(chapter_name)
            
            return book_name, chapter_name
            
        except Exception as e:
            # 出错时使用默认值
            return root_path.name, file_path.stem
    
    def _clean_name(self, name: str) -> str:
        """清理文件/目录名称"""
        # 移除可能的编号前缀
        name = re.sub(r'^\d+[-._]\s*', '', name)
        # 替换特殊字符
        name = re.sub(r'[_-]+', ' ', name)
        return name.strip()
    
    def _process_chapter(self, book: str, chapter: str, content: str, file_path: str = "") -> List[AncientTextSegment]:
        """处理单个篇章（使用新的层级分块）"""
        segments = []
        
        # 使用新的分块方法
        chunks = self.chunker.chunk_text(content)
        
        for chunk in chunks:
            # 分析文本
            topic = self.analyzer.classify_topic(chunk.content)
            
            # 创建片段ID
            segment_id = f"{book}_{chapter}_{chunk.paragraph_index:03d}_{chunk.sub_index:02d}"
            
            # 整合元数据
            metadata = {
                'length': len(chunk.content),
                'paragraph_index': chunk.paragraph_index,
                'sub_index': chunk.sub_index,
                'is_continuation': chunk.is_continuation,
                'prev_context': chunk.prev_context,
                'next_context': chunk.next_context,
                'classical_terms': self._extract_classical_terms(chunk.content),
            }
            
            # 如果chunk有额外的metadata，合并进来
            if chunk.metadata:
                metadata.update(chunk.metadata)
            
            # 构建完整的上下文（用于显示）
            full_context = f"{chunk.prev_context} **{chunk.content}** {chunk.next_context}"
            
            segment = AncientTextSegment(
                book=book,
                chapter=chapter,
                content=chunk.content,
                topic=topic,
                segment_id=segment_id,
                context=full_context,
                metadata=metadata
            )
            segments.append(segment)
        
        return segments
    
    def _extract_classical_terms(self, text: str) -> List[str]:
        """提取古典术语"""
        classical_terms = [
            '仁', '义', '礼', '智', '信', '德', '道', '天', '君子', '小人',
            '学', '习', '教', '诲', '政', '治', '民', '国', '家', '孝',
            '悌', '忠', '恕', '诚', '正', '修', '齐', '治', '平'
        ]
        
        found_terms = []
        for term in classical_terms:
            if term in text:
                found_terms.append(term)
        
        return found_terms
    
    def _read_text_file(self, file_path: Path) -> str:
        """读取文本文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # 处理可能的XML标签
            content_match = re.search(r'<content>(.*?)</content>', content, re.DOTALL)
            if content_match:
                return content_match.group(1).strip()
            
            return content
        except Exception as e:
            st.warning(f"读取文件失败 {file_path}: {e}")
            return ""
    
    def _build_vector_database(self):
        """构建向量数据库 - 修复版本，正确传递重排序配置"""
        # 清空旧数据
        try:
            if self.collection.count() > 0:
                all_data = self.collection.get()
                if all_data['ids']:
                    self.collection.delete(ids=all_data['ids'])
        except Exception as e:
            st.warning(f"清空数据库时出现问题，将重新创建：{e}")
            try:
                self.client.delete_collection(self.collection_name)
            except:
                pass
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "通用古文向量数据库"}
            )
        
        # 批量添加
        batch_size = 5000
        for i in range(0, len(self.segments), batch_size):
            batch_segments = self.segments[i:i+batch_size]
            
            documents = [seg.content for seg in batch_segments]
            
            # 过滤并转换元数据，确保只包含基本数据类型
            metadatas = []
            for seg in batch_segments:
                # 创建基础元数据
                base_metadata = {
                    'book': seg.book,
                    'chapter': seg.chapter,
                    'topic': seg.topic,
                    'segment_id': seg.segment_id
                }
                
                # 处理扩展元数据
                if hasattr(seg, 'metadata') and seg.metadata:
                    filtered_extended = self._filter_metadata_for_chromadb(seg.metadata)
                    base_metadata.update(filtered_extended)
                
                metadatas.append(base_metadata)
            
            ids = [seg.segment_id for seg in batch_segments]
            
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        if self.segments:
            # 🔧 修复：正确传递重排序配置
            st.info(f"🔧 正在初始化检索器")
            st.info(f"   重排序: {'启用' if self.use_reranker else '禁用'}")
            st.info(f"   模型: {self.reranker_model}")
            
            self.retriever = HybridRetriever(
                self.collection, 
                self.segments,
                use_reranker=self.use_reranker,      # 🔧 修复：传递配置
                reranker_model=self.reranker_model   # 🔧 修复：传递配置
            )

            st.success("✅ 混合检索器初始化完成")
            
    def _filter_metadata_for_chromadb(self, metadata: Dict) -> Dict:
        """过滤元数据，确保只包含ChromaDB支持的数据类型"""
        filtered = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                # 基本类型直接保留
                filtered[key] = value
            elif isinstance(value, list):
                # 将列表转换为字符串和计数
                if value:  # 非空列表
                    filtered[f"{key}_str"] = ', '.join(map(str, value))
                    filtered[f"{key}_count"] = len(value)
                else:
                    filtered[f"{key}_count"] = 0
            elif isinstance(value, dict):
                # 将字典转换为字符串
                filtered[f"{key}_str"] = str(value)
            else:
                # 其他类型转换为字符串
                filtered[f"{key}_str"] = str(value)
        
        return filtered

    def search(self, query: str, top_k: int = 5, 
            search_mode: str = 'hybrid',
            metadata_filter: Dict[str, str] = None,
            bm25_weight: float = None,
            vector_weight: float = None) -> List[Dict[str, Any]]:
        """
        统一的搜索接口
        """
        if not self.retriever:
            st.error("检索器未初始化，请先加载数据")
            return []
        
        try:
            # 定义搜索方法映射
            search_methods = {
                'hybrid': {
                    'with_filter': self.retriever.hybrid_search_with_filter,
                    'without_filter': self.retriever.hybrid_search,
                    'fallback': self.retriever._vector_search
                },
                'vector': {
                    'with_filter': self.retriever.vector_search_with_filter,
                    'without_filter': self.retriever._vector_search,
                    'fallback': None
                },
                'bm25': {
                    'with_filter': self.retriever.bm25_search_with_filter,
                    'without_filter': self.retriever._bm25_search,
                    'fallback': None
                }
            }
            
            # 获取对应的搜索方法
            mode_methods = search_methods.get(search_mode)
            if not mode_methods:
                st.error(f"不支持的搜索模式: {search_mode}")
                return []
            
            # 特殊处理：BM25不可用时的降级
            if search_mode == 'bm25' and not HAS_BM25:
                st.warning("BM25检索不可用，请安装 rank-bm25 包")
                return []
            
            if search_mode == 'hybrid' and not HAS_BM25:
                st.info("BM25不可用，自动切换到向量检索模式")
                # 使用降级方法
                if metadata_filter:
                    return self.retriever.vector_search_with_filter(
                        query, top_k, metadata_filter
                    )
                else:
                    return mode_methods['fallback'](query, top_k)
            
            # 执行搜索
            if metadata_filter:
                # 有过滤条件
                return mode_methods['with_filter'](
                    query, top_k, metadata_filter,
                    bm25_weight=bm25_weight,
                    vector_weight=vector_weight
                )
            else:
                # 无过滤条件
                method = mode_methods['without_filter']
                if search_mode == 'hybrid':
                    # hybrid模式需要传递权重参数
                    return method(
                        query, top_k,
                        bm25_weight=bm25_weight,
                        vector_weight=vector_weight
                    )
                else:
                    # 其他模式不需要权重参数
                    return method(query, top_k)
                
        except Exception as e:
            st.error(f"搜索过程中出现错误: {e}")
            return []

    def search_with_metadata(self, query: str, top_k: int = 5, 
                            search_mode: str = 'hybrid', 
                            metadata_filter: Dict[str, str] = None,
                            bm25_weight: float = None,
                            vector_weight: float = None) -> List[Dict[str, Any]]:
        """向后兼容的方法，调用统一的search方法"""
        return self.search(
            query=query,
            top_k=top_k,
            search_mode=search_mode,
            metadata_filter=metadata_filter,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight
        )
    
    def analyze_and_search(self, query: str, top_k: int = 5, 
                  search_mode: str = 'hybrid',
                  metadata_filter: Dict[str, str] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        综合分析并执行搜索
        返回：(分析结果, 搜索结果)
        """
        # 如果没有API客户端，直接执行搜索
        if not self.api_client:
            analysis_result = {
                'need_search': True,
                'reason': '未配置API客户端',
                'optimized_query': query,
                'bm25_weight': 0.3,
                'vector_weight': 0.7,
                'direct_answer': None
            }
            search_results = self.search(query, top_k, search_mode)
            return analysis_result, search_results
        
        # 调用综合分析API
        analysis_result = self.api_client.analyze_query(query)
        
        # 如果不需要搜索，返回空结果
        if not analysis_result['need_search']:
            return analysis_result, []
        
        # 使用优化后的查询和权重进行搜索
        optimized_query = analysis_result['optimized_query']
        bm25_weight = analysis_result['bm25_weight']
        vector_weight = analysis_result['vector_weight']
        
        # 执行搜索，传递权重参数
        if metadata_filter:
            search_results = self.search_with_metadata(
                optimized_query, top_k, search_mode, metadata_filter,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight
            )
        else:
            search_results = self.search(
                optimized_query, top_k, search_mode,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight
            )
        
        return analysis_result, search_results
    
    def generate_answer(self, query: str, context: list) -> str:
        """生成答案"""
        if self.api_client:
            return self.api_client.generate_answer(query, context)
        return self._generate_basic_answer(query, context)
    
    def _generate_basic_answer(self, query: str, context: List[Dict[str, Any]]) -> str:
        """生成基础回答"""
        answer = f"关于「{query}」，在古文中找到以下相关内容：\n\n"
        
        for i, item in enumerate(context[:3], 1):
            meta = item['metadata']
            
            # 智能选择显示分数
            score_info = self._get_score_info(item)
            
            answer += f"**{i}. 《{meta['book']}·{meta['chapter']}》**\n"
            answer += f"原文：「{item['content']}」\n"
            answer += f"话题：{meta['topic']} | {score_info}\n\n"
    
        # 添加统计信息
        books = set(item['metadata']['book'] for item in context)
        topics = set(item['metadata']['topic'] for item in context)
        
        answer += f"💡 **内容分析**：\n"
        answer += f"- 涉及典籍：{', '.join(books)}\n"
        answer += f"- 相关主题：{', '.join(topics)}\n"
        answer += f"- 检索模式：混合检索（BM25 + 向量相似度）"
        
        return answer
    
    def _get_score_info(self, item: Dict[str, Any]) -> str:
        """获取分数显示信息"""
        # 优先显示重排序分数
        if 'rerank_score' in item and item['rerank_score'] > 0:
            return f"重排序评分：{item['rerank_score']:.3f}"
        
        # 其次显示综合分数
        elif 'combined_score' in item and item['combined_score'] > 0:
            return f"综合评分：{item['combined_score']:.3f}"
        
        # 显示向量分数
        elif 'vector_score' in item and item['vector_score'] > 0:
            return f"相似度：{item['vector_score']:.3f}"
        
        # 显示BM25分数
        elif 'bm25_score' in item and item['bm25_score'] > 0:
            return f"匹配度：{item['bm25_score']:.3f}"
        
        # 默认显示
        else:
            return "相关度：高"
