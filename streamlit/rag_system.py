import streamlit as st
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

from config import HAS_RERANKER
from models import AncientTextSegment
from text_processing import SmartTextChunker, AncientTextAnalyzer
from retrieval import HybridRetriever

from config import HAS_BM25, HAS_RERANKER
from api_client import get_api_client, APIProvider  # 使用新的 API 客户端工厂

if HAS_RERANKER:
    import torch

class UniversalAncientRAG:
    """通用古文RAG系统"""
    
    def __init__(self, embedding_model: str = "BAAI/bge-large-zh-v1.5",
             max_chunk_size: int = 150,
             min_chunk_size: int = 20,
             context_window: int = 80):
        self.client = chromadb.Client()
        self.collection_name = "ancient_texts_collection"
        self.embedding_model = embedding_model
        self.segments: List[AncientTextSegment] = []
        
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
    
    def _setup_embedding_function(self):
        """启用GPU加速的embedding"""
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            st.info(f"🔧 使用设备: {device}")
            
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model,
                device=device  # 关键：指定GPU
            )
        except Exception as e:
            st.warning(f"GPU加速失败，使用CPU: {e}")
    
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
        progress_bar = st.progress(0, text="正在扫描文件...")
        
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
            progress_bar.progress(
                file_idx / len(all_text_files), 
                text=f"正在处理: {text_file.name} ({file_idx + 1}/{len(all_text_files)})"
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
                    
                    # 实时显示处理结果
                    if len(segments) > 0:
                        st.sidebar.success(f"✅ {book_name}/{chapter_name}: {len(segments)} 段")
                else:
                    processing_stats['empty_files'] += 1
                    st.sidebar.warning(f"⚠️ 空文件: {text_file.name}")
                    
            except Exception as e:
                processing_stats['error_files'] += 1
                st.sidebar.error(f"❌ 处理失败: {text_file.name} - {str(e)}")
        
        # 显示处理统计
        st.sidebar.markdown("### 📊 处理统计")
        st.sidebar.metric("成功处理", processing_stats['processed_files'])
        st.sidebar.metric("空文件", processing_stats['empty_files'])
        st.sidebar.metric("错误文件", processing_stats['error_files'])
        st.sidebar.metric("总文本段", processing_stats['total_segments'])
        
        # 完成数据加载
        progress_bar.progress(1.0, text="构建向量数据库...")
        
        # 构建向量数据库
        if self.segments:
            self._build_vector_database()
            progress_bar.progress(1.0, text=f"加载完成！共处理 {total_segments} 个文本片段")
        else:
            progress_bar.progress(1.0, text="未找到有效的文本数据")
        
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
        progress_bar = st.progress(0, text="正在扫描文件...")
        
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
            progress_bar.progress(
                file_idx / len(all_text_files), 
                text=f"正在处理: {text_file.name} ({file_idx + 1}/{len(all_text_files)})"
            )
            
            try:
                book_name, chapter_name = self._parse_file_path(text_file, root_path)
                content = self._read_text_file(text_file)
                
                if content and len(content.strip()) > 10:
                    segments = self._process_chapter(book_name, chapter_name, content, str(text_file))
                    self.segments.extend(segments)
                    total_segments += len(segments)
                    
                    if len(segments) > 0:
                        st.sidebar.success(f"✅ {book_name}/{chapter_name}: {len(segments)} 段")
                        
            except Exception as e:
                st.sidebar.error(f"❌ 处理失败: {text_file.name} - {str(e)}")
        
        # 构建向量数据库
        if self.segments:
            progress_bar.progress(1.0, text="构建向量数据库...")
            self._build_vector_database()
        
        progress_bar.progress(1.0, text=f"加载完成！共处理 {total_segments} 个文本片段")
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
        """构建向量数据库（修复删除错误）"""
        # 清空旧数据 - 修复版本
        try:
            if self.collection.count() > 0:
                # 方法1：获取所有ID然后删除
                all_data = self.collection.get()
                if all_data['ids']:
                    self.collection.delete(ids=all_data['ids'])
            
            
        except Exception as e:
            # 如果删除失败，尝试重新创建collection
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
            # 确保正确获取用户配置
            use_reranker = st.session_state.get('use_reranker', HAS_RERANKER)
            
            # 添加调试信息
            st.info(f"🔧 正在初始化检索器，重排序: {'启用' if use_reranker else '禁用'}")
            
            self.retriever = HybridRetriever(
                self.collection, 
                self.segments, 
                # use_reranker=use_reranker  # 确保正确传递参数
            )

            # 显示重排序状态
            st.info("✅ 混合检索器初始化完成")
            
    
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
    
    # rag_system.py 中修改

    def search(self, query: str, top_k: int = 5, 
            search_mode: str = 'hybrid',
            metadata_filter: Dict[str, str] = None,
            bm25_weight: float = None,
            vector_weight: float = None) -> List[Dict[str, Any]]:
        """
        统一的搜索接口
        
        Args:
            query: 查询字符串
            top_k: 返回结果数量
            search_mode: 检索模式 ('hybrid', 'vector', 'bm25')
            metadata_filter: 元数据过滤条件
            bm25_weight: BM25权重
            vector_weight: 向量权重
        
        Returns:
            搜索结果列表
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

    # 删除原来的search_with_metadata方法，改为别名或向后兼容
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
                'direct_answer': None,
                'extracted_book': None
            }
            search_results = self.search(query, top_k, search_mode, metadata_filter)
            return analysis_result, search_results
        
        # 调用综合分析API
        analysis_result = self.api_client.analyze_query(query)
        
        # 如果不需要搜索，返回空结果
        if not analysis_result['need_search']:
            return analysis_result, []
        
        # --- 新增逻辑：合并AI提取的过滤器和用户手动输入的过滤器 ---
        # 复制用户手动输入的过滤器，避免直接修改
        effective_metadata_filter = metadata_filter.copy() if metadata_filter else {}
        
        extracted_book = analysis_result.get('extracted_book')
        
        # 核心逻辑：如果AI提取了书名，并且用户没有手动指定书名，则使用AI提取的书名
        if extracted_book and 'book' not in effective_metadata_filter:
            effective_metadata_filter['book'] = extracted_book
            # 添加一个标志，方便UI显示提示信息
            analysis_result['filter_source'] = 'ai'

        # 使用优化后的查询和权重进行搜索
        optimized_query = analysis_result['optimized_query']
        bm25_weight = analysis_result['bm25_weight']
        vector_weight = analysis_result['vector_weight']
        
        # 执行搜索，传递合并后的过滤器
        # 注意：这里我们将 effective_metadata_filter 传递给搜索函数
        search_results = self.search(
            optimized_query, top_k, search_mode,
            metadata_filter=effective_metadata_filter, # --- 修改 ---
            bm25_weight=bm25_weight,
            vector_weight=vector_weight
        )
        
        return analysis_result, search_results
    
    # 在 UniversalAncientRAG 类中添加以下方法
    def multi_round_search(self, query: str, top_k_per_round: int = 5, 
                        search_mode: str = 'hybrid') -> Dict[str, Any]:
        """
        执行多轮检索
        返回：{
            'original_query': str,
            'decomposition': dict,  # 拆解结果
            'subtasks_results': list,  # 各子任务的检索结果
            'synthesis': str  # 综合回答
        }
        """
        # 如果没有API客户端，降级为单轮检索
        if not self.api_client:
            st.warning("多轮检索需要配置API客户端")
            single_results = self.search(query, top_k_per_round * 2, search_mode)
            return {
                'original_query': query,
                'decomposition': {'need_multi_round': False, 'reason': '无API客户端'},
                'subtasks_results': [{'subtask_query': query, 'results': single_results}],
                'synthesis': self._generate_basic_answer(query, single_results)
            }
        
        # 第一步：任务拆解
        with st.spinner("🔍 正在分析问题复杂度..."):
            decomposition = self.api_client.decompose_complex_query(query)
        
        # 如果不需要多轮检索，执行单轮检索
        if not decomposition.get('need_multi_round', False):
            st.info(f"💡 {decomposition.get('reason', '该问题不需要多轮检索')}")
            single_results = self.search(query, top_k_per_round * 2, search_mode)
            return {
                'original_query': query,
                'decomposition': decomposition,
                'subtasks_results': [{'subtask_query': query, 'results': single_results}],
                'synthesis': self.generate_answer(query, single_results)
            }
        
        # 第二步：执行多轮检索
        st.success(f"✅ 已将问题拆解为 {len(decomposition['subtasks'])} 个子任务")
        
        subtasks_results = []
        progress_bar = st.progress(0, text="开始多轮检索...")
        
        for idx, subtask in enumerate(decomposition['subtasks']):
            progress = (idx + 1) / len(decomposition['subtasks'])
            progress_bar.progress(
                progress, 
                text=f"正在检索子任务 {idx+1}: {subtask['subtask_focus']}"
            )
            
            # 对每个子任务执行检索
            with st.expander(f"子任务 {idx+1}: {subtask['subtask_query']}", expanded=False):
                # 使用子任务特定的权重
                weights = subtask.get('search_weight', {'bm25': 0.3, 'vector': 0.7})
                
                results = self.search(
                    subtask['subtask_query'],
                    top_k_per_round,
                    search_mode,
                    bm25_weight=weights['bm25'],
                    vector_weight=weights['vector']
                )
                
                # 显示子任务结果
                if results:
                    st.write(f"🎯 找到 {len(results)} 个相关片段")
                    for i, res in enumerate(results[:3]):  # 显示前3个
                        meta = res['metadata']
                        st.write(f"{i+1}. 《{meta['book']}·{meta['chapter']}》")
                
                subtasks_results.append({
                    'subtask_id': subtask['subtask_id'],
                    'subtask_query': subtask['subtask_query'],
                    'subtask_focus': subtask['subtask_focus'],
                    'results': results,
                    'synthesis_instruction': decomposition.get('synthesis_instruction', '')
                })
        
        progress_bar.progress(1.0, text="检索完成，正在综合分析...")
        
        # 第三步：综合多轮结果
        with st.spinner("🤖 正在综合多轮检索结果..."):
            synthesis = self.api_client.synthesize_multi_round_results(query, subtasks_results)
        
        return {
            'original_query': query,
            'decomposition': decomposition,
            'subtasks_results': subtasks_results,
            'synthesis': synthesis
        }
    
    
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
    
    def save_processing_results(self, output_dir: str = "./processing_results") -> bool:
        """保存处理结果到指定目录"""
        if not self.segments:
            st.warning("没有处理结果可保存")
            return False
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 生成时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. 保存详细的JSON格式结果
            self._save_detailed_json(output_path, timestamp)
            
            # 2. 保存便于查看的HTML格式结果
            self._save_html_report(output_path, timestamp)
            
            # 3. 保存CSV格式结果（便于Excel打开）
            self._save_csv_report(output_path, timestamp)
            
            # 4. 保存按书籍分类的文本文件
            self._save_by_books(output_path, timestamp)
            
            # 5. 生成统计报告
            self._save_statistics_report(output_path, timestamp)
            
            st.success(f"✅ 处理结果已保存到：{output_path.absolute()}")
            return True
            
        except Exception as e:
            st.error(f"保存处理结果失败：{str(e)}")
            return False
    
    def _save_detailed_json(self, output_path: Path, timestamp: str):
        """保存详细的JSON格式结果"""
        json_file = output_path / f"segments_detailed_{timestamp}.json"
        
        segments_data = []
        for seg in self.segments:
            segment_dict = asdict(seg)
            segments_data.append(segment_dict)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(segments_data, f, ensure_ascii=False, indent=2)
        
        st.info(f"📄 详细JSON结果：{json_file.name}")
    
    def _save_html_report(self, output_path: Path, timestamp: str):
        """保存HTML格式的可视化报告"""
        html_file = output_path / f"processing_report_{timestamp}.html"
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>古文处理结果报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 20px; margin-bottom: 30px; }}
        .stats {{ display: flex; justify-content: space-around; margin-bottom: 30px; }}
        .stat-card {{ background: #ecf0f1; padding: 15px; border-radius: 8px; text-align: center; min-width: 120px; }}
        .stat-number {{ font-size: 24px; font-weight: bold; color: #e74c3c; }}
        .segment {{ border: 1px solid #bdc3c7; margin-bottom: 15px; border-radius: 8px; overflow: hidden; }}
        .segment-header {{ background: #34495e; color: white; padding: 10px; font-weight: bold; }}
        .segment-content {{ padding: 15px; }}
        .metadata {{ background: #f8f9fa; padding: 10px; margin-top: 10px; border-radius: 4px; font-size: 0.9em; }}
        .content-text {{ line-height: 1.8; margin: 10px 0; font-size: 16px; }}
        .book-section {{ margin-bottom: 40px; }}
        .book-title {{ color: #8e44ad; font-size: 20px; font-weight: bold; margin-bottom: 15px; border-left: 4px solid #8e44ad; padding-left: 10px; }}
        .context {{ background: #fff3cd; padding: 10px; margin-top: 10px; border-radius: 4px; border-left: 4px solid #ffc107; }}
        .topic-tag {{ background: #17a2b8; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-right: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📜 古文处理结果报告</h1>
            <p>生成时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{len(self.segments)}</div>
                <div>文本片段总数</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(set(seg.book for seg in self.segments))}</div>
                <div>书籍数量</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(set(seg.chapter for seg in self.segments))}</div>
                <div>章节数量</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{sum(len(seg.content) for seg in self.segments)}</div>
                <div>总字符数</div>
            </div>
        </div>
        
        {self._generate_html_segments_by_book()}
    </div>
</body>
</html>"""
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        st.info(f"🌐 HTML可视化报告：{html_file.name}")
    
    def _generate_html_segments_by_book(self) -> str:
        """生成按书籍分组的HTML内容"""
        html_parts = []
        
        # 按书籍分组
        books = {}
        for seg in self.segments:
            if seg.book not in books:
                books[seg.book] = []
            books[seg.book].append(seg)
        
        for book_name, segments in books.items():
            html_parts.append(f'<div class="book-section">')
            html_parts.append(f'<div class="book-title">📖 {book_name} ({len(segments)} 个片段)</div>')
            
            for i, seg in enumerate(segments, 1):
                html_parts.append(f'''
                <div class="segment">
                    <div class="segment-header">
                        片段 {i} - {seg.chapter} 
                        <span class="topic-tag">{seg.topic}</span>
                    </div>
                    <div class="segment-content">
                        <div class="content-text">{seg.content}</div>
                        <div class="metadata">
                            <strong>片段ID：</strong>{seg.segment_id} | 
                            <strong>字符数：</strong>{len(seg.content)} | 
                            <strong>位置：</strong>{seg.metadata.get('position', '未知')}
                        </div>
                        {f'<div class="context"><strong>上下文：</strong><br>{seg.context}</div>' if seg.context != seg.content else ''}
                    </div>
                </div>
                ''')
            
            html_parts.append('</div>')
        
        return ''.join(html_parts)
    
    def _save_csv_report(self, output_path: Path, timestamp: str):
        """保存CSV格式报告（便于Excel查看）"""
        csv_file = output_path / f"segments_table_{timestamp}.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # 写入表头
            headers = [
                '序号', '书名', '章节', '说话人', '话题', '内容', 
                '字符数', '片段ID', '位置', '是否对话', '古典术语'
            ]
            writer.writerow(headers)
            
            # 写入数据
            for i, seg in enumerate(self.segments, 1):
                classical_terms = seg.metadata.get('classical_terms', [])
                if isinstance(classical_terms, list):
                    terms_str = ', '.join(classical_terms)
                else:
                    terms_str = str(classical_terms)
                
                row = [
                    i,
                    seg.book,
                    seg.chapter,
                    seg.topic,
                    seg.content,
                    len(seg.content),
                    seg.segment_id,
                    seg.metadata.get('position', ''),
                    seg.metadata.get('has_dialogue', False),
                    terms_str
                ]
                writer.writerow(row)
        
        st.info(f"📊 CSV表格文件：{csv_file.name}")
    
    def _save_by_books(self, output_path: Path, timestamp: str):
        """按书籍保存分类的文本文件"""
        books_dir = output_path / f"by_books_{timestamp}"
        books_dir.mkdir(exist_ok=True)
        
        # 按书籍分组
        books = {}
        for seg in self.segments:
            if seg.book not in books:
                books[seg.book] = {}
            if seg.chapter not in books[seg.book]:
                books[seg.book][seg.chapter] = []
            books[seg.book][seg.chapter].append(seg)
        
        for book_name, chapters in books.items():
            book_dir = books_dir / book_name
            book_dir.mkdir(exist_ok=True)
            
            # 为每个章节创建文件
            for chapter_name, segments in chapters.items():
                chapter_file = book_dir / f"{chapter_name}_segments.txt"
                
                with open(chapter_file, 'w', encoding='utf-8') as f:
                    f.write(f"《{book_name}》- {chapter_name}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for i, seg in enumerate(segments, 1):
                        f.write(f"【片段 {i}】\n")
                        f.write(f"话题：{seg.topic}\n")
                        f.write(f"内容：{seg.content}\n")
                        f.write(f"片段ID：{seg.segment_id}\n")
                        f.write(f"字符数：{len(seg.content)}\n")
                        
                        # 添加古典术语信息
                        terms = seg.metadata.get('classical_terms', [])
                        if terms:
                            if isinstance(terms, list):
                                f.write(f"古典术语：{', '.join(terms)}\n")
                            else:
                                f.write(f"古典术语：{terms}\n")
                        
                        f.write("-" * 30 + "\n\n")
        
        st.info(f"📚 按书籍分类的文件：{books_dir.name}/")
    
    def _save_statistics_report(self, output_path: Path, timestamp: str):
        """生成统计报告"""
        stats_file = output_path / f"statistics_{timestamp}.txt"
        
        # 收集统计数据
        total_segments = len(self.segments)
        books_stats = {}
        topics_stats = {}
        length_stats = []
        
        for seg in self.segments:
            # 书籍统计
            books_stats[seg.book] = books_stats.get(seg.book, 0) + 1
            
            # 话题统计
            topics_stats[seg.topic] = topics_stats.get(seg.topic, 0) + 1
            
            # 长度统计
            length_stats.append(len(seg.content))
        
        # 计算长度统计
        avg_length = sum(length_stats) / len(length_stats) if length_stats else 0
        min_length = min(length_stats) if length_stats else 0
        max_length = max(length_stats) if length_stats else 0
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("古文处理统计报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
            
            f.write("📊 基本统计\n")
            f.write("-" * 20 + "\n")
            f.write(f"文本片段总数：{total_segments}\n")
            f.write(f"平均片段长度：{avg_length:.1f} 字符\n")
            f.write(f"最短片段长度：{min_length} 字符\n")
            f.write(f"最长片段长度：{max_length} 字符\n")
            f.write(f"总字符数：{sum(length_stats)}\n\n")
            
            f.write("📚 书籍分布\n")
            f.write("-" * 20 + "\n")
            for book, count in sorted(books_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_segments) * 100
                f.write(f"{book}：{count} 段 ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write("🏷️ 话题分布\n")
            f.write("-" * 20 + "\n")
            for topic, count in sorted(topics_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_segments) * 100
                f.write(f"{topic}：{count} 段 ({percentage:.1f}%)\n")
            f.write("\n")
        
        st.info(f"📈 统计报告：{stats_file.name}")
