import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['TRANSFORMERS_OFFLINE'] = '1'
# os.environ['HF_HUB_OFFLINE'] = '1'
import streamlit as st
# 页面配置
st.set_page_config(
    page_title="通用古文智能问答系统",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

import re
import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import hashlib
import requests
from pathlib import Path
import jieba
from collections import defaultdict
import numpy as np
import json
import csv
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 尝试导入可选依赖
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    st.warning("⚠️ 未安装 rank-bm25，将无法使用BM25检索功能。运行: pip install rank-bm25")

try:
    from sentence_transformers import CrossEncoder
    import torch
    torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 
    HAS_RERANKER = True
except ImportError:
    HAS_RERANKER = False
    st.warning("⚠️ 未安装 sentence-transformers，将无法使用重排序功能。运行: pip install sentence-transformers")

@dataclass
class AncientTextSegment:
    """古文片段数据结构"""
    book: str          # 书名
    chapter: str       # 篇章
    speaker: str       # 说话人
    content: str       # 内容
    topic: str         # 话题
    segment_id: str    # 片段ID
    context: str       # 上下文
    metadata: Dict[str, Any]  # 扩展元数据

class SmartTextChunker:
    """智能文本分块器 - 严格按行分块版"""
    
    def __init__(self):
        # 古文常见的分句标点
        self.sentence_endings = ['。', '！', '？', '；', '：']
        # 对话标识词
        self.dialogue_markers = [
            '曰', '云', '问', '答', '对', '谓', '言', '曰：', '问：', '答：'
        ]
    
    def chunk_by_semantic_units(self, text: str, max_chunk_size: int = 200) -> List[str]:
        """
        主分块方法 - 优先使用严格按行分块
        """
        # 对于论语等按行组织的文本，直接按行分块，不合并
        if self._is_well_organized_by_lines(text):
            return self.strict_line_chunking(text)
        else:
            # 其他情况使用原有逻辑
            return self.fallback_semantic_chunking(text, max_chunk_size)
    
    def strict_line_chunking(self, text: str) -> List[str]:
        """
        严格按行分块 - 每行一个独立chunk，不合并
        """
        lines = text.strip().split('\n')
        chunks = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) >= 10:  # 只过滤明显的空行和过短行
                chunks.append(line)
        
        return chunks
    
    def fallback_semantic_chunking(self, text: str, max_chunk_size: int) -> List[str]:
        """
        备用语义分块方法（当文本不是按行组织时使用）
        """
        chunks = []
        current_chunk = ""
        
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return self._post_process_chunks(chunks)
    
    def _is_well_organized_by_lines(self, text: str) -> bool:
        """
        判断是否是按行组织的文本（论语格式检测）
        """
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False
        
        dialogue_lines = 0
        valid_lines = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            valid_lines += 1
            
            # 检查是否包含典型的论语对话模式
            if any(marker in line for marker in ['曰：', '问曰：', '对曰：', '谓']) or \
               line.startswith('子曰') or line.startswith('子谓') or \
               '问' in line and '曰' in line:
                dialogue_lines += 1
        
        # 如果超过60%的行包含对话标识，认为是按行组织的
        return valid_lines > 0 and (dialogue_lines / valid_lines) > 0.6
    
    def _split_sentences(self, text: str) -> List[str]:
        """智能分句"""
        pattern = f"([{''.join(self.sentence_endings)}])"
        parts = re.split(pattern, text)
        
        sentences = []
        current_sentence = ""
        
        for part in parts:
            if part in self.sentence_endings:
                current_sentence += part
                sentences.append(current_sentence)
                current_sentence = ""
            else:
                current_sentence += part
        
        if current_sentence.strip():
            sentences.append(current_sentence)
        
        return sentences
    
    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """后处理分块结果"""
        processed = []
        for chunk in chunks:
            if len(chunk.strip()) >= 5:  # 只过滤极短的块
                processed.append(chunk.strip())
        return processed

class AncientTextAnalyzer:
    """古文分析器 - 通用化设计"""
    
    def __init__(self):
        self.speaker_patterns = self._load_speaker_patterns()
        self.topic_keywords = self._load_topic_keywords()
    
    def _load_speaker_patterns(self) -> Dict[str, List[str]]:
        """加载说话人识别模式 - 可扩展"""
        return {
            # 论语模式
            'lunyu': {
                '孔子': ['子曰', '孔子曰', '孔子谓', '子谓'],
                '弟子': ['有子曰', '曾子曰', '子夏曰', '子游曰', '子贡曰', '子路曰', '颜渊曰'],
                '问者': ['问曰', '问于', '问：']
            },
            # 孟子模式
            'mengzi': {
                '孟子': ['孟子曰', '孟子谓'],
                '对话者': ['王问', '公问', '或问']
            },
            # 通用模式
            'general': {
                '作者': ['曰', '云', '言'],
                '引用': ['《', '经曰', '传曰'],
                '对话': ['问', '答', '对']
            }
        }
    
    def _load_topic_keywords(self) -> Dict[str, List[str]]:
        """加载话题关键词 - 可扩展"""
        return {
            '学习教育': ['学', '习', '教', '诲', '知', '智', '问', '思', '学而', '教学'],
            '品德修养': ['仁', '义', '礼', '智', '信', '德', '善', '修', '养', '品'],
            '政治治理': ['政', '君', '臣', '民', '国', '治', '邦', '王', '天下', '朝'],
            '人际关系': ['友', '朋', '交', '人', '亲', '群', '和', '睦', '信任'],
            '人生哲学': ['道', '天', '命', '生', '死', '乐', '忧', '志', '理想'],
            '社会礼仪': ['礼', '乐', '祭', '丧', '婚', '冠', '仪', '俗'],
            '经济生活': ['财', '货', '利', '商', '农', '工', '贸', '富'],
            '军事战争': ['兵', '战', '军', '武', '征', '伐', '守', '攻'],
            '文学艺术': ['诗', '书', '画', '乐', '文', '词', '赋', '雅'],
            '自然天象': ['天', '地', '日', '月', '星', '雨', '风', '山', '水']
        }
    
    def identify_speaker(self, text: str, book_type: str = 'general') -> str:
        """识别说话人 - 支持不同古文类型"""
        text = text.strip()
        
        # 根据书籍类型选择模式
        patterns = self.speaker_patterns.get(book_type, self.speaker_patterns['general'])
        
        for speaker_type, speaker_patterns in patterns.items():
            for pattern in speaker_patterns:
                if text.startswith(pattern):
                    # 进一步细化识别
                    if speaker_type == '弟子' and pattern in ['有子曰', '曾子曰']:
                        return pattern.replace('曰', '')
                    elif speaker_type == '孔子' or pattern == '子曰':
                        return '孔子'
                    else:
                        return speaker_type
        
        # 通用对话模式识别
        dialogue_match = re.search(r'([^问曰谓]{1,10})[问曰谓]', text)
        if dialogue_match:
            return dialogue_match.group(1)
        
        return '作者'
    
    def classify_topic(self, text: str) -> str:
        """分类话题 - 基于关键词匹配和语义分析"""
        topic_scores = defaultdict(int)
        
        # 关键词匹配
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    topic_scores[topic] += 1
        
        # 返回得分最高的话题
        if topic_scores:
            return max(topic_scores.items(), key=lambda x: x[1])[0]
        
        return '其他'
    
    def extract_context(self, text: str, full_text: str, window_size: int = 100) -> str:
        """提取上下文"""
        try:
            start_idx = full_text.find(text)
            if start_idx == -1:
                return text
            
            context_start = max(0, start_idx - window_size)
            context_end = min(len(full_text), start_idx + len(text) + window_size)
            context = full_text[context_start:context_end]
            
            # 标记当前文本
            context = context.replace(text, f"**{text}**")
            return context
        except:
            return text
        
class BGEReranker:
    """BGE重排序器"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model = None
        self.model_name = model_name
        self.is_loaded = False
        
        if HAS_RERANKER:
            self._load_model()
    
    def _load_model(self):
        """延迟加载模型"""
        try:
            # 显示更详细的加载信息
            with st.spinner(f"正在加载重排序模型 {self.model_name}..."):
                st.info(f"🤖 加载重排序模型: {self.model_name}")
                self.model = CrossEncoder(self.model_name)
                self.is_loaded = True
                st.success(f"✅ 重排序模型 {self.model_name} 加载成功")
        except Exception as e:
            st.error(f"❌ 重排序模型加载失败: {e}")
            st.info("💡 请确保已安装: pip install sentence-transformers torch")
            self.is_loaded = False
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """重排序候选结果"""
        if not self.is_loaded or not self.model:
            st.warning("重排序模型未加载，返回原始结果")
            return candidates[:top_k]
        
        try:
            # 准备查询-文档对
            pairs = [(query, candidate['content']) for candidate in candidates]
            
            # 获取重排序分数
            scores = self.model.predict(pairs)
            
            # 更新候选结果的分数
            for i, candidate in enumerate(candidates):
                candidate['rerank_score'] = float(scores[i])
            
            # 按重排序分数排序
            reranked_results = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
            
            return reranked_results[:top_k]
            
        except Exception as e:
            st.error(f"重排序过程出错: {e}")
            return candidates[:top_k]

class HybridRetriever:
    """混合检索器 - BM25 + 向量检索"""
    
    def __init__(self, vector_collection, segments: List[AncientTextSegment]):
        self.vector_collection = vector_collection
        self.segments = segments
        self.bm25 = None
        self.segment_map = {seg.segment_id: seg for seg in segments}
        self._build_bm25_index()
        
    
    def _build_bm25_index(self):
        """构建BM25索引"""
        try:
            # 分词
            corpus = []
            for segment in self.segments:
                tokens = list(jieba.cut(segment.content))
                corpus.append(tokens)
            
            if corpus and HAS_BM25:
                self.bm25 = BM25Okapi(corpus)
        except Exception as e:
            st.error(f"构建BM25索引失败: {e}")
            self.bm25 = None
    
    def hybrid_search(self, query: str, top_k: int = 10, 
                     bm25_weight: float = None, vector_weight: float = None) -> List[Dict[str, Any]]:
        """混合检索 - 动态检查重排序设置"""
        
        # 动态获取重排序设置
        use_reranker = st.session_state.get('use_reranker', False) and HAS_RERANKER
        
        if bm25_weight is None or vector_weight is None:
            bm25_w, vector_w = self._adaptive_weights(query)
        else:
            bm25_w, vector_w = bm25_weight, vector_weight
        
        # 动态重排序
        if use_reranker:
            return self._hybrid_search_with_rerank(
                query, top_k, initial_k=min(50, top_k*3), 
                bm25_weight=bm25_w, vector_weight=vector_w
            )
        
        # 原有逻辑
        if not HAS_BM25 or not self.bm25:
            return self._vector_search(query, top_k)
        
        bm25_results = self._bm25_search(query, top_k * 2)
        vector_results = self._vector_search(query, top_k * 2)
        combined_results = self._combine_results(
            bm25_results, vector_results, bm25_w, vector_w
        )
        
        return combined_results[:top_k]
    
    def _hybrid_search_with_rerank(self, query: str, top_k: int = 10, 
                                 initial_k: int = 50,
                                 bm25_weight: float = 0.3, 
                                 vector_weight: float = 0.7) -> List[Dict[str, Any]]:
        """带重排序的混合检索 - 动态加载重排序器"""
        
        # 获取候选结果
        if not HAS_BM25 or not self.bm25:
            candidates = self._vector_search(query, initial_k)
        else:
            bm25_results = self._bm25_search(query, initial_k)
            vector_results = self._vector_search(query, initial_k)
            candidates = self._combine_results(
                bm25_results, vector_results, bm25_weight, vector_weight
            )
        
        # 动态创建重排序器
        model_name = st.session_state.get('reranker_model', 'BAAI/bge-reranker-base')
        reranker = BGEReranker(model_name)
        
        if reranker.is_loaded:
            return reranker.rerank(query, candidates[:initial_k], top_k)
        else:
            return candidates[:top_k]
    
    def _adaptive_weights(self, query: str) -> Tuple[float, float]:
        """自适应权重调节"""
        # 实体查询检测（包含具体的人名、概念等）
        entity_keywords = ['孔子', '孟子', '仁', '义', '礼', '智', '信', '君子', '小人']
        
        # 概念查询检测
        concept_keywords = ['如何', '什么是', '为什么', '怎样', '方法', '态度', '思想']
        
        entity_score = sum(1 for keyword in entity_keywords if keyword in query)
        concept_score = sum(1 for keyword in concept_keywords if keyword in query)
        
        if entity_score > concept_score:
            # 实体查询：提高BM25权重
            return 0.6, 0.4
        elif concept_score > entity_score:
            # 概念查询：提高向量权重
            return 0.2, 0.8
        else:
            # 平衡查询：使用默认权重
            return 0.3, 0.7
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """BM25检索"""
        if not self.bm25 or not HAS_BM25:
            return []
        
        try:
            query_tokens = list(jieba.cut(query))
            scores = self.bm25.get_scores(query_tokens)
            
            # 获取top_k结果
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.segments):
                    segment = self.segments[idx]
                    results.append({
                        'segment_id': segment.segment_id,
                        'content': segment.content,
                        'metadata': asdict(segment),
                        'bm25_score': float(scores[idx]),
                        'vector_score': 0.0
                    })
            
            return results
        except Exception as e:
            st.error(f"BM25检索错误: {e}")
            return []
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """向量检索"""
        try:
            results = self.vector_collection.query(
                query_texts=[query],
                n_results=min(top_k, self.vector_collection.count())
            )
            
            vector_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    segment_id = results['ids'][0][i]
                    segment = self.segment_map.get(segment_id)
                    if segment:
                        similarity = 1 - results['distances'][0][i] if results['distances'] else 1.0
                        vector_results.append({
                            'segment_id': segment_id,
                            'content': doc,
                            'metadata': asdict(segment),
                            'bm25_score': 0.0,
                            'vector_score': float(similarity)
                        })
            
            return vector_results
        except Exception as e:
            st.error(f"向量检索错误: {e}")
            return []
    
    def _combine_results(self, bm25_results: List[Dict], vector_results: List[Dict],
                        bm25_weight: float, vector_weight: float) -> List[Dict[str, Any]]:
        """合并检索结果"""
        # 归一化分数
        bm25_results = self._normalize_scores(bm25_results, 'bm25_score')
        vector_results = self._normalize_scores(vector_results, 'vector_score')
        
        # 合并结果
        combined = {}
        
        for result in bm25_results:
            sid = result['segment_id']
            combined[sid] = result.copy()
            combined[sid]['combined_score'] = result['bm25_score'] * bm25_weight
        
        for result in vector_results:
            sid = result['segment_id']
            if sid in combined:
                combined[sid]['vector_score'] = result['vector_score']
                combined[sid]['combined_score'] += result['vector_score'] * vector_weight
            else:
                combined[sid] = result.copy()
                combined[sid]['combined_score'] = result['vector_score'] * vector_weight
        
        # 按综合分数排序
        sorted_results = sorted(combined.values(), 
                              key=lambda x: x['combined_score'], reverse=True)
        
        return sorted_results
    
    def _normalize_scores(self, results: List[Dict], score_key: str) -> List[Dict]:
        """归一化分数"""
        if not results:
            return results
        
        scores = [r[score_key] for r in results]
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 0.0
        
        if max_score == min_score:
            return results
        
        for result in results:
            result[score_key] = (result[score_key] - min_score) / (max_score - min_score)
        
        return results

class UniversalAncientRAG:
    """通用古文RAG系统"""
    
    def __init__(self, embedding_model: str = "BAAI/bge-large-zh-v1.5"):
        self.client = chromadb.Client()
        self.collection_name = "ancient_texts_collection"
        self.embedding_model = embedding_model
        self.segments: List[AncientTextSegment] = []
        self.chunker = SmartTextChunker()
        self.analyzer = AncientTextAnalyzer()
        self.retriever: Optional[HybridRetriever] = None
        
        # 配置embedding函数
        self._setup_embedding_function()
        
        # 初始化向量数据库
        self._initialize_collection()
    
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
        """处理单个篇章（严格按行分块版）"""
        segments = []
        
        # 使用严格按行分块，不合并
        chunks = self.chunker.chunk_by_semantic_units(content)
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 5:  # 跳过过短的块
                continue
                
            # 分析文本
            speaker = self.analyzer.identify_speaker(chunk, book.lower())
            topic = self.analyzer.classify_topic(chunk)
            context = self.analyzer.extract_context(chunk, content)
            
            # 创建片段ID，使用更精确的命名
            segment_id = f"{book}_{chapter}_{i:03d}"
            
            # 扩展元数据
            metadata = {
                'length': len(chunk),
                'position': i,
                'book_type': self._detect_book_type(book),
                'file_path': file_path,
                'chunk_count': len(chunks),
                'has_dialogue': '曰' in chunk or '问' in chunk,
                'classical_terms': self._extract_classical_terms(chunk),
                'chunking_method': 'strict_line_based',  # 标记为严格按行分块
                'original_line': True  # 标记为保持原始行结构
            }
            
            segment = AncientTextSegment(
                book=book,
                chapter=chapter,
                speaker=speaker,
                content=chunk,
                topic=topic,
                segment_id=segment_id,
                context=context,
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
    
    
    def _detect_book_type(self, book_name: str) -> str:
        """检测书籍类型"""
        book_types = {
            '论语': 'lunyu',
            '孟子': 'mengzi',
            '大学': 'general',
            '中庸': 'general'
        }
        return book_types.get(book_name, 'general')
    
    
    def _build_vector_database(self):
        """构建向量数据库（修复删除错误）"""
        # 清空旧数据 - 修复版本
        try:
            if self.collection.count() > 0:
                # 方法1：获取所有ID然后删除
                all_data = self.collection.get()
                if all_data['ids']:
                    self.collection.delete(ids=all_data['ids'])
            
            # 或者使用方法2：重新创建collection（更简单）
            # try:
            #     self.client.delete_collection(self.collection_name)
            # except:
            #     pass  # collection可能不存在
            # self.collection = self.client.create_collection(
            #     name=self.collection_name,
            #     metadata={"description": "通用古文向量数据库"}
            # )
            
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
        batch_size = 200
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
                    'speaker': seg.speaker,
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
        
    def search(self, query: str, top_k: int = 5, 
              search_mode: str = 'hybrid') -> List[Dict[str, Any]]:
        """搜索功能"""
        if not self.retriever:
            st.error("检索器未初始化，请先加载数据")
            return []
        
        try:
            if search_mode == 'hybrid':
                if HAS_BM25:
                    return self.retriever.hybrid_search(query, top_k)
                else:
                    st.info("BM25不可用，自动切换到向量检索模式")
                    return self.retriever._vector_search(query, top_k)
            elif search_mode == 'vector':
                return self.retriever._vector_search(query, top_k)
            elif search_mode == 'bm25':
                if HAS_BM25:
                    return self.retriever._bm25_search(query, top_k)
                else:
                    st.warning("BM25检索不可用，请安装 rank-bm25 包")
                    return []
            else:
                return []
        except Exception as e:
            st.error(f"搜索过程中出现错误: {e}")
            return []
        
    def optimize_query(self, raw_query: str, api_key: str) -> Tuple[str, str]:
        """优化用户输入的查询"""
        if not api_key or not api_key.strip():
            return raw_query, "未使用API优化"
        
        prompt = f"""你是古文检索专家。请将用户的问题转换为更适合古文检索的查询。

    用户原始问题：{raw_query}

    请按以下要求优化：
    1. 提取核心关键词（古文术语、人物、概念）
    2. 补充相关的古典表述
    3. 保持问题的核心意图
    4. 生成简洁但精准的检索查询

    只返回优化后的查询，不要解释。

    优化查询："""

        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key.strip()}'
            }
            
            data = {
                'model': 'deepseek-chat',
                'messages': [
                    {'role': 'user', 'content': prompt}
                ],
                'stream': False,
                'max_tokens': 200,
                'temperature': 0.3
            }
            
            response = requests.post(
                'https://api.deepseek.com/chat/completions',
                headers=headers,
                json=data,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                optimized_query = result['choices'][0]['message']['content'].strip()
                return optimized_query, "API优化成功"
            else:
                return raw_query, f"API调用失败: {response.status_code}"
                
        except Exception as e:
            return raw_query, f"优化失败: {str(e)}"
    
    def generate_answer(self, query: str, context: List[Dict[str, Any]], 
                       api_key: str = None) -> str:
        """生成答案"""
        if not context:
            return "抱歉，没有找到相关的古文内容。"
        
        # 构建增强的上下文
        context_parts = []
        for item in context[:3]:
            meta = item['metadata']
            context_parts.append(
                f"【{meta['book']} · {meta['chapter']}】{meta['speaker']}: {item['content']}"
            )
        
        context_text = "\n\n".join(context_parts)
        
        # 构建提示词
        prompt = f"""你是一位精通中国古典文学的学者，请基于以下古文原文回答用户的问题。

相关古文原文：
{context_text}

用户问题：{query}

请按以下要求回答：
1. 首先引用最相关的原文
2. 详细完整地回答用户问题
3. 解释古文的字面含义
4. 阐述其深层思想内涵
5. 结合现代观点进行分析
6. 提供实际的指导意义

回答要求：第二部分详细回答，其余回答要简洁明了，重点突出，既有学术深度又通俗易懂。

回答："""

        # 调用API或生成基础回答
        if api_key and api_key.strip():
            return self._call_api(prompt, api_key.strip())
        else:
            return self._generate_basic_answer(query, context)
    
    def _call_api(self, prompt: str, api_key: str) -> str:
        """调用AI API"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            
            data = {
                'model': 'deepseek-chat',
                'messages': [
                    {'role': 'system', 'content': '你是一位专业的古典文学学者。'},
                    {'role': 'user', 'content': prompt}
                ],
                'stream': False,
                'max_tokens': 1500,
                'temperature': 0.7
            }
            
            response = requests.post(
                'https://api.deepseek.com/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"API调用失败: {response.status_code}"
                
        except Exception as e:
            return f"API调用出错: {str(e)}"
    
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
        .speaker-tag {{ background: #28a745; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-right: 5px; }}
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
                        <span class="speaker-tag">{seg.speaker}</span>
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
                    seg.speaker,
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
                        f.write(f"说话人：{seg.speaker}\n")
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
        speakers_stats = {}
        length_stats = []
        
        for seg in self.segments:
            # 书籍统计
            books_stats[seg.book] = books_stats.get(seg.book, 0) + 1
            
            # 话题统计
            topics_stats[seg.topic] = topics_stats.get(seg.topic, 0) + 1
            
            # 说话人统计
            speakers_stats[seg.speaker] = speakers_stats.get(seg.speaker, 0) + 1
            
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
            
            f.write("👤 说话人分布\n")
            f.write("-" * 20 + "\n")
            for speaker, count in sorted(speakers_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_segments) * 100
                f.write(f"{speaker}：{count} 段 ({percentage:.1f}%)\n")
        
        st.info(f"📈 统计报告：{stats_file.name}")


def main():
    """主应用"""
    st.title("📜 通用古文智能问答系统")
    st.markdown("*基于混合检索技术的智能古文RAG系统*")
    st.markdown("---")
    
    # 初始化系统
    if 'rag_system' not in st.session_state:
        # 初始化时指定中文优化的embedding模型
        st.session_state.rag_system = UniversalAncientRAG(
            embedding_model="BAAI/bge-large-zh-v1.5"
        )
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # 数据加载
        st.subheader("📚 数据管理")
        data_dir = st.text_input("古文数据目录路径", value="./论语")
        
        # 文件格式选择
        file_formats = st.multiselect(
            "支持的文件格式",
            ['.txt', '.md', '.text', '.doc'],
            default=['.txt', '.md'],
            help="选择要加载的文件格式"
        )
        
        # 高级选项
        with st.expander("🔧 高级选项"):
            recursive_load = st.checkbox("递归加载子目录", value=True, help="是否处理所有子目录中的文件")
            min_content_length = st.slider("最小内容长度", 5, 100, 10, help="忽略过短的文件")
            show_processing_details = st.checkbox("显示处理详情", value=True)
        
        if st.button("🔄 加载古文数据", type="primary"):
            with st.spinner("正在加载古文数据..."):
                # 清空之前的数据
                st.session_state.rag_system.segments = []
                
                if recursive_load:
                    count = st.session_state.rag_system.load_from_directory(
                        data_dir, 
                        file_extensions=file_formats
                    )
                else:
                    count = st.session_state.rag_system.load_single_directory(
                        data_dir, 
                        file_extensions=file_formats
                    )
                
                if count > 0:
                    st.success(f"✅ 成功加载 {count} 个文本片段！")
                    st.balloons()  # 添加庆祝动画
                else:
                    st.error("❌ 数据加载失败，请检查目录路径和文件格式")
        
        # API配置
        st.subheader("🤖 AI配置")
        api_key = st.text_input("DeepSeek API Key", type="password")

        # API配置部分后添加
        query_optimization = st.checkbox(
            "🔍 智能查询优化", 
            value=bool(api_key and api_key.strip()),  # 有API KEY时默认开启
            disabled=not bool(api_key and api_key.strip()),  # 无API KEY时禁用
            help="使用AI优化用户输入的问题，提高检索准确性"
        )
        st.session_state.query_optimization = query_optimization
        
        # 检索配置
        st.subheader("🔍 检索设置")

        # Embedding模型选择
        embedding_models = [
            "BAAI/bge-large-zh-v1.5",
            # "BAAI/bge-base-zh-v1.5", 
            # "text2vec-chinese",
            # "multilingual-e5-large",
            # "all-MiniLM-L6-v2"  # 英文模型作为对比
        ]
        
        selected_model = st.selectbox(
            "Embedding模型",
            embedding_models,
            help="选择向量化模型，中文模型对古文效果更好"
        )
        
        # 检查是否需要切换模型
        current_model = getattr(st.session_state.rag_system, 'embedding_model', 'default')
        if selected_model != current_model:
            if st.button("🔄 切换Embedding模型"):
                with st.spinner("正在切换模型并重建索引..."):
                    st.session_state.rag_system.change_embedding_model(selected_model)
        
        # 显示当前模型
        if hasattr(st.session_state.rag_system, 'embedding_model'):
            st.info(f"当前模型: {st.session_state.rag_system.embedding_model}")
        search_mode = st.selectbox(
            "检索模式",
            ["hybrid", "vector", "bm25"],
            format_func=lambda x: {"hybrid": "混合检索", "vector": "向量检索", "bm25": "关键词检索"}[x]
        )
        top_k = st.slider("返回结果数量", 1, 20, 5)
        
        # 添加重排序选项（关键修改）
        if HAS_RERANKER:
            use_reranker = st.checkbox(
                "🔄 启用深度重排序", 
                value=True, 
                help="使用BGE模型进行深度语义重排序，提高结果准确性"
            )
            st.session_state.use_reranker = use_reranker
            
            if use_reranker:
                reranker_model = st.selectbox(
                    "重排序模型",
                    ["BAAI/bge-reranker-base"],
                    help="选择重排序模型"
                )
                st.session_state.reranker_model = reranker_model
        else:
            st.warning("⚠️ 需要安装 sentence-transformers 才能使用重排序功能")
            st.code("pip install sentence-transformers torch")
            st.session_state.use_reranker = False
        
        # 显示统计信息
        if hasattr(st.session_state.rag_system, 'segments') and st.session_state.rag_system.segments:
            st.subheader("📊 数据统计")
            segments = st.session_state.rag_system.segments
            
            # 基本统计
            st.metric("文本片段总数", len(segments))
            
            # 书籍分布
            books = defaultdict(int)
            topics = defaultdict(int)
            speakers = defaultdict(int)
            
            for seg in segments:
                books[seg.book] += 1
                topics[seg.topic] += 1
                speakers[seg.speaker] += 1
            
            st.write("📖 **书籍分布**")
            for book, count in sorted(books.items(), key=lambda x: x[1], reverse=True):
                st.write(f"- {book}: {count} 段")
            
            st.write("🏷️ **话题分布**")
            for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.write(f"- {topic}: {count} 段")
        
        # 在数据统计部分后添加保存功能
        if hasattr(st.session_state.rag_system, 'segments') and st.session_state.rag_system.segments:
            st.subheader("💾 结果导出")
            
            # 输出目录设置
            output_dir = st.text_input(
                "保存目录", 
                value="./processing_results",
                help="处理结果将保存到此目录"
            )
            
            # 保存选项
            save_options = st.multiselect(
                "选择保存格式",
                ["HTML报告", "CSV表格", "JSON数据", "按书籍分类", "统计报告"],
                default=["HTML报告", "CSV表格", "统计报告"],
                help="选择要生成的文件格式"
            )
            
            if st.button("📥 导出处理结果", type="secondary"):
                with st.spinner("正在保存处理结果..."):
                    success = st.session_state.rag_system.save_processing_results(output_dir)
                    if success:
                        st.balloons()
                        
                        # 显示保存的文件信息
                        st.success("🎉 结果导出成功！")
                        st.info(f"""
                        **已生成以下文件：**
                        - 📄 详细JSON数据
                        - 🌐 HTML可视化报告  
                        - 📊 CSV表格文件
                        - 📚 按书籍分类的文本
                        - 📈 统计分析报告
                        
                        **保存位置：** `{output_dir}`
                        """)
    
    # 主界面
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 智能问答")
        
        # 预设问题
        st.subheader("📋 示例问题")
        example_questions = [
            "什么是仁？",
            "孔子的教育思想",
            "君子与小人的区别",
            "如何修身养性？",
            "古人的政治理想",
            "学习的方法和态度",
            "人际交往的智慧",
            "面对困难的态度"
        ]
        
        selected_question = st.selectbox(
            "选择示例问题或输入自定义问题：",
            [""] + example_questions
        )
        
        # 问题输入
        query = st.text_input(
            "您的问题",
            value=selected_question if selected_question else "",
            placeholder="请输入您想了解的古文相关问题..."
        )
        
        # 搜索按钮
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            search_btn = st.button("🔍 智能搜索", type="primary")
        with col_btn2:
            clear_btn = st.button("🗑️ 清空结果")
        
        if clear_btn:
            st.session_state.last_results = None
            st.rerun()
        
        # 处理搜索
        if search_btn and query:
            # 检查是否启用查询优化
            use_optimization = st.session_state.get('query_optimization', False)
            
            if use_optimization and api_key and api_key.strip():
                with st.spinner("正在优化查询..."):
                    optimized_query, optimization_status = st.session_state.rag_system.optimize_query(query, api_key)
                    
                # 显示优化结果
                if optimized_query != query:
                    st.info(f"🔍 **查询优化**: {optimization_status}")
                    st.write(f"**原始问题**: {query}")
                    st.write(f"**优化查询**: {optimized_query}")
                    actual_query = optimized_query
                else:
                    actual_query = query
                    if optimization_status != "未使用API优化":
                        st.warning(f"查询优化: {optimization_status}")
            else:
                actual_query = query
            
            with st.spinner("正在搜索相关古文..."):
                results = st.session_state.rag_system.search(
                    actual_query, top_k, search_mode
                )
                st.session_state.last_results = (query, results, actual_query if use_optimization else None)
        
        # 在搜索结果显示中添加重排序分数
        if hasattr(st.session_state, 'last_results') and st.session_state.last_results:
            # display_query, results = st.session_state.last_results

            if len(st.session_state.last_results) == 3:
                original_query, results, optimized_query = st.session_state.last_results
                display_query = optimized_query if optimized_query else original_query
            else:
                # 向后兼容
                display_query, results = st.session_state.last_results[:2]
            
            if results:
                st.subheader(f"📖 「{display_query}」相关古文")
                
                for i, result in enumerate(results):
                    meta = result['metadata']
                                    
                    # 构建标题，包含分数
                    title_parts = [f"📜 《{meta['book']}·{meta['chapter']}》"]

                    if 'rerank_score' in result and result['rerank_score'] > 0:
                        title_parts.append(f"(重排序: {result['rerank_score']:.3f})")
                    elif 'combined_score' in result and result['combined_score'] > 0:
                        title_parts.append(f"(综合: {result['combined_score']:.3f})")
                    elif 'vector_score' in result and result['vector_score'] > 0:
                        title_parts.append(f"(相似度: {result['vector_score']:.3f})")
                    elif 'bm25_score' in result and result['bm25_score'] > 0:
                        title_parts.append(f"(匹配度: {result['bm25_score']:.3f})")
                    else:
                        title_parts.append("(相关)")
                    
                    with st.expander(" ".join(title_parts)):
                        st.write(f"**原文内容**：{result['content']}")
                        st.write(f"**话题分类**：{meta['topic']}")
                        
                        # 显示各种分数
                        if search_mode == 'hybrid' or 'rerank_score' in result:
                            score_cols = st.columns(4)
                            with score_cols[0]:
                                if 'bm25_score' in result:
                                    st.metric("BM25", f"{result['bm25_score']:.3f}")
                            with score_cols[1]:
                                if 'vector_score' in result:
                                    st.metric("向量", f"{result['vector_score']:.3f}")
                            with score_cols[2]:
                                if 'combined_score' in result:
                                    st.metric("融合", f"{result['combined_score']:.3f}")
                            with score_cols[3]:
                                if 'rerank_score' in result:
                                    st.metric("重排序", f"{result['rerank_score']:.3f}")
                        
                        # 上下文显示
                        if 'context' in meta and meta['context'] != result['content']:
                            st.write("**📄 上下文：**")
                            st.text_area("", meta['context'], height=100, disabled=True, key=f"context_{i}")

                # 生成智能回答
                st.subheader("🤖 智能解答")
                with st.spinner("正在生成智能回答..."):
                    answer = st.session_state.rag_system.generate_answer(
                        original_query, results, api_key
                    )
                    st.markdown(answer)
    with col2:
        st.header("ℹ️ 系统信息")
        
        # 技术特点
        st.info("""
        **🔧 核心技术**
        - ✅ 智能语义分块
        - ✅ 混合检索 (BM25+向量)
        - ✅ 多源数据融合
        - ✅ 上下文感知
        - 🔑 AI智能问答
        """)
        
        # 支持的古文类型
        st.subheader("📚 支持的古文")
        st.markdown("""
        - 📖 **经典**: 论语、孟子、大学、中庸
        - 📜 **史书**: 史记、汉书等历史文献
        - 🎭 **文学**: 诗经、楚辞等文学作品
        - ⚖️ **法家**: 韩非子、商君书等
        - 🏛️ **其他**: 各类古代典籍
        
        *系统采用通用化设计，可适配各种古文格式*
        """)
        
        # 使用说明
        st.subheader("💡 使用指南")
        st.markdown("""
        **📝 数据准备**：
        1. 按 `书名/篇章/text.txt` 组织文件
        2. 确保文本编码为 UTF-8
        
        **🔍 检索模式**：
        - **混合检索**: 综合关键词+语义
        - **向量检索**: 纯语义相似度
        - **关键词检索**: 传统BM25算法
        
        **🎯 提问技巧**：
        - 使用古文中的关键概念
        - 可以询问思想、人物、事件
        - 支持现代语言表达
        """)

if __name__ == "__main__":
    # 自定义样式
    st.markdown("""
    <style>
    .stApp {
        background-color: #fafafa;
    }
    .stButton>button {
        border-radius: 25px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .stExpander {
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
    }
    .stSelectbox>div>div>div {
        border-radius: 10px;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    main()