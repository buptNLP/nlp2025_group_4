import streamlit as st
import numpy as np
import jieba
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
from config import HAS_BM25, HAS_RERANKER
from models import AncientTextSegment

if HAS_BM25:
    from rank_bm25 import BM25Okapi

if HAS_RERANKER:
    from sentence_transformers import CrossEncoder

        
class BGEReranker:
    """BGE重排序器 - 修复版"""
    
    # 类级别的模型缓存，避免重复加载
    _model_cache = {}
    _loading_status = {}
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.model_name = model_name
        self.model = None
        self.is_loaded = False
        
        # 检查是否已经加载过该模型
        if model_name in self._model_cache:
            self.model = self._model_cache[model_name]
            self.is_loaded = self._loading_status[model_name]
            if self.is_loaded:
                st.info(f"✅ 使用已缓存的重排序模型: {model_name}")
    
    def _load_model(self):
        """延迟加载模型 - 改进版"""
        if not HAS_RERANKER:
            st.warning("⚠️ 重排序功能需要安装 sentence-transformers")
            st.code("pip install sentence-transformers torch")
            return False
        
        # 如果已经在加载中，直接返回
        if self.model_name in self._loading_status and self._loading_status[self.model_name]:
            return True
        
        # 如果已经加载过且失败了，不再重试
        if self.model_name in self._model_cache and not self._loading_status[self.model_name]:
            st.warning(f"重排序模型 {self.model_name} 之前加载失败，跳过重排序")
            return False
        
        try:
            with st.spinner(f"首次加载重排序模型 {self.model_name}，请稍候..."):
                self.model = CrossEncoder(self.model_name)
                
                # 缓存模型和状态
                self._model_cache[self.model_name] = self.model
                self._loading_status[self.model_name] = True
                self.is_loaded = True
                
                st.success(f"✅ 重排序模型加载成功: {self.model_name}")
                return True
                
        except Exception as e:
            error_msg = f"重排序模型加载失败: {str(e)}"
            st.error(f"❌ {error_msg}")
            
            # 缓存失败状态，避免重复尝试
            self._model_cache[self.model_name] = None
            self._loading_status[self.model_name] = False
            self.is_loaded = False
            
            # 提供解决建议
            if "No module named" in str(e):
                st.info("💡 请安装依赖: pip install sentence-transformers torch")
            elif "CUDA" in str(e) or "GPU" in str(e):
                st.info("💡 GPU相关错误，请检查CUDA环境或切换到CPU模式")
            elif "timeout" in str(e).lower():
                st.info("💡 网络超时，请检查网络连接或使用本地模型")
            
            return False
    
    def ensure_loaded(self) -> bool:
        """确保模型已加载"""
        if self.is_loaded and self.model is not None:
            return True
        return self._load_model()
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """重排序候选结果 - 改进版"""
        # 检查候选结果
        if not candidates:
            return []
        
        # 确保模型已加载
        if not self.ensure_loaded():
            st.warning("重排序模型不可用，返回原始排序结果")
            return candidates[:top_k]
        
        try:
            # 限制候选数量，避免处理过多结果
            max_candidates = min(len(candidates), 100)  # 最多处理100个候选
            limited_candidates = candidates[:max_candidates]
            
            # 准备查询-文档对
            pairs = [(query, candidate['content']) for candidate in limited_candidates]
            
            # 获取重排序分数
            with st.spinner("正在执行深度重排序..."):
                scores = self.model.predict(pairs)
            
            # 更新候选结果的分数
            for i, candidate in enumerate(limited_candidates):
                candidate['rerank_score'] = float(scores[i])
                # 保留原始分数用于调试
                candidate['rerank_score_raw'] = float(scores[i])
            
            # 按重排序分数排序
            reranked_results = sorted(
                limited_candidates, 
                key=lambda x: x['rerank_score'], 
                reverse=True
            )
            
            # 为未重排序的结果添加标记
            for candidate in candidates[max_candidates:]:
                candidate['rerank_score'] = 0.0
                candidate['rerank_note'] = "未重排序"
            
            # 合并结果
            final_results = reranked_results + candidates[max_candidates:]
            
            return final_results[:top_k]
            
        except Exception as e:
            st.error(f"重排序过程出错: {e}")
            st.warning("将返回原始排序结果")
            return candidates[:top_k]
             

class HybridRetriever:
    """混合检索器 - BM25 + 向量检索"""
    
    def __init__(self, vector_collection, segments: List[AncientTextSegment]):
        self.vector_collection = vector_collection
        self.segments = segments
        self.bm25 = None
        self.segment_map = {seg.segment_id: seg for seg in segments}
        self._build_bm25_index()
        self._reranker_cache = {}  # 重排序器缓存

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

    def _get_reranker(self, model_name: str = None) -> Optional[BGEReranker]:
        """获取重排序器实例（带缓存）"""
        if not HAS_RERANKER:
            return None
        
        # 使用默认模型名称
        if model_name is None:
            model_name = st.session_state.get('reranker_model', 'BAAI/bge-reranker-large')
        
        # 检查缓存
        if model_name not in self._reranker_cache:
            self._reranker_cache[model_name] = BGEReranker(model_name)
        
        return self._reranker_cache[model_name]
    
    def _should_use_reranker(self) -> bool:
        """判断是否应该使用重排序"""
        # 检查全局开关
        if not st.session_state.get('use_reranker', False):
            return False
        
        # 检查依赖
        if not HAS_RERANKER:
            return False
        
        return True
    
    def hybrid_search_with_filter(self, query: str, top_k: int = 10, 
                                metadata_filter: Dict[str, str] = None,
                                bm25_weight: float = None,
                                vector_weight: float = None) -> List[Dict[str, Any]]:
        """带元数据过滤的混合搜索 - 修复版"""
        
        # 检查是否应该使用重排序
        if self._should_use_reranker():
            return self._hybrid_search_with_rerank(
                query, top_k, 
                initial_k=min(50, max(top_k*3, 20)),  # 动态调整初始候选数
                metadata_filter=metadata_filter,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight
            )
        
        # 不使用重排序的常规混合搜索
        bm25_w = bm25_weight if bm25_weight is not None else 0.3
        vector_w = vector_weight if vector_weight is not None else 0.7
        
        if not HAS_BM25 or not self.bm25:
            return self.vector_search_with_filter(query, top_k, metadata_filter)
        
        bm25_results = self.bm25_search_with_filter(query, top_k * 2, metadata_filter)
        vector_results = self.vector_search_with_filter(query, top_k * 2, metadata_filter)
        
        combined_results = self._combine_results(
            bm25_results, vector_results, bm25_w, vector_w
        )
        
        return combined_results[:top_k]

    def vector_search_with_filter(self, query: str, top_k: int, 
                             metadata_filter: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """带元数据过滤的向量搜索"""
        try:
            # 构建where条件 - 修复语法
            where_condition = None
            if metadata_filter:
                if len(metadata_filter) == 1:
                    # 单个条件
                    key, value = next(iter(metadata_filter.items()))
                    where_condition = {key: {"$eq": value}}
                else:
                    # 多个条件使用$and
                    conditions = []
                    for key, value in metadata_filter.items():
                        conditions.append({key: {"$eq": value}})
                    where_condition = {"$and": conditions}
            
            results = self.vector_collection.query(
                query_texts=[query],
                n_results=min(top_k, self.vector_collection.count()),
                where=where_condition
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

    def bm25_search_with_filter(self, query: str, top_k: int, 
                            metadata_filter: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """带元数据过滤的BM25搜索"""
        if not self.bm25 or not HAS_BM25:
            return []
        
        try:
            query_tokens = list(jieba.cut(query))
            scores = self.bm25.get_scores(query_tokens)
            
            # 获取排序后的索引
            top_indices = np.argsort(scores)[::-1]
            
            results = []
            for idx in top_indices:
                if len(results) >= top_k:
                    break
                    
                if idx < len(self.segments):
                    segment = self.segments[idx]
                    
                    # 应用元数据过滤
                    if metadata_filter:
                        match = True
                        for key, value in metadata_filter.items():
                            segment_value = getattr(segment, key, None)
                            if segment_value != value:
                                match = False
                                break
                        if not match:
                            continue
                    
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
        
    def hybrid_search(self, query: str, top_k: int = 10,
                 bm25_weight: float = None,
                 vector_weight: float = None) -> List[Dict[str, Any]]:
        """基础混合搜索（无过滤）"""
        return self.hybrid_search_with_filter(
            query, top_k, 
            metadata_filter=None,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight
        )
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """基础向量搜索（无过滤）"""
        return self.vector_search_with_filter(query, top_k, metadata_filter=None)

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """基础BM25搜索（无过滤）"""
        return self.bm25_search_with_filter(query, top_k, metadata_filter=None)
    
    def _hybrid_search_with_rerank(self, query: str, top_k: int = 10, 
                                 initial_k: int = 50,
                                 bm25_weight: float = None, 
                                 vector_weight: float = None,
                                 metadata_filter: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """带重排序的混合检索 - 修复版"""
        
        # 使用传入的权重，如果没有传入则使用默认值
        bm25_w = bm25_weight if bm25_weight is not None else 0.3
        vector_w = vector_weight if vector_weight is not None else 0.7
        
        # 调整初始检索数量，确保有足够的候选
        effective_initial_k = max(initial_k, top_k * 3)
        
        # 获取候选结果
        if not HAS_BM25 or not self.bm25:
            if metadata_filter:
                candidates = self.vector_search_with_filter(query, effective_initial_k, metadata_filter)
            else:
                candidates = self._vector_search(query, effective_initial_k)
        else:
            if metadata_filter:
                bm25_results = self.bm25_search_with_filter(query, effective_initial_k, metadata_filter)
                vector_results = self.vector_search_with_filter(query, effective_initial_k, metadata_filter)
            else:
                bm25_results = self._bm25_search(query, effective_initial_k)
                vector_results = self._vector_search(query, effective_initial_k)
            
            candidates = self._combine_results(
                bm25_results, vector_results, bm25_w, vector_w
            )
        
        # 核心修复：检查候选结果
        if not candidates:
            st.info("未找到候选结果，跳过重排序")
            return []
        
        # 获取重排序器
        reranker = self._get_reranker()
        if reranker and reranker.ensure_loaded():
            st.info(f"🔄 使用重排序处理 {len(candidates)} 个候选结果")
            return reranker.rerank(query, candidates, top_k)
        else:
            st.info("重排序不可用，使用原始排序")
            return candidates[:top_k]
    
        
    def _combine_results(self, bm25_results: List[Dict], vector_results: List[Dict],
                    bm25_weight: float, vector_weight: float) -> List[Dict[str, Any]]:
        """合并检索结果（保留原始分数）"""
        # 归一化分数
        bm25_results = self._normalize_scores(bm25_results, 'bm25_score')
        vector_results = self._normalize_scores(vector_results, 'vector_score')
        
        # 合并结果
        combined = {}
        
        for result in bm25_results:
            sid = result['segment_id']
            combined[sid] = result.copy()
            combined[sid]['combined_score'] = result['bm25_score'] * bm25_weight
            # 保留原始BM25分数
            if 'bm25_score_raw' in result:
                combined[sid]['bm25_score_raw'] = result['bm25_score_raw']
        
        for result in vector_results:
            sid = result['segment_id']
            if sid in combined:
                combined[sid]['vector_score'] = result['vector_score']
                combined[sid]['combined_score'] += result['vector_score'] * vector_weight
                # 保留原始向量分数
                if 'vector_score_raw' in result:
                    combined[sid]['vector_score_raw'] = result['vector_score_raw']
            else:
                combined[sid] = result.copy()
                combined[sid]['combined_score'] = result['vector_score'] * vector_weight
        
        # 按综合分数排序
        sorted_results = sorted(combined.values(), 
                            key=lambda x: x['combined_score'], reverse=True)
        
        return sorted_results
    
    def _normalize_scores(self, results: List[Dict], score_key: str) -> List[Dict]:
        """归一化分数（保留原始分数）"""
        if not results:
            return results
        
        scores = [r[score_key] for r in results]
        max_score = max(scores) if scores else 1.0
        min_score = min(scores) if scores else 0.0
        
        if max_score == min_score:
            return results
        
        for result in results:
            # 保存原始分数
            result[f'{score_key}_raw'] = result[score_key]
            # 计算归一化分数
            result[score_key] = (result[score_key] - min_score) / (max_score - min_score)
        
        return results
