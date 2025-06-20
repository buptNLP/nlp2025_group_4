import numpy as np
import jieba
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import asdict
from backend_config import HAS_BM25, HAS_RERANKER
from models import AncientTextSegment
from backend_utils import backend_logger as st

if HAS_BM25:
    from rank_bm25 import BM25Okapi

if HAS_RERANKER:
    from sentence_transformers import CrossEncoder

        
class BGEReranker:
    """BGE重排序器 - 修复版本，避免重复加载"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.model = None
        self.model_name = model_name
        self.is_loaded = False
        self._loading = False  # 防止并发加载
        
        if HAS_RERANKER:
            self._load_model()
    
    def _load_model(self):
        """延迟加载模型 - 单次加载"""
        if self._loading:  # 防止重复加载
            return
            
        self._loading = True
        try:
            st.info(f"🤖 加载重排序模型: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            self.is_loaded = True
            st.success(f"✅ 重排序模型加载成功")
        except Exception as e:
            st.error(f"❌ 重排序模型加载失败: {e}")
            st.info("💡 请确保已安装: pip install sentence-transformers torch")
            self.is_loaded = False
        finally:
            self._loading = False
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """重排序候选结果"""
        if not self.is_loaded or not self.model:
            st.warning("重排序模型未加载，返回原始结果")
            return candidates[:top_k]
        
        if not candidates:
            return []
        
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
    """混合检索器 - 修复版本，避免重复加载模型"""
    
    def __init__(self, vector_collection, segments: List[AncientTextSegment], 
                 use_reranker: bool = None, reranker_model: str = None):
        self.vector_collection = vector_collection
        self.segments = segments
        self.bm25 = None
        self.segment_map = {seg.segment_id: seg for seg in segments}
        
        # 🔧 修复：配置重排序器，避免重复加载
        self.use_reranker = use_reranker if use_reranker is not None else HAS_RERANKER
        self.reranker_model = reranker_model or "BAAI/bge-reranker-large"
        self.reranker: Optional[BGEReranker] = None
        
        # 初始化组件
        self._build_bm25_index()
        if self.use_reranker:
            self._initialize_reranker()

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
                st.success(f"✅ BM25索引构建完成，共 {len(corpus)} 个文档")
            else:
                st.warning("⚠️ BM25依赖包未安装，将仅使用向量检索")
        except Exception as e:
            st.error(f"构建BM25索引失败: {e}")
            self.bm25 = None

    def _initialize_reranker(self):
        """初始化重排序器 - 单次加载"""
        if not HAS_RERANKER:
            st.warning("⚠️ 重排序依赖包未安装")
            self.use_reranker = False
            return
            
        try:
            # 🔧 修复：只创建一次重排序器
            self.reranker = BGEReranker(self.reranker_model)
            if self.reranker.is_loaded:
                st.success("✅ 重排序器初始化完成")
            else:
                st.warning("⚠️ 重排序器加载失败，将使用普通混合搜索")
                self.use_reranker = False
        except Exception as e:
            st.error(f"重排序器初始化失败: {e}")
            self.use_reranker = False

    def hybrid_search_with_filter(self, query: str, top_k: int = 10, 
                            metadata_filter: Dict[str, str] = None,
                            bm25_weight: float = None,
                            vector_weight: float = None) -> List[Dict[str, Any]]:
        """带元数据过滤的混合搜索 - 修复版本"""
        
        # 🔧 修复：直接使用实例变量，而不是session_state
        if self.use_reranker and self.reranker and self.reranker.is_loaded:
            return self._hybrid_search_with_rerank(
                query, top_k, initial_k=min(50, top_k*3),
                metadata_filter=metadata_filter,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight
            )
        
        # 使用传入的权重，如果没有传入则使用默认值
        bm25_w = bm25_weight if bm25_weight is not None else 0.3
        vector_w = vector_weight if vector_weight is not None else 0.7
        
        # 获取候选结果
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
            # 构建where条件
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
        """带重排序的混合检索 - 修复版本，使用已初始化的重排序器"""
        
        # 使用传入的权重，如果没有传入则使用默认值
        bm25_w = bm25_weight if bm25_weight is not None else 0.3
        vector_w = vector_weight if vector_weight is not None else 0.7
        
        # 获取候选结果
        if not HAS_BM25 or not self.bm25:
            if metadata_filter:
                candidates = self.vector_search_with_filter(query, initial_k, metadata_filter)
            else:
                candidates = self._vector_search(query, initial_k)
        else:
            if metadata_filter:
                bm25_results = self.bm25_search_with_filter(query, initial_k, metadata_filter)
                vector_results = self.vector_search_with_filter(query, initial_k, metadata_filter)
            else:
                bm25_results = self._bm25_search(query, initial_k)
                vector_results = self._vector_search(query, initial_k)
            
            candidates = self._combine_results(
                bm25_results, vector_results, bm25_w, vector_w
            )

        # 🔧 修复：使用已初始化的重排序器，而不是每次创建新的
        if self.reranker and self.reranker.is_loaded:
            return self.reranker.rerank(query, candidates[:initial_k], top_k)
        else:
            st.warning("重排序器不可用，返回普通混合搜索结果")
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

    def update_config(self, use_reranker: bool = None, reranker_model: str = None):
        """更新配置 - 用于运行时配置更改"""
        if use_reranker is not None:
            old_use_reranker = self.use_reranker
            self.use_reranker = use_reranker and HAS_RERANKER
            
            if self.use_reranker and not old_use_reranker:
                # 需要初始化重排序器
                self._initialize_reranker()
            elif not self.use_reranker and old_use_reranker:
                # 需要清理重排序器
                self.reranker = None
                st.info("🔄 已禁用重排序器")
        
        if reranker_model is not None and reranker_model != self.reranker_model:
            self.reranker_model = reranker_model
            if self.use_reranker:
                # 重新初始化重排序器
                st.info(f"🔄 切换重排序模型到: {reranker_model}")
                self._initialize_reranker()