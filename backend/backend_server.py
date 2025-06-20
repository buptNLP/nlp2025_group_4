import os
import sys
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from contextlib import asynccontextmanager
import logging
from datetime import datetime

# 导入您的 RAG 系统模块
from backend_rag_system import UniversalAncientRAG
from api_client import APIProvider
from config import HAS_BM25, HAS_RERANKER

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局 RAG 系统实例
rag_system = None

# 配置类
class RAGConfig:
    """RAG 系统配置"""
    DATA_DIR = os.getenv("RAG_DATA_DIR", "../data")  # 数据目录
    FILE_EXTENSIONS = ['.txt', '.md', '.text']  # 支持的文件格式
    EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "../finetuned_bge_ancient")
    MAX_CHUNK_SIZE = int(os.getenv("RAG_MAX_CHUNK_SIZE", "150"))
    MIN_CHUNK_SIZE = int(os.getenv("RAG_MIN_CHUNK_SIZE", "20"))
    CONTEXT_WINDOW = int(os.getenv("RAG_CONTEXT_WINDOW", "80"))
    
    # API 配置
    API_PROVIDER = os.getenv("RAG_API_PROVIDER", "glm")  # 默认使用 GLM
    API_KEY = os.getenv("RAG_API_KEY", "bccd263f4dca4e2c9670fb6a805f957c.nzHa43Y6Qt40U8x4")  # 从环境变量读取 API Key
    print(f"API_KEY:{API_KEY}")
    
    # 重排序配置
    USE_RERANKER = os.getenv("RAG_USE_RERANKER", "true").lower() == "true"
    RERANKER_MODEL = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-large")

# 请求和响应模型
class DirectSearchRequest(BaseModel):
    """直接搜索请求模型"""
    query: str = Field(..., description="搜索查询")
    bm25_weight: float = Field(0.3, ge=0, le=1, description="BM25权重")
    vector_weight: float = Field(0.7, ge=0, le=1, description="向量权重")
    top_k: int = Field(10, ge=1, le=50, description="返回结果数量")
    metadata_filter: Optional[Dict[str, str]] = Field(None, description="元数据过滤条件")

class SmartSearchRequest(BaseModel):
    """智能搜索请求模型"""
    query: str = Field(..., description="搜索查询")
    top_k: int = Field(10, ge=1, le=50, description="返回结果数量")
    metadata_filter: Optional[Dict[str, str]] = Field(None, description="元数据过滤条件")

class SearchResult(BaseModel):
    """搜索结果模型"""
    content: str = Field(..., description="文本内容")
    book: str = Field(..., description="书名")
    chapter: str = Field(..., description="章节")
    topic: str = Field(..., description="话题分类")
    segment_id: str = Field(..., description="片段ID")
    prev_context: Optional[str] = Field(None, description="前文上下文")
    next_context: Optional[str] = Field(None, description="后文上下文")
    scores: Dict[str, float] = Field(..., description="各种评分")
    metadata: Dict[str, Any] = Field(..., description="其他元数据")

class SearchResponse(BaseModel):
    """搜索响应模型"""
    success: bool = Field(..., description="是否成功")
    query: str = Field(..., description="原始查询")
    optimized_query: Optional[str] = Field(None, description="优化后的查询")
    results: List[SearchResult] = Field(..., description="搜索结果列表")
    total_results: int = Field(..., description="结果总数")
    search_mode: str = Field(..., description="搜索模式")
    weights: Optional[Dict[str, float]] = Field(None, description="使用的权重")
    analysis_info: Optional[Dict[str, Any]] = Field(None, description="分析信息")
    timestamp: str = Field(..., description="响应时间戳")

class SystemInfoResponse(BaseModel):
    """系统信息响应模型"""
    status: str = Field(..., description="系统状态")
    total_segments: int = Field(..., description="文本片段总数")
    total_books: int = Field(..., description="书籍总数")
    total_chapters: int = Field(..., description="章节总数")
    embedding_model: str = Field(..., description="使用的 Embedding 模型")
    has_bm25: bool = Field(..., description="是否支持 BM25")
    has_reranker: bool = Field(..., description="是否支持重排序")
    api_configured: bool = Field(..., description="API 是否配置")
    data_loaded_at: Optional[str] = Field(None, description="数据加载时间")

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global rag_system
    
    # 启动时初始化
    logger.info("正在初始化 RAG 系统...")
    
    try:
        # 创建 RAG 系统实例
        rag_system = UniversalAncientRAG(
            embedding_model=RAGConfig.EMBEDDING_MODEL,
            max_chunk_size=RAGConfig.MAX_CHUNK_SIZE,
            min_chunk_size=RAGConfig.MIN_CHUNK_SIZE,
            context_window=RAGConfig.CONTEXT_WINDOW
        )
        
        # 配置 API（如果有）
        if RAGConfig.API_KEY:
            api_config = {
                'provider': RAGConfig.API_PROVIDER,
                'api_key': RAGConfig.API_KEY
            }
            rag_system.set_api_config(api_config)
            logger.info(f"已配置 {RAGConfig.API_PROVIDER.upper()} API")
        else:
            logger.warning("未配置 API Key，智能搜索功能将不可用")
        
        # 加载数据
        logger.info(f"正在从 {RAGConfig.DATA_DIR} 加载数据...")
        count = rag_system.load_from_directory(
            RAGConfig.DATA_DIR,
            file_extensions=RAGConfig.FILE_EXTENSIONS
        )
        
        if count > 0:
            logger.info(f"成功加载 {count} 个文本片段")
            # 记录加载时间
            app.state.data_loaded_at = datetime.now().isoformat()
        else:
            logger.error("未能加载任何数据")
            raise RuntimeError("数据加载失败")
            
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        raise
    
    yield  # 应用运行期间
    
    # 关闭时清理
    logger.info("正在关闭 RAG 系统...")

# 创建 FastAPI 应用
app = FastAPI(
    title="通用古文 RAG 系统 API",
    description="基于混合检索技术的智能古文问答系统后端服务",
    version="1.0.0",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 端点
@app.get("/", response_model=SystemInfoResponse)
async def get_system_info():
    """获取系统信息"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="系统未初始化")
    
    return SystemInfoResponse(
        status="running",
        total_segments=len(rag_system.segments) if rag_system.segments else 0,
        total_books=len(set(seg.book for seg in rag_system.segments)) if rag_system.segments else 0,
        total_chapters=len(set(seg.chapter for seg in rag_system.segments)) if rag_system.segments else 0,
        embedding_model=rag_system.embedding_model,
        has_bm25=HAS_BM25,
        has_reranker=HAS_RERANKER and RAGConfig.USE_RERANKER,
        api_configured=bool(rag_system.api_client and rag_system.api_client.is_available()),
        data_loaded_at=getattr(app.state, 'data_loaded_at', None)
    )

@app.post("/search/direct", response_model=SearchResponse)
async def direct_search(request: DirectSearchRequest):
    """直接搜索接口 - 指定权重进行搜索"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="系统未初始化")
    
    try:
        # 确保权重和为1
        total_weight = request.bm25_weight + request.vector_weight
        if abs(total_weight - 1.0) > 0.01:
            # 自动归一化
            request.bm25_weight = request.bm25_weight / total_weight
            request.vector_weight = request.vector_weight / total_weight
        
        # 执行搜索
        results = rag_system.search(
            query=request.query,
            top_k=request.top_k,
            search_mode='hybrid',
            metadata_filter=request.metadata_filter,
            bm25_weight=request.bm25_weight,
            vector_weight=request.vector_weight
        )
        
        # 转换结果格式
        search_results = []
        for result in results:
            meta = result['metadata']
            
            nested_meta = meta.get('metadata', {})
            # 提取上下文信息
            prev_context = nested_meta.get('prev_context', '')
            next_context = nested_meta.get('next_context', '')
            
            # 收集所有分数
            scores = {}
            if 'bm25_score' in result:
                scores['bm25_score'] = result['bm25_score']
            if 'vector_score' in result:
                scores['vector_score'] = result['vector_score']
            if 'combined_score' in result:
                scores['combined_score'] = result['combined_score']
            if 'rerank_score' in result:
                scores['rerank_score'] = result['rerank_score']
            
            search_results.append(SearchResult(
                content=result['content'],
                book=meta['book'],
                chapter=meta['chapter'],
                topic=meta['topic'],
                segment_id=meta['segment_id'],
                prev_context=prev_context if prev_context else None,
                next_context=next_context if next_context else None,
                scores=scores,
                metadata=meta
            ))
        
        return SearchResponse(
            success=True,
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_mode='hybrid',
            weights={
                'bm25_weight': request.bm25_weight,
                'vector_weight': request.vector_weight
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"直接搜索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/smart", response_model=SearchResponse)
async def smart_search(request: SmartSearchRequest):
    """智能搜索接口 - 使用 AI 优化查询"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="系统未初始化")
    
    if not rag_system.api_client or not rag_system.api_client.is_available():
        raise HTTPException(
            status_code=400, 
            detail="智能搜索需要配置 API Key。请使用直接搜索接口或配置 RAG_API_KEY 环境变量。"
        )
    
    try:
        # 使用智能分析和搜索
        analysis_result, results = rag_system.analyze_and_search(
            query=request.query,
            top_k=request.top_k,
            search_mode='hybrid',
            metadata_filter=request.metadata_filter
        )
        
        # 如果不需要搜索
        if not analysis_result['need_search']:
            return SearchResponse(
                success=True,
                query=request.query,
                results=[],
                total_results=0,
                search_mode='no_search',
                analysis_info={
                    'need_search': False,
                    'reason': analysis_result['reason'],
                    'direct_answer': analysis_result.get('direct_answer')
                },
                timestamp=datetime.now().isoformat()
            )
        
        # 转换结果格式
        search_results = []
        for result in results:
            meta = result['metadata']
            
            # 提取上下文信息
            nested_meta = meta.get('metadata', {})
            prev_context = nested_meta.get('prev_context', '')
            next_context = nested_meta.get('next_context', '')
            
            # 收集所有分数
            scores = {}
            if 'bm25_score' in result:
                scores['bm25_score'] = result['bm25_score']
            if 'vector_score' in result:
                scores['vector_score'] = result['vector_score']
            if 'combined_score' in result:
                scores['combined_score'] = result['combined_score']
            if 'rerank_score' in result:
                scores['rerank_score'] = result['rerank_score']
            
            search_results.append(SearchResult(
                content=result['content'],
                book=meta['book'],
                chapter=meta['chapter'],
                topic=meta['topic'],
                segment_id=meta['segment_id'],
                prev_context=prev_context if prev_context else None,
                next_context=next_context if next_context else None,
                scores=scores,
                metadata=meta
            ))
        
        return SearchResponse(
            success=True,
            query=request.query,
            optimized_query=analysis_result['optimized_query'],
            results=search_results,
            total_results=len(search_results),
            search_mode='hybrid',
            weights={
                'bm25_weight': analysis_result['bm25_weight'],
                'vector_weight': analysis_result['vector_weight']
            },
            analysis_info={
                'need_search': True,
                'reason': analysis_result['reason'],
                'optimization_applied': analysis_result['optimized_query'] != request.query
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"智能搜索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# 主函数
def main():
    """启动服务器"""
    # 可以通过环境变量配置
    host = os.getenv("RAG_HOST", "0.0.0.0")
    port = int(os.getenv("RAG_PORT", "8000"))
    
    logger.info(f"启动 RAG 后端服务: http://{host}:{port}")
    logger.info(f"数据目录: {RAGConfig.DATA_DIR}")
    logger.info(f"API 提供商: {RAGConfig.API_PROVIDER}")
    
    uvicorn.run(
        "backend_server:app",
        host=host,
        port=port,
        reload=False,  # 生产环境应该设为 False
        log_level="info"
    )

if __name__ == "__main__":
    main()