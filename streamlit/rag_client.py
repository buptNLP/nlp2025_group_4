"""
RAG 系统客户端库

使用示例:
    from rag_client import RAGClient
    
    client = RAGClient("http://localhost:8000")
    
    # 直接搜索
    results = client.direct_search("什么是仁", bm25_weight=0.3, vector_weight=0.7)
    
    # 智能搜索
    results = client.smart_search("如何修身养性")
"""

import requests
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SearchResult:
    """搜索结果数据类"""
    content: str
    book: str
    chapter: str
    topic: str
    segment_id: str
    prev_context: Optional[str] = None
    next_context: Optional[str] = None
    scores: Dict[str, float] = None
    metadata: Dict[str, Any] = None
    
    def __str__(self):
        return f"《{self.book}·{self.chapter}》: {self.content[:50]}..."
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """从字典创建实例"""
        return cls(**data)


@dataclass
class SearchResponse:
    """搜索响应数据类"""
    success: bool
    query: str
    results: List[SearchResult]
    total_results: int
    search_mode: str
    timestamp: str
    optimized_query: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    analysis_info: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResponse':
        """从字典创建实例"""
        # 转换结果列表
        results = [SearchResult.from_dict(r) for r in data.get('results', [])]
        data['results'] = results
        return cls(**data)


class RAGClient:
    """RAG 系统客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        初始化客户端
        
        Args:
            base_url: API 基础 URL
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        """关闭会话"""
        self.session.close()
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息
        
        Returns:
            系统信息字典
        """
        response = self._get("/")
        return response
    
    def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            服务是否健康
        """
        try:
            response = self._get("/health")
            return response.get('status') == 'healthy'
        except:
            return False
    
    def direct_search(
        self,
        query: str,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, str]] = None
    ) -> SearchResponse:
        """
        直接搜索
        
        Args:
            query: 搜索查询
            bm25_weight: BM25 权重 (0-1)
            vector_weight: 向量权重 (0-1)
            top_k: 返回结果数量
            metadata_filter: 元数据过滤条件
            
        Returns:
            搜索响应
        """
        # 归一化权重
        total = bm25_weight + vector_weight
        if total > 0:
            bm25_weight = bm25_weight / total
            vector_weight = vector_weight / total
        
        request_data = {
            "query": query,
            "bm25_weight": bm25_weight,
            "vector_weight": vector_weight,
            "top_k": top_k
        }
        
        if metadata_filter:
            request_data["metadata_filter"] = metadata_filter
        
        response = self._post("/search/direct", request_data)
        return SearchResponse.from_dict(response)
    
    def smart_search(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, str]] = None
    ) -> SearchResponse:
        """
        智能搜索（使用 AI 优化）
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            metadata_filter: 元数据过滤条件
            
        Returns:
            搜索响应
        """
        request_data = {
            "query": query,
            "top_k": top_k
        }
        
        if metadata_filter:
            request_data["metadata_filter"] = metadata_filter
        
        response = self._post("/search/smart", request_data)
        return SearchResponse.from_dict(response)
    
    def search(
        self,
        query: str,
        mode: str = "smart",
        **kwargs
    ) -> SearchResponse:
        """
        通用搜索接口
        
        Args:
            query: 搜索查询
            mode: 搜索模式 ("smart" 或 "direct")
            **kwargs: 其他参数
            
        Returns:
            搜索响应
        """
        if mode == "smart":
            return self.smart_search(query, **kwargs)
        else:
            return self.direct_search(query, **kwargs)
    
    def batch_search(
        self,
        queries: List[str],
        mode: str = "smart",
        **kwargs
    ) -> List[SearchResponse]:
        """
        批量搜索
        
        Args:
            queries: 查询列表
            mode: 搜索模式
            **kwargs: 其他参数
            
        Returns:
            搜索响应列表
        """
        results = []
        for query in queries:
            try:
                response = self.search(query, mode=mode, **kwargs)
                results.append(response)
            except Exception as e:
                # 创建错误响应
                error_response = SearchResponse(
                    success=False,
                    query=query,
                    results=[],
                    total_results=0,
                    search_mode=mode,
                    timestamp=datetime.now().isoformat(),
                    analysis_info={"error": str(e)}
                )
                results.append(error_response)
        return results
    
    def _get(self, endpoint: str) -> Dict[str, Any]:
        """发送 GET 请求"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RAGClientError(f"GET 请求失败: {e}")
    
    def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """发送 POST 请求"""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if hasattr(e.response, 'text'):
                raise RAGClientError(f"POST 请求失败: {e}\n详情: {e.response.text}")
            raise RAGClientError(f"POST 请求失败: {e}")


class RAGClientError(Exception):
    """RAG 客户端错误"""
    pass


# 便捷函数
def create_client(base_url: str = "http://localhost:8000") -> RAGClient:
    """创建 RAG 客户端实例"""
    return RAGClient(base_url)


def quick_search(
    query: str,
    base_url: str = "http://localhost:8000",
    mode: str = "smart",
    top_k: int = 5
) -> List[str]:
    """
    快速搜索并返回内容列表
    
    Args:
        query: 搜索查询
        base_url: API 地址
        mode: 搜索模式
        top_k: 结果数量
        
    Returns:
        内容列表
    """
    with create_client(base_url) as client:
        response = client.search(query, mode=mode, top_k=top_k)
        return [r.content for r in response.results]


# 使用示例
if __name__ == "__main__":
    # 创建客户端
    client = RAGClient()
    
    # 检查服务健康状态
    if not client.health_check():
        print("服务未启动或不可用")
        exit(1)
    
    # 获取系统信息
    info = client.get_system_info()
    print(f"系统状态: {info['status']}")
    print(f"文本片段: {info['total_segments']}")
    
    # 直接搜索示例
    print("\n=== 直接搜索 ===")
    response = client.direct_search(
        "什么是仁",
        bm25_weight=0.4,
        vector_weight=0.6,
        top_k=10
    )
    
    print(f"查询: {response.query}")
    print(f"找到 {response.total_results} 个结果:")
    for i, result in enumerate(response.results, 1):
        print(f"{i}. {result.content}")
        print(f"{i}. {result.prev_context}")
        print(f"{i}. {result.next_context}")
        if result.scores:
            print(f"   评分: {result.scores}")
    
    # 智能搜索示例
    print("\n=== 智能搜索 ===")
    try:
        response = client.smart_search("如何成为君子", top_k=3)
        print(f"原始查询: {response.query}")
        if response.optimized_query:
            print(f"优化查询: {response.optimized_query}")
        print(f"找到 {response.total_results} 个结果:")
        for i, result in enumerate(response.results, 1):
            print(f"{i}. {result}")
            print(f"{i}. {result.prev_context}")
            print(f"{i}. {result.next_context}")
    except RAGClientError as e:
        print(f"智能搜索失败（可能未配置 API Key）: {e}")
    
    # 快速搜索示例
    print("\n=== 快速搜索 ===")
    contents = quick_search("学而时习之", mode="direct", top_k=2)
    for content in contents:
        print(f"- {content}")
    
    # 关闭客户端
    client.close()