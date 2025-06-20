from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class AncientTextSegment:
    """古文片段数据结构"""
    book: str          # 书名
    chapter: str       # 篇章
    content: str       # 内容
    topic: str         # 话题
    segment_id: str    # 片段ID
    context: str       # 上下文
    metadata: Dict[str, Any]  # 扩展元数据