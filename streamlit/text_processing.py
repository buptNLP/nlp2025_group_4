import re
import jieba
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class TextChunk:
    """文本块数据结构"""
    content: str                    # 块内容
    paragraph_index: int           # 原始段落索引
    sub_index: int                 # 子块索引（如果是分割的段落）
    is_continuation: bool          # 是否是长段落的延续
    prev_context: str              # 前文上下文
    next_context: str              # 后文上下文
    metadata: Dict[str, any]       # 其他元数据


class SmartTextChunker:
    """智能文本分块器 - 层级分块版"""
    
    def __init__(self, 
                 max_chunk_size: int = 250,
                 min_chunk_size: int = 20,
                 context_window: int = 80):
        """
        初始化分块器
        
        Args:
            max_chunk_size: 最大块大小
            min_chunk_size: 最小块大小（过滤噪音）
            context_window: 上下文窗口大小
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.context_window = context_window
        
        # 句子分割标点
        self.sentence_endings = ['。', '！', '？']
        self.secondary_endings = ['；', '：']  # 次要分割点
        
    def chunk_text(self, text: str) -> List[TextChunk]:
        """
        主分块方法
        
        Args:
            text: 输入文本
            
        Returns:
            TextChunk列表
        """
        # 第一级：按行分割
        paragraphs = text.strip().split('\n')
        
        chunks = []
        
        for para_idx, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            
            # 过滤空行和过短行
            if not paragraph or len(paragraph) < self.min_chunk_size:
                continue
            
            # 判断是否需要进一步分割
            if len(paragraph) <= self.max_chunk_size:
                # 不需要分割，直接作为一个块
                chunk = TextChunk(
                    content=paragraph,
                    paragraph_index=para_idx,
                    sub_index=0,
                    is_continuation=False,
                    prev_context="",  # 稍后填充
                    next_context="",  # 稍后填充
                    metadata={"original_length": len(paragraph)}
                )
                chunks.append(chunk)
            else:
                # 需要分割长段落
                sub_chunks = self._split_long_paragraph(paragraph, para_idx)
                chunks.extend(sub_chunks)
        
        # 填充上下文信息
        self._add_context(chunks, paragraphs)
        
        return chunks
    
    def _split_long_paragraph(self, paragraph: str, para_idx: int) -> List[TextChunk]:
        """
        分割长段落
        
        Args:
            paragraph: 段落内容
            para_idx: 段落索引
            
        Returns:
            子块列表
        """
        sub_chunks = []
        
        # 先尝试按主要标点分句
        sentences = self._split_sentences(paragraph)
        
        current_chunk = ""
        sub_idx = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 判断是否需要新建块
            if current_chunk and len(current_chunk) + len(sentence) > self.max_chunk_size:
                # 保存当前块
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    paragraph_index=para_idx,
                    sub_index=sub_idx,
                    is_continuation=(sub_idx > 0),
                    prev_context="",
                    next_context="",
                    metadata={
                        "original_length": len(paragraph),
                        "split_reason": "length_exceeded"
                    }
                )
                sub_chunks.append(chunk)
                
                # 开始新块
                current_chunk = sentence
                sub_idx += 1
            else:
                # 累积到当前块
                current_chunk += sentence
        
        # 保存最后一个块
        if current_chunk.strip():
            chunk = TextChunk(
                content=current_chunk.strip(),
                paragraph_index=para_idx,
                sub_index=sub_idx,
                is_continuation=(sub_idx > 0),
                prev_context="",
                next_context="",
                metadata={
                    "original_length": len(paragraph),
                    "split_reason": "final_chunk"
                }
            )
            sub_chunks.append(chunk)
        
        return sub_chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        智能分句
        
        Args:
            text: 待分句文本
            
        Returns:
            句子列表
        """
        # 构建分句正则
        primary_pattern = f"([{''.join(self.sentence_endings)}])"
        secondary_pattern = f"([{''.join(self.secondary_endings)}])"
        
        # 先按主要标点分句
        parts = re.split(primary_pattern, text)
        sentences = []
        current_sentence = ""
        
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i + 1] in self.sentence_endings:
                # 当前部分 + 标点
                current_sentence += parts[i] + parts[i + 1]
                
                # 检查是否是引号内的句子（避免分割对话）
                if not self._is_in_quotes(current_sentence, text):
                    sentences.append(current_sentence)
                    current_sentence = ""
                
                i += 2
            else:
                current_sentence += parts[i]
                i += 1
        
        # 处理剩余部分
        if current_sentence.strip():
            # 如果剩余部分太长，尝试用次要标点分割
            if len(current_sentence) > self.max_chunk_size:
                sub_parts = re.split(secondary_pattern, current_sentence)
                temp_sentence = ""
                for j, part in enumerate(sub_parts):
                    temp_sentence += part
                    if len(temp_sentence) > self.max_chunk_size * 0.8:  # 80%阈值
                        sentences.append(temp_sentence.strip())
                        temp_sentence = ""
                if temp_sentence.strip():
                    sentences.append(temp_sentence.strip())
            else:
                sentences.append(current_sentence.strip())
        
        return [s for s in sentences if s.strip()]
    
    def _is_in_quotes(self, sentence: str, full_text: str) -> bool:
        """
        检查句子是否在引号内
        
        Args:
            sentence: 句子
            full_text: 完整文本
            
        Returns:
            是否在引号内
        """
        # 简单检查：计算句子前的引号数量
        pos = full_text.find(sentence)
        if pos == -1:
            return False
        
        before_text = full_text[:pos]
        quote_count = before_text.count('"') + before_text.count('"')
        
        # 如果引号数是奇数，说明在引号内
        return quote_count % 2 == 1
    
    def _add_context(self, chunks: List[TextChunk], original_paragraphs: List[str]):
        """
        为每个块添加上下文信息
        
        Args:
            chunks: 块列表
            original_paragraphs: 原始段落列表
        """
        for i, chunk in enumerate(chunks):
            # 前文上下文
            if i > 0:
                prev_chunk = chunks[i - 1]
                # 取前一个块的末尾部分作为上下文
                chunk.prev_context = self._get_context_snippet(
                    prev_chunk.content, 
                    self.context_window, 
                    from_end=True
                )
            else:
                # 第一个块，尝试从前一个段落获取上下文
                if chunk.paragraph_index > 0:
                    for j in range(chunk.paragraph_index - 1, -1, -1):
                        if j < len(original_paragraphs) and original_paragraphs[j].strip():
                            chunk.prev_context = self._get_context_snippet(
                                original_paragraphs[j], 
                                self.context_window, 
                                from_end=True
                            )
                            break
            
            # 后文上下文
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]
                # 取后一个块的开头部分作为上下文
                chunk.next_context = self._get_context_snippet(
                    next_chunk.content, 
                    self.context_window, 
                    from_end=False
                )
            else:
                # 最后一个块，尝试从后一个段落获取上下文
                if chunk.paragraph_index < len(original_paragraphs) - 1:
                    for j in range(chunk.paragraph_index + 1, len(original_paragraphs)):
                        if j < len(original_paragraphs) and original_paragraphs[j].strip():
                            chunk.next_context = self._get_context_snippet(
                                original_paragraphs[j], 
                                self.context_window, 
                                from_end=False
                            )
                            break
    
    def _get_context_snippet(self, text: str, window_size: int, from_end: bool = False) -> str:
        """
        获取上下文片段
        
        Args:
            text: 源文本
            window_size: 窗口大小
            from_end: 是否从末尾截取
            
        Returns:
            上下文片段
        """
        if len(text) <= window_size:
            return text
        
        if from_end:
            snippet = text[-window_size:]
            # 尝试在标点处截断，使上下文更完整
            for punct in self.sentence_endings + self.secondary_endings + ['，']:
                last_punct = snippet.rfind(punct)
                if last_punct > window_size * 0.5:  # 至少保留一半长度
                    return snippet[last_punct + 1:].strip()
            return "..." + snippet.strip()
        else:
            snippet = text[:window_size]
            # 尝试在标点处截断
            for punct in self.sentence_endings + self.secondary_endings + ['，']:
                first_punct = snippet.find(punct)
                if first_punct > window_size * 0.5:
                    return snippet[:first_punct + 1].strip()
            return snippet.strip() + "..."


class AncientTextAnalyzer:
    """古文分析器 - 通用化设计"""
    
    def __init__(self):
        self.topic_keywords = self._load_topic_keywords()
    
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
        """提取上下文 - 保持向后兼容"""
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