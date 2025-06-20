import requests
import streamlit as st
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
from enum import Enum
import re

class APIProvider(Enum):
    """支持的 API 提供商"""
    DEEPSEEK = "deepseek"
    GLM = "glm"

class BaseAPIClient(ABC):
    """基础 API 客户端抽象类"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key.strip() if api_key else None
    
    def is_available(self) -> bool:
        """检查 API 是否可用"""
        return bool(self.api_key and self.api_key.strip())
    
    @abstractmethod
    def call_chat_completion(
        self, 
        prompt: str, 
        system_prompt: str = None,
        model: str = None,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        timeout: int = 30
    ) -> str:
        """聊天完成 API 调用的抽象方法"""
        pass
    
    def analyze_query(self, raw_query: str) -> Dict[str, Any]:
        """
        综合分析查询 - 一次API调用完成所有任务
        返回结构化数据：
        {
            'need_search': bool,
            'reason': str,
            'optimized_query': str,
            'bm25_weight': float,
            'vector_weight': float,
            'direct_answer': str,
            'extracted_book': str | None  # --- 新增 ---
        }
        """
        if not self.is_available():
            return {
                'need_search': True,
                'reason': '未配置API',
                'optimized_query': raw_query,
                'bm25_weight': 0.3,
                'vector_weight': 0.7,
                'direct_answer': None,
                'extracted_book': None, # --- 新增 ---
            }
       
        # --- 修改提示词 ---
        # 优化后的提示词
        prompt = f"""你是古典文献检索专家，擅长分析用户查询意图并提供最佳检索策略。

    【用户问题】
    {raw_query}

    【分析任务】
    请按以下步骤进行结构化分析：

    ## 第一步：检索必要性判断
    判断标准：
    ✅ 需要检索的情况：
    - 涉及古典文献内容（论语、庄子、史记、三国演义等）
    - 查询古代人物、思想、典故、制度
    - 需要引用原文或考证的问题
    - 古代历史事件或文化现象

    ❌ 无需检索的情况：
    - 纯现代概念或技术问题
    - 简单的常识性问题
    - 与古典文献无关的内容

    ## 第二步：查询优化（仅当需要检索时）
    遵循四大原则：

    ### 2.1 词汇古典化
    - 现代词汇 → 古代对应词汇
    - 示例：
    * "皇帝" → "天子"/"君主"/"帝王" 
    * "官员" → "臣工"/"仕人"/"朝臣"
    * "管理" → "治理"/"统御"/"经纬"
    * "教育" → "教化"/"育才"/"启蒙"

    ### 2.2 句式文言化
    - 采用古典句式结构

    ### 2.3 保留关键信息
    - 重要人名、地名、书名保持不变
    - 增加核心的古典词描述，拓展一些同义词
    - 核心概念不能偏离原意
    - 优化查询时保持原意不变，只改变表达方式
    
    - 示例：
    * "孔子认为什么是仁？" → "孔子论仁之义何在？克己复礼之说何解？"
    * "如何治理国家？" → "治国之道何如？德政王道之术何择？"
    * "好官员的标准？" → "贤臣之资何以辨之？清廉循吏之道何在？"
    
    ### 2.4 完整保留古文原句
    - 如果query是古文原句，完整保留，不需要做优化

    ## 第三步：检索权重策略
    根据查询类型选择最佳权重组合：

    ### 🎯 高BM25权重 (0.6-0.8)
    **适用场景**：
    - 精确古文匹配："学而时习之"
    - 专有名词查询："诸葛亮"、"赤壁之战"  
    - 术语定义："什么是仁"
    **推荐配置**：BM25: 0.7, Vector: 0.3

    ### 🧠 高Vector权重 (0.6-0.8)  
    **适用场景**：
    - 概念理解："儒家思想的核心"
    - 主题查询："古代教育方法"
    - 哲学探讨："庄子的人生观"
    **推荐配置**：BM25: 0.3, Vector: 0.7

    ### ⚖️ 平衡权重 (0.4-0.6)
    **适用场景**：
    - 复合查询："孔子关于仁的具体论述"
    - 比较分析："孔子和老子思想差异"  
    - 不确定意图的查询
    **推荐配置**：BM25: 0.5, Vector: 0.5

    ## 第四步：书名提取
    精确识别问题中明确提及的古典书名：
    - 涉及到多本书，返回null
    - 完整书名：《论语》、《庄子》、《史记》、《三国演义》等
    - 简称：论语、庄子、史记、三国等
    - 如未明确提及任何书名，返回null

    【输出要求】
    请严格按照以下JSON格式输出，只输出JSON，不要包含任何解释性文字，确保所有字段都有明确的值：

    ```json
    {{
        "need_search": true,
        "reason": "具体的判断理由说明",
        "optimized_query": "按上述规则优化后的查询语句",
        "bm25_weight": 具体的权重或者None,
        "vector_weight": 具体的权重或者None,
        "direct_answer": None,
        "extracted_book": "明确的单本书名"或者None(表示正常检索)
    }}
    ```    

    开始分析："""
        try:
            result = self.call_chat_completion(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3,
                timeout=20
            )
            
            print(result)
            
            import json
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # 修复：安全处理可能为 null 的权重值
                bm25_weight = parsed.get('bm25_weight')
                vector_weight = parsed.get('vector_weight')
                
                return {
                    'need_search': bool(parsed.get('need_search', True)),
                    'reason': parsed.get('reason', ''),
                    'optimized_query': parsed.get('optimized_query', raw_query),
                    'bm25_weight': float(bm25_weight) if bm25_weight is not None else 0.3,
                    'vector_weight': float(vector_weight) if vector_weight is not None else 0.7,
                    'direct_answer': parsed.get('direct_answer', None),
                    'extracted_book': parsed.get('extracted_book')
                }
            else:
                raise ValueError("无法解析JSON响应")
                
        except Exception as e:
            return {
                'need_search': True,
                'reason': f'API分析失败: {str(e)}',
                'optimized_query': raw_query,
                'bm25_weight': 0.3,
                'vector_weight': 0.7,
                'direct_answer': None,
                'extracted_book': None
            }
            
    
    # 在 BaseAPIClient 类中添加以下方法
    def decompose_complex_query(self, raw_query: str) -> Dict[str, Any]:
        """
        判断是否需要多轮检索并拆解复杂任务
        返回结构：
        {
            'need_multi_round': bool,
            'reason': str,
            'subtasks': [
                {
                    'subtask_id': int,
                    'subtask_query': str,
                    'subtask_focus': str,
                    'search_weight': {'bm25': float, 'vector': float}
                }
            ],
            'synthesis_instruction': str
        }
        """
        if not self.is_available():
            return {
                'need_multi_round': False,
                'reason': '未配置API',
                'subtasks': [],
                'synthesis_instruction': ''
            }
        
        prompt = f"""你是一个古文检索专家。请分析用户的问题是否需要多轮检索。

    用户问题：{raw_query}

    分析步骤：
    1. 判断是否需要多轮检索：
    - 简单的概念查询（如"什么是仁"）不需要多轮检索
    - 涉及多个方面的复杂问题需要多轮检索
    - 需要对比、分析、综合的问题需要多轮检索
    - 涉及多个人物、时期、观点的问题需要多轮检索

    2. 如果需要多轮检索，将问题拆解成2-5个子任务：
    - 每个子任务应该聚焦一个具体方面
    - 子任务之间应该互补，覆盖原问题的不同维度
    - 需要对子任务的query做一个古文检索的优化，采用古文的风格并保留核心关键词
    
    3. 如果需要检索，分析检索权重：
        3.1 高BM25权重场景（0.6-0.8）
    - **精确古文匹配**：用户提供完整古文句子
    - **专有名词查询**：特定人名、地名、官职名
    - **典籍书名查询**：明确指向特定古籍
    - **术语定义查询**：查询特定古典概念
    推荐权重：BM25: 0.7, Vector: 0.3

        3.2 高向量权重场景（0.6-0.8）
    - **概念性理解**：查询思想、哲学观点
    - **主题性查询**：关于某个话题的广泛内容
    - **情境描述**：用现代语言描述古代情境
    - **比较分析**：不同观点、人物的对比
    推荐权重：BM25: 0.3, Vector: 0.7

        3.3 平衡权重场景（0.4-0.6）
    - **混合查询**：既有精确词汇又有概念理解
    - **不确定查询**：无法明确判断用户意图
    - **多层次查询**：涉及多个维度的复杂问题
    推荐权重：BM25: 0.5, Vector: 0.5

    4. 提供综合指导：如何将各子任务的结果综合成完整答案

    示例拆解：
    - "孔子和孟子的政治思想有何异同？" → 
    子任务1: "孔子的政治思想核心观点"
    子任务2: "孟子的政治思想核心观点"
    子任务3: "儒家政治思想的演变"

    请严格按照以下JSON格式返回结果：
    {{
        "need_multi_round": true/false,
        "reason": "判断理由",
        "subtasks": [
            {{
                "subtask_id": 1,
                "subtask_query": "优化后的子查询",
                "subtask_focus": "该子任务的核心关注点",
                "search_weight": {{"bm25": 0.3, "vector": 0.7}}
            }}
        ],
        "synthesis_instruction": "如何综合各子任务结果的指导"
    }}"""

        try:
            result = self.call_chat_completion(
                prompt=prompt,
                max_tokens=800,
                temperature=0.3,
                timeout=25
            )
            
            print(result)
            
            import json
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                
                # 安全处理返回的数据
                need_multi_round = bool(parsed.get('need_multi_round', False))
                
                # 处理子任务列表
                subtasks = []
                if need_multi_round and 'subtasks' in parsed and parsed['subtasks']:
                    for subtask in parsed['subtasks']:
                        # 确保每个子任务有完整的结构
                        processed_subtask = {
                            'subtask_id': subtask.get('subtask_id', 1),
                            'subtask_query': subtask.get('subtask_query', ''),
                            'subtask_focus': subtask.get('subtask_focus', ''),
                            'search_weight': self._process_search_weight(
                                subtask.get('search_weight', {})
                            )
                        }
                        subtasks.append(processed_subtask)
                
                return {
                    'need_multi_round': need_multi_round,
                    'reason': parsed.get('reason', ''),
                    'subtasks': subtasks,
                    'synthesis_instruction': parsed.get('synthesis_instruction', '')
                }
            else:
                raise ValueError("无法解析JSON响应")
                
        except Exception as e:
            return {
                'need_multi_round': False,
                'reason': f'任务拆解失败: {str(e)}',
                'subtasks': [],
                'synthesis_instruction': ''
            }
            
    def _process_search_weight(self, weight_dict: Dict) -> Dict[str, float]:
        """
        安全处理搜索权重
        """
        if not weight_dict:
            return {'bm25': 0.3, 'vector': 0.7}
        
        bm25_weight = weight_dict.get('bm25')
        vector_weight = weight_dict.get('vector')
        
        # 处理 null 或无效值
        bm25 = float(bm25_weight) if bm25_weight is not None else 0.3
        vector = float(vector_weight) if vector_weight is not None else 0.7
        
        # 确保权重在有效范围内
        bm25 = max(0.0, min(1.0, bm25))
        vector = max(0.0, min(1.0, vector))
        
        # 归一化权重
        total = bm25 + vector
        if total > 0:
            bm25 = bm25 / total
            vector = vector / total
        else:
            bm25, vector = 0.3, 0.7
        
        return {'bm25': bm25, 'vector': vector}

    def synthesize_multi_round_results(self, query: str, subtasks_results: List[Dict]) -> str:
        """
        综合多轮检索的结果
        subtasks_results: 每个子任务的检索结果列表
        """
        if not self.is_available():
            return self._basic_synthesis(query, subtasks_results)
        
        # 构建上下文
        context_parts = []
        for subtask_result in subtasks_results:
            subtask_query = subtask_result['subtask_query']
            subtask_focus = subtask_result['subtask_focus']
            results = subtask_result['results']
            
            context_parts.append(f"\n### 子任务：{subtask_query}")
            context_parts.append(f"关注点：{subtask_focus}")
            
            for i, item in enumerate(results[:2], 1):  # 每个子任务取前2个结果
                meta = item['metadata']
                context_parts.append(
                    f"{i}. 《{meta['book']}·{meta['chapter']}》: {item['content']}"
                )
        
        context_text = "\n".join(context_parts)
        
        prompt = f"""你是一位精通中国古典文学的学者。用户提出了一个复杂问题，我们通过多轮检索获得了相关古文。

    用户原始问题：{query}

    多轮检索结果：
    {context_text}

    综合指导：{subtasks_results[0].get('synthesis_instruction', '请综合所有相关内容，给出全面的回答')}

    请基于以上多轮检索的结果，综合回答用户的问题。要求：
    1. 整合各子任务的发现，形成连贯的回答
    2. 突出重点，条理清晰
    3. 引用具体原文支撑观点
    4. 提供深入的分析和见解
    5. 如果涉及对比，要明确指出异同

    请给出综合、深入、有见地的回答："""

        return self.call_chat_completion(
            prompt=prompt,
            system_prompt="你是一位专业的古典文学学者，擅长综合分析和深度解读。",
            max_tokens=2000,
            temperature=0.7
        )

    def _basic_synthesis(self, query: str, subtasks_results: List[Dict]) -> str:
        """基础综合方法（无API时使用）"""
        synthesis = f"关于「{query}」的多角度分析：\n\n"
        
        for idx, subtask_result in enumerate(subtasks_results, 1):
            synthesis += f"**{idx}. {subtask_result['subtask_focus']}**\n"
            synthesis += f"检索：{subtask_result['subtask_query']}\n"
            
            for item in subtask_result['results'][:2]:
                meta = item['metadata']
                synthesis += f"- 《{meta['book']}·{meta['chapter']}》：{item['content']}\n"
            
            synthesis += "\n"
        
        return synthesis    
    
    def generate_answer(self, query: str, context: list) -> str:
        """生成答案 - 通用实现"""
        if not self.is_available():
            return self._generate_basic_answer(query, context)
        
        context_parts = []
        for i,item in enumerate(context[:3]):
            meta = item['metadata']
            context_parts.append(
                f"【{meta['book']} · {meta['chapter']}】: {item['content']}"
            )
            # 如果有上下文信息，也显示出来
            nested_meta = meta.get('metadata', {})
            if 'prev_context' in nested_meta and nested_meta['prev_context']:
                context_parts[i] += f"\n前文：{nested_meta['prev_context']}"
            if 'next_context' in nested_meta and nested_meta['next_context']:
                context_parts[i] += f"\n后文：{nested_meta['next_context']}"

        context_text = "\n\n".join(context_parts)
        
        prompt = f"""你是一位精通中国古典文学的学者，请基于以下古文原文回答用户的问题。

相关古文原文及上下文：
{context_text}

用户问题：{query}

请按以下要求分点回答：
1. 首先引用最相关的原文
2. 详细完整地回答用户问题
3. 解释古文的字面含义
4. 阐述其深层思想内涵
5. 结合现代观点进行分析
6. 提供实际的指导意义

回答要求：第二部分详细回答，其余回答要简洁明了，重点突出，既有学术深度又通俗易懂。

回答："""
        return self.call_chat_completion(
            prompt=prompt,
            system_prompt="你是一位专业的古典文学学者。"
        )
    
    def _generate_basic_answer(self, query: str, context: list) -> str:
        """生成基础回答（无API时使用）"""
        answer = f"关于「{query}」，在古文中找到以下相关内容：\n\n"
        
        for i, item in enumerate(context[:3], 1):
            meta = item['metadata']
            answer += f"**{i}. 《{meta['book']}·{meta['chapter']}》**\n"
            answer += f"原文：「{item['content']}」\n"
            answer += f"话题：{meta['topic']}\n\n"
        
        return answer


class DeepSeekAPIClient(BaseAPIClient):
    """DeepSeek API 客户端实现"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        super().__init__(api_key)
        self.base_url = base_url
        self.default_model = "deepseek-chat"
    
    def call_chat_completion(
        self, 
        prompt: str, 
        system_prompt: str = None,
        model: str = None,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        timeout: int = 30
    ) -> str:
        if not self.is_available():
            return "API密钥未配置"
        
        try:
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})
            
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            data = {
                'model': model or self.default_model,
                'messages': messages,
                'stream': False,
                'max_tokens': max_tokens,
                'temperature': temperature
            }
            
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=data,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                return f"API调用失败: HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "API调用超时，请检查网络连接"
        except requests.exceptions.RequestException as e:
            return f"网络错误: {str(e)}"
        except Exception as e:
            return f"API调用出错: {str(e)}"


class GLMAPIClient(BaseAPIClient):
    """智谱 GLM API 客户端实现"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.default_model = "glm-4-air-250414"
        self._client = None
    
    @property
    def client(self):
        """延迟初始化 ZhipuAI 客户端"""
        if self._client is None and self.api_key:
            try:
                from zhipuai import ZhipuAI
                self._client = ZhipuAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("请先安装 zhipuai 库: pip install zhipuai")
        return self._client
    
    def call_chat_completion(
        self, 
        prompt: str, 
        system_prompt: str = None,
        model: str = None,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        timeout: int = 30
    ) -> str:
        if not self.is_available():
            return "API密钥未配置"
        
        try:
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})
            
            # 调用 GLM API
            response = self.client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout
            )
            
            # 提取响应内容
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                return "GLM API 返回格式异常"
                
        except ImportError as e:
            return f"GLM 库未安装: {str(e)}"
        except Exception as e:
            return f"GLM API 调用出错: {str(e)}"


class APIClientFactory:
    """API 客户端工厂类"""
    
    @staticmethod
    def create_client(
        provider: APIProvider,
        api_key: str,
        **kwargs
    ) -> BaseAPIClient:
        """
        创建 API 客户端实例
        
        Args:
            provider: API 提供商
            api_key: API 密钥
            **kwargs: 其他参数（如 base_url）
            
        Returns:
            BaseAPIClient 实例
        """
        if provider == APIProvider.DEEPSEEK:
            return DeepSeekAPIClient(
                api_key=api_key,
                base_url=kwargs.get('base_url', 'https://api.deepseek.com')
            )
        elif provider == APIProvider.GLM:
            return GLMAPIClient(api_key=api_key)
        else:
            raise ValueError(f"不支持的 API 提供商: {provider}")


# 使用示例
def get_api_client(config: Dict[str, Any]) -> BaseAPIClient:
    """
    从配置中获取 API 客户端
    
    配置示例:
    config = {
        'provider': 'deepseek',  # 或 'glm'
        'api_key': 'your-api-key',
        'base_url': 'https://api.deepseek.com'  # 可选，仅 DeepSeek 需要
    }
    """
    provider = APIProvider(config.get('provider', 'deepseek'))
    api_key = config.get('api_key', '')
    
    return APIClientFactory.create_client(
        provider=provider,
        api_key=api_key,
        base_url=config.get('base_url')
    )
