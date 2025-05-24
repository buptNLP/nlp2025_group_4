import streamlit as st
import re
import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import hashlib
import requests


# 页面配置
st.set_page_config(
    page_title="论语智能问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class LunyuSegment:
    """论语片段数据结构"""
    chapter: str
    speaker: str
    content: str
    topic: str
    segment_id: str

class LunyuRAG:
    """论语RAG系统"""
    
    def __init__(self):
        self.client = chromadb.Client()
        self.collection_name = "lunyu_collection"
        self.segments: List[LunyuSegment] = []
        
        # 初始化向量数据库
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "论语向量数据库"}
            )
    
    def preprocess_lunyu_data(self, raw_text: str) -> List[LunyuSegment]:
        """预处理论语数据"""
        segments = []
        segment_counter = 0  # 添加计数器确保唯一性
        
        # 清理文本
        text = raw_text.strip()
        
        # 按篇章分割
        chapters = []
        current_chapter = ""
        current_content = []
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 检测篇章标题（如"学而第一"）
            if re.match(r'[^第]*第[一二三四五六七八九十]+', line):
                if current_chapter and current_content:
                    chapters.append((current_chapter, '\n'.join(current_content)))
                current_chapter = line
                current_content = []
            else:
                current_content.append(line)
        
        # 添加最后一章
        if current_chapter and current_content:
            chapters.append((current_chapter, '\n'.join(current_content)))
        
        for chapter_name, chapter_content in chapters:
            sentences = self.split_sentences(chapter_content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                
                speaker = self.identify_speaker(sentence)
                topic = self.classify_topic(sentence)
                
                # 修改这里：使用计数器生成唯一ID
                segment_id = f"{segment_counter:04d}"  # 0001, 0002, 0003...
                segment_counter += 1
                
                segment = LunyuSegment(
                    chapter=chapter_name,
                    speaker=speaker,
                    content=sentence,
                    topic=topic,
                    segment_id=segment_id
                )
                segments.append(segment)
        
        return segments
    
    def split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        # 按照明显的分句标记分割
        sentences = re.split(r'[。！？]', text)
        
        # 进一步处理复杂对话
        result = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # 如果句子很长，尝试按对话分割
            if len(sentence) > 100 and '曰' in sentence:
                parts = re.split(r'([^曰]{1,20}曰)', sentence)
                temp_sentence = ""
                for part in parts:
                    temp_sentence += part
                    if '曰' in part and temp_sentence.strip():
                        result.append(temp_sentence.strip())
                        temp_sentence = ""
                if temp_sentence.strip():
                    result.append(temp_sentence.strip())
            else:
                result.append(sentence)
        
        return result
    
    def identify_speaker(self, text: str) -> str:
        """识别说话人"""
        speakers = {
            '子曰': '孔子',
            '有子曰': '有子',
            '曾子曰': '曾子',
            '子夏曰': '子夏',
            '子游曰': '子游',
            '子贡曰': '子贡',
            '子贡问': '子贡',
            '子路曰': '子路',
            '子路问': '子路',
            '颜渊曰': '颜渊',
            '颜渊问': '颜渊',
            '子张曰': '子张',
            '子张问': '子张',
            '樊迟问': '樊迟',
            '季康子问': '季康子',
            '哀公问': '哀公',
            '定公问': '定公'
        }
        
        for pattern, speaker in speakers.items():
            if text.startswith(pattern):
                return speaker
        
        # 检查其他对话模式
        if '问曰' in text or '问于' in text:
            match = re.search(r'([^问]+)问', text)
            if match:
                return match.group(1)
        
        return '孔子'  # 默认
    
    def classify_topic(self, text: str) -> str:
        """分类话题"""
        topics = {
            '学习教育': ['学', '习', '教', '诲', '知', '问', '思'],
            '品德修养': ['仁', '义', '礼', '智', '信', '德', '善', '修'],
            '政治治理': ['政', '君', '臣', '民', '国', '治', '邦'],
            '人际关系': ['友', '朋', '交', '人', '亲', '群'],
            '人生哲学': ['道', '天', '命', '生', '死', '乐', '忧'],
            '社会礼仪': ['礼', '乐', '祭', '丧', '婚', '冠']
        }
        
        for topic, keywords in topics.items():
            if any(keyword in text for keyword in keywords):
                return topic
        
        return '其他'
    
    def load_data(self, text_data: str):
        """加载数据到向量数据库"""
        with st.spinner("正在处理论语数据..."):
            # 预处理数据
            self.segments = self.preprocess_lunyu_data(text_data)
            
            # 检查是否已有数据
            if self.collection.count() > 0:
                # 清空旧数据
                self.collection.delete(where={})
            
            # 批量存储
            documents = []
            metadatas = []
            ids = []
            
            for segment in self.segments:
                documents.append(segment.content)
                metadatas.append({
                    'chapter': segment.chapter,
                    'speaker': segment.speaker,
                    'topic': segment.topic
                })
                ids.append(segment.segment_id)
            
            # 分批添加到向量数据库
            batch_size = 50
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
        
        return len(self.segments)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相关内容"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    result = {
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'score': 1 - results['distances'][0][i] if results['distances'] else 1.0
                    }
                    search_results.append(result)
            
            return search_results
        except Exception as e:
            st.error(f"搜索出错: {e}")
            return []
    
    def call_deepseek_api(self, prompt: str, api_key: str) -> str:
        """调用DeepSeek API"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}'
            }
            
            data = {
                'model': 'deepseek-chat',
                'messages': [
                    {'role': 'system', 'content': '你是一位专业的古典文学学者，精通论语等古代典籍。'},
                    {'role': 'user', 'content': prompt}
                ],
                'stream': False,
                'max_tokens': 1000,
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
                return f"API调用失败: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"API调用出错: {str(e)}"
    
    def generate_answer(self, query: str, context: List[Dict[str, Any]], api_key: str = None) -> str:
        """生成答案"""
        if not context:
            return "抱歉，没有找到相关的论语内容。"
        
        # 构建上下文
        context_text = ""
        for item in context[:3]:  # 使用前3个最相关的结果
            meta = item['metadata']
            context_text += f"【{meta['chapter']}】{meta['speaker']}: {item['content']}\n\n"
        
        # 构建提示词
        prompt = f"""你是一位精通中国古典文学的学者，请基于以下论语原文回答用户的问题。

相关论语原文：
{context_text.strip()}

用户问题：{query}

请按以下要求回答：
1. 首先引用相关的原文
2. 解释古文的含义
3. 结合现代观点进行阐述
4. 回答要简洁明了，重点突出

回答："""

        # 如果有API密钥，使用DeepSeek API
        if api_key and api_key.strip():
            try:
                answer = self.call_deepseek_api(prompt, api_key.strip())
                return answer
            except Exception as e:
                return f"AI生成回答时出错: {e}\n\n基于搜索结果：\n{context_text}"
        else:
            # 生成简单的基于规则的回答
            answer = f"关于「{query}」，根据论语相关内容：\n\n"
            
            for i, item in enumerate(context[:3], 1):
                meta = item['metadata']
                answer += f"**{i}. {meta['chapter']} - {meta['speaker']}**\n"
                answer += f"原文：「{item['content']}」\n"
                answer += f"话题：{meta['topic']} | 相关度：{item['score']:.2f}\n\n"
            
            # 添加简单总结
            main_topics = list(set([item['metadata']['topic'] for item in context[:3]]))
            if main_topics:
                answer += f"💡 **总结**：以上内容主要涉及{' '.join(main_topics)}等方面的思想。"
            
            return answer

def load_lunyu_from_file():
    """从论语.txt文件中加载数据"""
    try:
        with open('论语.txt', 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 提取<content></content>标签中的内容
        match = re.search(r'<content>(.*?)</content>', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            st.error("未找到<content>标签")
            return ""
    except FileNotFoundError:
        st.error("未找到论语.txt文件")
        return ""
    except Exception as e:
        st.error(f"读取文件出错: {e}")
        return ""


def main():
    """主应用"""
    st.title("📚 论语智能问答系统")
    st.markdown("*基于RAG技术的古典文献问答系统*")
    st.markdown("---")
    
    # 初始化系统
    if 'rag_system' not in st.session_state:
        try:
            st.session_state.rag_system = LunyuRAG()
        except Exception as e:
            st.error(f"系统初始化失败: {e}")
            st.info("建议使用简化版本，或检查ChromaDB安装。")
            return
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # API配置
        st.subheader("AI配置")
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            help="输入你的DeepSeek API密钥以获得AI回答，留空则使用基础回答"
        )
        
        if not api_key:
            st.info("💡 没有API密钥？系统仍可搜索和显示相关原文，只是不会有AI生成的回答。")
        
        # 搜索配置
        st.subheader("搜索设置")
        top_k = st.slider("返回结果数量", 1, 10, 5)
        
        # 数据管理
        st.subheader("数据管理")

        
        if st.button("🔄 加载论语数据", type="primary"):
            try:
                with st.spinner("正在加载数据..."):
                    lunyu_data = load_lunyu_from_file()  # 从文件加载
                    if lunyu_data:
                        count = st.session_state.rag_system.load_data(lunyu_data)
                        st.success(f"✅ 成功加载 {count} 条数据！")
            except Exception as e:
                st.error(f"数据加载失败: {e}")
        
        # 显示数据状态
        try:
            data_count = st.session_state.rag_system.collection.count()
            st.metric("已加载数据", f"{data_count} 条")
        except:
            st.metric("已加载数据", "0 条")
    
    # 主界面
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 智能问答")
        
        # 预设问题
        st.subheader("📋 常见问题")
        example_questions = [
            "孔子对学习的看法是什么？",
            "什么是仁？",
            "君子和小人的区别是什么？",
            "孔子的教育思想有哪些？",
            "论语中关于政治的观点",
            "如何修身养性？"
        ]
        
        selected_question = st.selectbox(
            "选择一个问题，或在下方输入自定义问题：",
            [""] + example_questions
        )
        
        # 问题输入
        if selected_question:
            query = st.text_input("您的问题", value=selected_question)
        else:
            query = st.text_input("您的问题", placeholder="请输入您想了解的内容...")
        
        # 搜索和问答
        if st.button("🔍 提问", type="primary") and query:
            with st.spinner("正在搜索相关内容..."):
                # 搜索
                search_results = st.session_state.rag_system.search(query, top_k)
                
                if search_results:
                    # 显示搜索结果
                    st.subheader("📖 相关原文")
                    for i, result in enumerate(search_results):
                        meta = result['metadata']
                        with st.expander(f"📜 {meta['chapter']} - {meta['speaker']} (相关度: {result['score']:.2f})"):
                            st.write(f"**话题**: {meta['topic']}")
                            st.write(f"**原文**: {result['content']}")
                    
                    # 生成答案
                    st.subheader("🤖 智能解答")
                    with st.spinner("正在生成回答..."):
                        answer = st.session_state.rag_system.generate_answer(
                            query, search_results, api_key
                        )
                        st.markdown(answer)
                else:
                    st.warning("😔 没有找到相关内容，请尝试其他问题。")
    
    with col2:
        st.header("📊 系统信息")
        
        # 系统状态
        st.info("""
        **🔧 系统状态**
        - ✅ ChromaDB向量搜索
        - ✅ 古文智能分析
        - 🔑 DeepSeek AI回答 (需API密钥)
        - 📊 数据统计分析
        """)
        
        # 数据统计
        if hasattr(st.session_state.rag_system, 'segments') and st.session_state.rag_system.segments:
            segments = st.session_state.rag_system.segments
            
            st.metric("文本片段", len(segments))
            
            # 说话人统计
            speakers = {}
            topics = {}
            chapters = {}
            
            for seg in segments:
                speakers[seg.speaker] = speakers.get(seg.speaker, 0) + 1
                topics[seg.topic] = topics.get(seg.topic, 0) + 1
                chapters[seg.chapter] = chapters.get(seg.chapter, 0) + 1
            
            st.subheader("🗣️ 说话人分布")
            for speaker, count in sorted(speakers.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.write(f"**{speaker}**: {count} 条")
            
            st.subheader("🏷️ 话题分布")
            for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]:
                st.write(f"**{topic}**: {count} 条")
        
        # 使用说明
        st.subheader("💡 使用提示")
        st.markdown("""
        **基础功能**（无需API）:
        - 🔍 搜索论语相关内容
        - 📖 显示原文和出处
        - 📊 查看数据统计
        
        **高级功能**（需API密钥）:
        - 🤖 AI智能回答
        - 💬 古文现代解释
        - 🎯 个性化问答
        """)

if __name__ == "__main__":
    # 样式优化
    st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .stButton>button {
        border-radius: 20px;
    }
    .stSelectbox>div>div>div {
        border-radius: 10px;
    }
    .stExpander {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    main()