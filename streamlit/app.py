import streamlit as st
# 页面配置
st.set_page_config(
    page_title="通用古文智能问答系统",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)
from collections import defaultdict

from config import HAS_RERANKER
from rag_system import UniversalAncientRAG


def main():
    def perform_regular_search(query, top_k, search_mode, metadata_filter):
            """执行常规检索的函数"""
            # 检查是否启用智能分析
            use_intelligent_analysis = st.session_state.get('query_optimization', False) and api_key and api_key.strip()
            
            if use_intelligent_analysis:
                with st.spinner("正在智能分析查询..."):
                    # 使用综合分析方法
                    analysis_result, results = st.session_state.rag_system.analyze_and_search(
                        query, top_k, search_mode, metadata_filter
                    )
                    
                    # 显示AI自动筛选的提示
                    if analysis_result.get('filter_source') == 'ai':
                        st.info(f"🤖 已智能识别书名《{analysis_result['extracted_book']}》并自动筛选。")
                    
                    # 显示分析结果
                    if not analysis_result['need_search']:
                        # 不需要检索的情况
                        st.info(f"🤖 **智能判断**: {analysis_result['reason']}")
                        
                        if analysis_result['direct_answer']:
                            st.markdown("### 💡 直接回答")
                            st.markdown(analysis_result['direct_answer'])
                        else:
                            st.markdown("这个问题不需要检索古文数据库。如果您坚持要搜索，请关闭'智能查询优化'选项。")
                        
                        # 清空结果
                        st.session_state.last_results = None
                        return  # 提前返回，不需要继续执行
                    
                    else:
                        # 需要检索的情况
                        st.success("✅ **智能分析完成**")
                        
                        # 显示优化信息
                        col1_info, col2_info = st.columns(2)
                        
                        with col1_info:
                            if analysis_result['optimized_query'] != query:
                                st.markdown(f"""
                                🔍 **查询优化**
                                - **原始**: {query}
                                - **优化**: {analysis_result['optimized_query']}
                                """)
                        
                        with col2_info:
                            st.info(f"⚖️ **检索权重**\nBM25: {analysis_result['bm25_weight']:.1%}\n向量: {analysis_result['vector_weight']:.1%}")
                        
                        # 保存权重信息
                        st.session_state.weights_info = {
                            'bm25_weight': analysis_result['bm25_weight'],
                            'vector_weight': analysis_result['vector_weight'],
                            'query': analysis_result['optimized_query'],
                            'has_api': True,
                            'analysis_reason': analysis_result['reason']
                        }
                        
                        # 保存结果
                        st.session_state.last_results = (query, results, analysis_result['optimized_query'])
            
            else:
                # 不使用智能分析，直接搜索
                with st.spinner("正在搜索相关古文..."):
                    if metadata_filter:
                        st.info(f"🎯 使用精确过滤: {metadata_filter}")
                        results = st.session_state.rag_system.search_with_metadata(
                            query, top_k, search_mode, metadata_filter
                        )
                        
                        if not results:
                            st.warning(f"⚠️ 在指定范围内未找到相关内容，扩展到全部数据搜索")
                            results = st.session_state.rag_system.search(
                                query, top_k, search_mode
                            )
                    else:
                        results = st.session_state.rag_system.search(
                            query, top_k, search_mode
                        )
                    
                    st.session_state.last_results = (query, results, None)
    """主应用"""
    st.title("📜 通用古文智能问答系统")
    st.markdown("*基于混合检索技术的智能古文RAG系统*")
    st.markdown("---")
    
    # 初始化系统
    if 'rag_system' not in st.session_state:
        # 初始化时指定中文优化的embedding模型
        st.session_state.rag_system = UniversalAncientRAG(
            embedding_model="BAAI/bge-large-zh-v1.5",
            max_chunk_size=150,  # 可配置
            min_chunk_size=20,
            context_window=80
        )
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # 数据加载
        st.subheader("📚 数据管理")
        data_dir = st.text_input("古文数据目录路径", value="../data")
        
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
            # 新增分块配置
            st.subheader("📏 分块参数")
            max_chunk_size = st.slider("最大块大小", 100, 300, 150, 
                                    help="超过此长度的段落将被分割")
            min_chunk_size = st.slider("最小块大小", 10, 50, 20,
                                    help="小于此长度的段落将被过滤")
            context_window = st.slider("上下文窗口", 50, 100, 80,
                                    help="保留的上下文长度")
            show_processing_details = st.checkbox("显示处理详情", value=True)
            
        if st.button("🔄 加载古文数据", type="primary"):
            with st.spinner("正在加载古文数据..."):
                # 清空之前的数据
                st.session_state.rag_system.segments = []
                
                # 更新分块器参数
                current_params = st.session_state.rag_system.update_chunker_params(
                    max_chunk_size=max_chunk_size,
                    min_chunk_size=min_chunk_size,
                    context_window=context_window
                )
                
                # 显示当前参数
                st.info(f"📏 分块参数：最大{current_params['max_chunk_size']}字，"
                        f"最小{current_params['min_chunk_size']}字，"
                        f"上下文{current_params['context_window']}字")
                
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
        
        # 选择 API 提供商
        api_provider = st.selectbox(
            "选择 AI 模型提供商",
            options=['deepseek', 'glm'],
            format_func=lambda x: 'DeepSeek' if x == 'deepseek' else '智谱 GLM',
            help="选择要使用的 AI 模型提供商"
        )
        
        # API 密钥输入
        api_key = st.text_input(
            f"{api_provider.upper()} API Key", 
            type="password",
            placeholder=f"请输入您的 {api_provider.upper()} API 密钥"
        )
        
        # 高级设置（仅 DeepSeek 需要）
        if api_provider == 'deepseek':
            base_url = "https://api.deepseek.com"
        else:
            base_url = None

        # 设置API客户端
        if api_key:
            # 构建配置
            api_config = {
                'provider': api_provider,
                'api_key': api_key
            }
            if base_url:
                api_config['base_url'] = base_url
            
            # 保存配置到 session state
            if 'api_config' not in st.session_state:
                st.session_state.api_config = api_config
            else:
                # 检查配置是否改变
                config_changed = (
                    st.session_state.api_config.get('provider') != api_provider or
                    st.session_state.api_config.get('api_key') != api_key or
                    st.session_state.api_config.get('base_url') != base_url
                )
                if config_changed:
                    st.session_state.api_config = api_config
                    # 重新设置 API 客户端
                    st.session_state.rag_system.set_api_config(api_config)
                    st.success(f"✅ 已切换到 {api_provider.upper()} API")
            
            # 初始设置
            st.session_state.rag_system.set_api_config(api_config)

        # API配置部分后添加
        query_optimization = st.checkbox(
            "🔍 智能查询优化", 
            value=bool(api_key and api_key.strip()),  # 有API KEY时默认开启
            disabled=not bool(api_key and api_key.strip()),  # 无API KEY时禁用
            help="使用AI优化用户输入的问题，提高检索准确性"
        )
        st.session_state.query_optimization = query_optimization
        
        st.subheader("🔄 高级检索选项")

        # 多轮检索开关
        enable_multi_round = st.checkbox(
            "🎯 启用多轮检索", 
            value=False,
            disabled=not bool(api_key and api_key.strip()),
            help="将复杂问题拆解为多个子任务分别检索，适合需要多角度分析的问题"
        )

        if enable_multi_round and not api_key:
            st.warning("多轮检索需要配置AI API")
        
        # 检索配置
        st.subheader("🔍 检索设置")

        # Embedding模型选择
        embedding_models = [
            "BAAI/bge-large-zh-v1.5",
            "../finetuned_bge_ancient"
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
        
        # 修复后的重排序选项
        st.subheader("🔄 深度重排序")
        
        if HAS_RERANKER:
            # 检查依赖状态
            try:
                from sentence_transformers import CrossEncoder
                dependency_ok = True
            except ImportError:
                dependency_ok = False            
            
            use_reranker = st.checkbox(
                "启用深度重排序", 
                value=dependency_ok and st.session_state.get('use_reranker', False),
                disabled=not dependency_ok,
                help="使用BGE模型进行深度语义重排序，提高结果准确性"
            )
            
            # 保存到session state
            st.session_state.use_reranker = use_reranker and dependency_ok
            
            if use_reranker and dependency_ok:
                reranker_model = st.selectbox(
                    "重排序模型",
                    ["BAAI/bge-reranker-large", "BAAI/bge-reranker-base"],
                    help="选择重排序模型，large版本效果更好但速度较慢"
                )
                st.session_state.reranker_model = reranker_model
        
        # 显示统计信息
        if hasattr(st.session_state.rag_system, 'segments') and st.session_state.rag_system.segments:
            st.subheader("📊 数据统计")
            segments = st.session_state.rag_system.segments
            
            # 基本统计
            st.metric("文本片段总数", len(segments))
            
            # 书籍分布
            books = defaultdict(int)
            topics = defaultdict(int)
            
            for seg in segments:
                books[seg.book] += 1
                topics[seg.topic] += 1
            
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

        # 元数据过滤输入框
        st.subheader("🎯 精确范围（可选）")
        col_book, col_chapter = st.columns(2)
        
        with col_book:
            filter_book = st.text_input(
                "指定书名",
                placeholder="如：论语、孟子",
                help="留空表示不限制书籍范围"
            )
        
        with col_chapter:
            filter_chapter = st.text_input(
                "指定篇章", 
                placeholder="如：学而、梁惠王",
                help="留空表示不限制章节范围"
            )
        
        # 搜索按钮
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            search_btn = st.button("🔍 智能搜索", type="primary")
        with col_btn2:
            clear_btn = st.button("🗑️ 清空结果")
        
        if clear_btn:
            st.session_state.last_results = None
            if 'weights_info' in st.session_state:
                del st.session_state.weights_info
            st.rerun()
        
        # 处理搜索
        if search_btn and query:
            # 构建元数据过滤条件
            metadata_filter = {}
            if filter_book and filter_book.strip():
                metadata_filter['book'] = filter_book.strip()
            if filter_chapter and filter_chapter.strip():
                metadata_filter['chapter'] = filter_chapter.strip()
            
            # 判断是否使用多轮检索
            if enable_multi_round and api_key and api_key.strip():
                # 执行多轮检索
                with st.spinner("正在执行多轮检索..."):
                    multi_results = st.session_state.rag_system.multi_round_search(
                        query, top_k, search_mode
                    )
                    
                    # 保存结果
                    st.session_state.multi_round_results = multi_results
                    
                # 显示多轮检索结果
                if multi_results['decomposition'].get('need_multi_round'):
                    st.markdown("### 🎯 多轮检索分析")
                    
                    # 显示任务拆解
                    decomp = multi_results['decomposition']
                    st.info(f"**分析结果**: {decomp['reason']}")
                    
                    # 显示子任务
                    st.markdown("#### 📋 子任务列表")
                    for subtask in decomp['subtasks']:
                        st.write(f"{subtask['subtask_id']}. **{subtask['subtask_focus']}**")
                        st.caption(f"   检索语句: {subtask['subtask_query']}")
                    
                    # 显示综合结果
                    st.markdown("### 🤖 综合分析结果")
                    st.markdown(multi_results['synthesis'])
                    
                    # 可选：显示详细的各子任务结果
                    with st.expander("查看各子任务详细结果"):
                        for subtask_result in multi_results['subtasks_results']:
                            st.subheader(f"子任务: {subtask_result['subtask_focus']}")
                            for res in subtask_result['results'][:3]:
                                meta = res['metadata']
                                st.write(f"- 《{meta['book']}·{meta['chapter']}》: {res['content']}")
                else:
                    # 多轮检索判断不需要，转为常规检索
                    st.info("该问题不需要多轮检索，使用常规检索")
                    # 执行常规检索（调用下面的函数）
                    perform_regular_search(query, top_k, search_mode, metadata_filter)
            
            else:
                # 执行常规检索
                perform_regular_search(query, top_k, search_mode, metadata_filter)
        
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
                        
                        # 显示段落信息
                        if 'paragraph_index' in meta and 'sub_index' in meta:
                            para_info = f"第 {meta['paragraph_index']+1} 段"
                            if meta.get('is_continuation', False):
                                para_info += f" - 第 {meta['sub_index']+1} 部分"
                            st.caption(f"📌 {para_info}")
                                            
                        # 显示各种分数 - 增强版
                        if search_mode == 'hybrid' or 'rerank_score' in result:
                            # 创建两行分数显示
                            st.write("**📊 检索评分**")
                            
                            # 第一行：归一化分数（相对分数）
                            score_cols = st.columns(4)
                            with score_cols[0]:
                                if 'bm25_score' in result:
                                    bm25_score = result['bm25_score']
                                    color = "🟢" if bm25_score > 0.5 else "🟡" if bm25_score > 0.2 else "🔴"
                                    st.metric(
                                        "BM25 (归一化)", 
                                        f"{bm25_score:.3f}", 
                                        help=f"{color} 关键词匹配度（相对）"
                                    )
                            with score_cols[1]:
                                if 'vector_score' in result:
                                    vector_score = result['vector_score']
                                    color = "🟢" if vector_score > 0.8 else "🟡" if vector_score > 0.6 else "🔴"
                                    st.metric(
                                        "向量 (归一化)", 
                                        f"{vector_score:.3f}", 
                                        help=f"{color} 语义相似度（相对）"
                                    )
                            with score_cols[2]:
                                if 'combined_score' in result:
                                    combined_score = result['combined_score']
                                    color = "🟢" if combined_score > 0.7 else "🟡" if combined_score > 0.5 else "🔴"
                                    st.metric(
                                        "融合 (加权)", 
                                        f"{combined_score:.3f}", 
                                        help=f"{color} 综合评分"
                                    )
                            with score_cols[3]:
                                if 'rerank_score' in result:
                                    rerank_score = result['rerank_score']
                                    color = "🟢" if rerank_score > 0.8 else "🟡" if rerank_score > 0.6 else "🔴"
                                    st.metric(
                                        "重排序", 
                                        f"{rerank_score:.3f}", 
                                        help=f"{color} 深度语义评分"
                                    )
                            
                            # 第二行：原始分数（绝对分数） - 优化版
                            if any(key in result for key in ['bm25_score_raw', 'vector_score_raw']):
                                # 使用更小的字体显示原始分数，与上面的4列对应
                                raw_cols = st.columns(4)
                                
                                with raw_cols[0]:
                                    if 'bm25_score_raw' in result:
                                        st.caption(f"原始: {result['bm25_score_raw']:.4f}")
                                
                                with raw_cols[1]:
                                    if 'vector_score_raw' in result:
                                        raw_vector = result.get('vector_score_raw', 0)
                                        st.caption(f"原始: {raw_vector:.4f}")
                                
                                with raw_cols[2]:
                                    # 显示融合相关的原始信息（如果有的话）
                                    if 'combined_score_raw' in result:
                                        st.caption(f"原始: {result['combined_score_raw']:.4f}")
                                    elif 'length' in meta:
                                        st.caption(f"长度: {meta['length']}字")
                                
                                with raw_cols[3]:
                                    # 显示重排序相关的原始信息或其他元数据
                                    if 'rerank_score_raw' in result:
                                        st.caption(f"原始: {result['rerank_score_raw']:.4f}")
                                    elif 'paragraph_index' in meta:
                                        st.caption(f"第{meta['paragraph_index']+1}段")
                                        
                        # 改进的上下文显示
                        if 'prev_context' in meta and 'next_context' in meta:
                            col_prev, col_next = st.columns(2)
                            
                            with col_prev:
                                if meta['prev_context']:
                                    st.text_area("前文", meta['prev_context'], height=60, 
                                                disabled=True, key=f"prev_{i}")
                            
                            with col_next:
                                if meta['next_context']:
                                    st.text_area("后文", meta['next_context'], height=60, 
                                                disabled=True, key=f"next_{i}")
                        
                        # 原有的完整上下文显示
                        elif 'context' in meta and meta['context'] != result['content']:
                            st.write("**📄 上下文：**")
                            st.text_area("", meta['context'], height=100, disabled=True, 
                                        key=f"context_{i}")

                # 生成智能回答
                st.subheader("🤖 智能解答")
                with st.spinner("正在生成智能回答..."):
                    answer = st.session_state.rag_system.generate_answer(
                        original_query, results
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