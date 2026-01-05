import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ---------------------------------------------------------
# 1. 核心設定 & CSS (按鈕化標籤風格)
# ---------------------------------------------------------
st.set_page_config(
    page_title="設備維修智能中樞 Ultimate V3",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 Session State
if 'active_tab' not in st.session_state:
    st.session_state['active_tab'] = "tab_ai"
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = None
if 'target_case_id' not in st.session_state:
    st.session_state['target_case_id'] = None
if 'target_category' not in st.session_state:
    st.session_state['target_category'] = "全部顯示"
if 'target_topic' not in st.session_state:
    st.session_state['target_topic'] = "全部顯示"

# CSS 高度客製化
st.markdown("""
<style>
    /* 全域字體 */
    html, body, [class*="css"] {
        font-family: "Microsoft JhengHei", "Segoe UI", sans-serif;
    }
    
    /* 側邊欄按鈕 */
    div[data-testid="stSidebar"] button {
        width: 100%;
        text-align: left;
        background-color: transparent;
        border: 1px solid #e0e0e0;
        margin-bottom: 5px;
        color: #31333F;
        transition: all 0.2s;
    }
    div[data-testid="stSidebar"] button:hover {
        background-color: #f0f2f6;
        border-color: #ff4b4b;
        color: #ff4b4b;
        padding-left: 15px;
        font-weight: bold;
    }
    
    /* 隱藏原生 Tabs */
    .stTabs [data-baseweb="tab-list"] { display: none; }
    
    /* === 魔改 Radio Button 變成 按鈕標籤 (Directory Style) === */
    div.row-widget.stRadio > div {
        flex-direction: row;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
    }
    div.row-widget.stRadio > div > label {
        background-color: #f0f2f6;
        padding: 8px 16px;
        border-radius: 20px;
        border: 1px solid #e0e0e0;
        cursor: pointer;
        transition: all 0.2s;
        margin-right: 0px !important;
    }
    div.row-widget.stRadio > div > label:hover {
        background-color: #e2e8f0;
        border-color: #cbd5e0;
    }
    /* 被選中的狀態 */
    div.row-widget.stRadio > div > label[data-checked="true"] {
        background-color: #ff4b4b !important;
        color: white !important;
        border-color: #ff4b4b !important;
    }

    /* === 聚合式卡片設計 === */
    .topic-container {
        border: 1px solid #ddd;
        border-radius: 12px;
        margin-bottom: 20px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        background: white;
    }
    .topic-header {
        background-color: #f8f9fa;
        padding: 15px 20px;
        border-bottom: 1px solid #eee;
        font-size: 1.1rem;
        font-weight: bold;
        color: #2c3e50;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .record-row {
        padding: 20px;
        border-bottom: 1px solid #f0f0f0;
    }
    .record-row:last-child {
        border-bottom: none;
    }
    
    /* 目標資料高亮 */
    .highlight-record {
        background-color: #fff5f5; /* 淡淡的紅色背景 */
        border-left: 5px solid #ff4b4b;
    }

    /* 標籤小裝飾 */
    .badge {
        font-size: 0.8rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 10px;
        font-weight: normal;
    }
    .badge-gray { background: #e2e8f0; color: #4a5568; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 資料處理
# ---------------------------------------------------------
HAS_AI = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_AI = True
except ImportError:
    HAS_AI = False

CSV_FILE = '分類.csv'
SYNONYMS = {
    "聲音": "異音 噪音 吵雜 聲響", "怪聲": "異音 磨損",
    "不動": "卡死 異常 停止 無法運作失效", "壞掉": "異常 故障 損壞",
    "溫度": "過熱 發燙 高溫", "漏水": "洩漏 滲水",
    "轉速": "速度 變慢", "sensor": "感應器 光電",
    "馬達": "motor", "皮帶": "斷裂 磨損",
}

def expand_query(query):
    q = query
    for k, v in SYNONYMS.items():
        if k in query.lower(): q += " " + v
    return q

@st.cache_data(ttl=600)
def load_data():
    if not os.path.exists(CSV_FILE): return None
    try:
        try:
            df = pd.read_csv(CSV_FILE, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(CSV_FILE, encoding='cp950')
        df = df.fillna("無")
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        df['original_id'] = df.index
        df['search_content'] = (
            (df['設備型號'] + " ") * 3 + (df['主題(事件簡述)'] + " ") * 4 + 
            (df['原因(異常查找、分析)'] + " ") * 2 + df['處置、應對']
        )
        return df
    except Exception as e:
        st.error(f"讀取錯誤: {e}")
        return None

@st.cache_resource
def build_search_engine(df_content):
    if not HAS_AI: return None, None
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(df_content)
    return vectorizer, tfidf_matrix

def super_smart_search(query, df, vectorizer, tfidf_matrix):
    if not query: return pd.DataFrame(), ""
    smart_query = expand_query(query)
    results = pd.DataFrame()
    
    if HAS_AI and vectorizer:
        try:
            query_vec = vectorizer.transform([smart_query])
            sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
            top_idx = sim_scores.argsort()[-10:][::-1]
            results = df.iloc[top_idx].copy()
            results['score'] = sim_scores[top_idx]
            results = results[results['score'] > 0.1]
        except: pass

    if results.empty or len(results) < 2:
        keywords = query.split()
        mask = pd.Series([False]*len(df))
        for k in keywords: mask |= df['search_content'].str.contains(k, case=False, regex=False)
        keyword_res = df[mask].copy()
        keyword_res['score'] = 1.0
        results = pd.concat([results, keyword_res]).drop_duplicates(subset=['original_id']).head(10)

    summary = ""
    if not results.empty:
        results = results.reset_index(drop=True)
        top_cause = results['原因(異常查找、分析)'].iloc[0]
        top_sol = results['處置、應對'].iloc[0]
        summary = f"""
        🤖 **AI 分析報告**：
        推測問題核心與 **「{top_cause}」** 有關。
        建議處置：**「{top_sol}」**。
        """
    else:
        summary = "🤖 查無完全符合資料，請嘗試簡化關鍵字。"

    return results, summary

# ---------------------------------------------------------
# 3. 頁面控制
# ---------------------------------------------------------
def jump_to_case(model_name, case_id, category, topic):
    st.session_state['active_tab'] = "tab_catalog"
    st.session_state['selected_model'] = model_name
    st.session_state['target_case_id'] = case_id 
    st.session_state['target_category'] = category
    st.session_state['target_topic'] = topic

def set_tab(tab_name):
    st.session_state['active_tab'] = tab_name
    st.session_state['target_case_id'] = None

def set_model(model_name):
    st.session_state['active_tab'] = "tab_catalog"
    st.session_state['selected_model'] = model_name
    st.session_state['target_case_id'] = None
    st.session_state['target_category'] = "全部顯示"
    st.session_state['target_topic'] = "全部顯示"

# ---------------------------------------------------------
# 4. 主程式
# ---------------------------------------------------------
def main():
    df = load_data()
    if df is None: st.stop()
    
    vectorizer, tfidf_matrix = build_search_engine(df['search_content'])
    all_models = sorted(df['設備型號'].unique())

    # === SIDEBAR ===
    with st.sidebar:
        st.header("🎛️ 中控台")
        if st.button("🧠 AI 智能診斷", use_container_width=True): set_tab("tab_ai")
        if st.button("📊 全域戰情室", use_container_width=True): set_tab("tab_chart")
        st.markdown("---")
        st.caption("📂 設備目錄")
        with st.container(height=450):
            for model in all_models:
                prefix = "📍" if st.session_state.get('selected_model') == model else "📄"
                if st.button(f"{prefix} {model}", key=f"nav_{model}"):
                    set_model(model)
                    st.rerun()

    # === TAB 1: AI ===
    if st.session_state['active_tab'] == "tab_ai":
        st.markdown('<h1>🧠 設備維修大腦 Ultimate</h1>', unsafe_allow_html=True)
        query = st.text_input("💬 故障描述", placeholder="例如：主軸異音...", key="search")
        
        if query:
            with st.spinner("⚡ AI 檢索中..."):
                results, summary = super_smart_search(query, df, vectorizer, tfidf_matrix)
            st.info(summary)
            if not results.empty:
                for i, row in results.iterrows():
                    st.markdown(f"""
                    <div style="background:white; padding:15px; border-radius:10px; border-left:5px solid #ff4b4b; box-shadow:0 2px 5px rgba(0,0,0,0.05); margin-bottom:10px;">
                        <h3 style="margin:0; font-size:1.1rem;">🔧 {row['主題(事件簡述)']} <span style="font-size:0.8rem; background:#eee; padding:2px 6px; border-radius:4px;">{row['設備型號']}</span></h3>
                        <div style="margin-top:8px; color:#444;">
                            <b>🔴 原因：</b>{row['原因(異常查找、分析)'][:40]}...<br>
                            <b>🟢 對策：</b>{row['處置、應對'][:40]}...
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"🚀 開啟完整履歷", key=f"jump_{i}"):
                        jump_to_case(row['設備型號'], row['original_id'], row['大標'], row['主題(事件簡述)'])
                        st.rerun()

    # === TAB 2: CHART ===
    elif st.session_state['active_tab'] == "tab_chart":
        st.markdown('<h1>📊 全域戰情室</h1>', unsafe_allow_html=True)
        with st.expander("⚙️ 圖表資料篩選", expanded=True):
            col_ctrl_1, col_ctrl_2 = st.columns([1, 4])
            with col_ctrl_1:
                select_all = st.checkbox("全選所有機型", value=True)
            with col_ctrl_2:
                if select_all:
                    selected_models_chart = st.multiselect("選擇分析機型", all_models, default=all_models, disabled=True)
                    df_chart = df
                else:
                    default_sel = [all_models[0]] if all_models else []
                    selected_models_chart = st.multiselect("選擇分析機型", all_models, default=default_sel)
                    df_chart = df[df['設備型號'].isin(selected_models_chart)]
        st.divider()
        if not df_chart.empty:
            m1, m2, m3 = st.columns(3)
            m1.metric("案件數", len(df_chart))
            m2.metric("機型數", df_chart['設備型號'].nunique())
            m3.metric("分類數", df_chart['大標'].nunique())
            st.plotly_chart(px.treemap(df_chart, path=[px.Constant("全廠"), '設備型號', '大標', '主題(事件簡述)'], color='大標', color_discrete_sequence=px.colors.qualitative.Set3), use_container_width=True)
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(df_chart, names='設備型號', hole=0.4), use_container_width=True)
            c2.plotly_chart(px.bar(df_chart['主題(事件簡述)'].value_counts().head(10).reset_index(), x='count', y='主題(事件簡述)', orientation='h'), use_container_width=True)

    # === TAB 3: CATALOG (目錄聚合版) ===
    elif st.session_state['active_tab'] == "tab_catalog":
        target_model = st.session_state['selected_model']
        target_id = st.session_state['target_case_id']
        target_cat = st.session_state.get('target_category', "全部顯示")
        target_topic = st.session_state.get('target_topic', "全部顯示")
        
        if not target_model:
            st.warning("⚠️ 請從左側選擇機型")
            st.stop()
            
        st.markdown(f'<h1>📄 {target_model} 完整履歷</h1>', unsafe_allow_html=True)
        df_model = df[df['設備型號'] == target_model]

        # --- Level 1: 大標篩選 (按鈕式) ---
        st.markdown("### 1️⃣ 選擇分類")
        cats = ["全部顯示"] + sorted(df_model['大標'].unique().tolist())
        idx_cat = cats.index(target_cat) if target_cat in cats else 0
        
        # 使用 Radio 並透過 CSS 偽裝成按鈕群
        sel_cat = st.radio("大標", cats, index=idx_cat, horizontal=True, key="cat_filter", label_visibility="collapsed")
        st.session_state['target_category'] = sel_cat

        df_l1 = df_model if sel_cat == "全部顯示" else df_model[df_model['大標'] == sel_cat]

        # --- Level 2: 主題篩選 (按鈕式) ---
        if not df_l1.empty:
            st.markdown("### 2️⃣ 選擇主題")
            topics = ["全部顯示"] + sorted(df_l1['主題(事件簡述)'].unique().tolist())
            idx_topic = topics.index(target_topic) if target_topic in topics else 0
            
            # 使用 Radio 偽裝成按鈕群
            sel_topic = st.radio("主題", topics, index=idx_topic, horizontal=True, key="topic_filter", label_visibility="collapsed")
            st.session_state['target_topic'] = sel_topic
            
            df_final = df_l1 if sel_topic == "全部顯示" else df_l1[df_l1['主題(事件簡述)'] == sel_topic]
        else:
            df_final = pd.DataFrame()
            
        st.divider()

        # --- Level 3: 聚合式顯示 (Grouped View) ---
        if df_final.empty:
            st.info("此分類下無資料")
        else:
            # 依據「主題」進行分組 (Groupby)
            # 這樣相同主題的資料就會被包在同一個框框裡
            grouped = df_final.groupby('主題(事件簡述)')
            
            # 如果是 AI 跳轉，我們要確保目標主題排在第一個
            sorted_groups = []
            target_group_key = None
            
            if target_id is not None:
                # 找出 target_id 對應的主題
                target_row = df_final[df_final['original_id'] == target_id]
                if not target_row.empty:
                    target_group_key = target_row['主題(事件簡述)'].iloc[0]

            # 排序邏輯：目標主題置頂，其他依名稱排序
            group_keys = sorted(grouped.groups.keys())
            if target_group_key and target_group_key in group_keys:
                group_keys.remove(target_group_key)
                group_keys.insert(0, target_group_key)

            # 開始渲染每一個主題區塊
            for topic_name in group_keys:
                group_data = grouped.get_group(topic_name)
                
                # 建立主題大框框
                st.markdown(f"""
                <div class="topic-container">
                    <div class="topic-header">
                        <span>📌 {topic_name}</span>
                        <span class="badge badge-gray">{len(group_data)} 筆紀錄</span>
                    </div>
                """, unsafe_allow_html=True)
                
                # 在框框內列出每一筆紀錄
                for idx, row in group_data.iterrows():
                    # 判斷是否為 AI 跳轉的目標 (高亮顯示)
                    is_target = (row['original_id'] == target_id)
                    row_class = "highlight-record" if is_target else ""
                    target_icon = "✅ [AI精選]" if is_target else ""
                    
                    st.markdown(f"""
                    <div class="record-row {row_class}">
                        <div style="font-weight:bold; color:#ff4b4b; margin-bottom:5px;">{target_icon}</div>
                        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                            <div style="flex: 2; min-width: 300px;">
                                <p><strong style="color:#c53030;">🔴 原因：</strong> {row['原因(異常查找、分析)']}</p>
                                <p><strong style="color:#2f855a;">🟢 對策：</strong> {row['處置、應對']}</p>
                            </div>
                            <div style="flex: 1; min-width: 200px; border-left: 3px solid #eee; padding-left: 15px; font-size: 0.9em; color:#555;">
                                <p><b>驗證：</b> {row['驗證是否排除(驗證作法)']}</p>
                                <p><b>備註：</b> {row['備註(建議事項及補充事項)']}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()