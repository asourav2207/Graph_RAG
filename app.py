"""
GraphRAG Streamlit Application - Ollama Edition
With persistent SQLite-based query history
"""
import streamlit as st
import utils
import database as db
import os
from datetime import datetime

st.set_page_config(layout="wide", page_title="GraphRAG - Ollama", page_icon="🕸️")

# ============== Initialize Session State ==============
if 'logs' not in st.session_state:
    st.session_state.logs = ""
if 'selected_report' not in st.session_state:
    st.session_state.selected_report = None

# Initialize Database Explicitly
db.init_db()

# Check Dependencies
rag_found = utils.GRAPHRAG_CMD is not None
# Note: Check status of *configured* URL, not just localhost default
ollama_ok = utils.check_ollama_status() # default check

# ============== Styling ==============
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E1E1E;
        border-radius: 4px;
        padding: 8px 16px;
    }
    .history-item {
        background-color: #1E1E1E;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 3px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ============== Sidebar ==============
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # LLM Settings
    with st.expander("🤖 LLM Settings", expanded=True):
        model_name = st.text_input("Model Name", value="gpt-oss:20b-cloud", help="e.g., llama3, gpt-4o")
        api_base = st.text_input("API Base URL", value="http://localhost:11434/v1", help="Ollama: http://localhost:11434/v1, OpenAI: https://api.openai.com/v1")
        api_key = st.text_input("API Key", value="ollama", type="password", help="Use 'ollama' for local, or your actual key for providers")
    
    st.divider()
    
    st.subheader("1️⃣ Initialize")
    if st.button("Initialize Project", use_container_width=True):
        with st.spinner("Initializing..."):
            success, msg = utils.init_graphrag()
            if success:
                st.success(msg)
                s, m = utils.update_settings(model_name, api_base, api_key)
                st.success("Configured for Ollama!" if s else m)
            else:
                if "already initialized" in msg.lower():
                    st.info("Project already initialized.")
                    s, m = utils.update_settings(model_name, api_base, api_key)
                    st.success("Settings refreshed!" if s else m)
                else:
                    st.error(msg)
    
    # Validation Status
    if not rag_found:
        st.error("❌ GraphRAG not found! `pip install graphrag`")
    
    if not ollama_ok:
        st.warning("⚠️ Ollama Local not reachable")
        st.caption("If on Cloud, use OpenAI or a public URL.")
    else:
        st.success("✅ Ollama Local Online")
                
    st.subheader("2️⃣ Configure")
    if st.button("Update Settings", use_container_width=True):
        success, msg = utils.update_settings(model_name, api_base, api_key)
        st.success(msg) if success else st.error(msg)
    
    if st.button("🗑️ Clear Cache/Output", use_container_width=True):
        success, msg = utils.clear_project()
        st.success(msg) if success else st.error(msg)

    st.divider()
    st.divider()
    if ollama_ok:
        st.info("💡 Connected to local Ollama")
    else:
        st.warning("💡 You need a working LLM endpoint")
    
    # Query History Summary in Sidebar
    st.divider()
    st.subheader("📜 Query History")
    query_count = db.get_query_count()
    st.caption(f"{query_count} queries stored in database")
    if st.button("Clear History", use_container_width=True):
        db.clear_history()
        st.rerun()

# ============== Main Interface ==============
st.title("🕸️ GraphRAG with Ollama")
st.caption("Microsoft GraphRAG • Local LLMs • Persistent query storage")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📂 Upload & Index", "📊 Graph Data", "💬 Query", "📜 History", "📋 Logs"])

# ============== Tab 1: Upload & Index ==============
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload text or PDF files", 
            type=['txt', 'pdf'], 
            accept_multiple_files=True,
            help="Files will be converted to text and indexed"
        )
        
        if uploaded_files:
            for f in uploaded_files:
                success, msg = utils.save_uploaded_file(f)
                st.toast(msg, icon="✅" if success else "❌")
                
        input_dir = utils.INPUT_DIR
        if os.path.exists(input_dir):
            files = os.listdir(input_dir)
            if files:
                st.markdown("**Current Input Files:**")
                for f in files:
                    st.text(f"📄 {f}")
    
    with col2:
        st.header("Build Knowledge Graph")
        st.warning("⚠️ Indexing uses the LLM heavily and can take time.")
        
        if st.button("🚀 Start Indexing", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Starting...")
            
            process = utils.run_indexing()
            if process:
                for line in iter(process.stdout.readline, ''):
                    st.session_state.logs += line
                    
                    line_lower = line.lower()
                    if "load_input" in line_lower:
                        progress_bar.progress(10, "Loading documents...")
                    elif "text_units" in line_lower:
                        progress_bar.progress(20, "Creating text units...")
                    elif "extract_graph" in line_lower and "starting" in line_lower:
                        progress_bar.progress(30, "Extracting entities (slow)...")
                    elif "extract graph progress" in line_lower:
                        progress_bar.progress(50, "Extracting graph...")
                    elif "finalize_graph" in line_lower:
                        progress_bar.progress(60, "Finalizing graph...")
                    elif "communities" in line_lower:
                        progress_bar.progress(70, "Creating communities...")
                    elif "community_reports" in line_lower:
                        progress_bar.progress(80, "Generating reports...")
                    elif "embeddings" in line_lower:
                        progress_bar.progress(90, "Generating embeddings...")
                    elif "pipeline complete" in line_lower:
                        progress_bar.progress(100, "Complete!")
                
                process.stdout.close()
                return_code = process.wait()
                
                if return_code == 0:
                    progress_bar.progress(100, "✅ Indexing Complete!")
                    st.success("Knowledge graph built successfully!")
                    st.balloons()
                else:
                    st.error(f"Indexing failed (exit code {return_code}). Check Logs tab.")
            else:
                st.error("Failed to start indexing process.")

# ============== Tab 2: Graph Data ==============
with tab2:
    st.header("Knowledge Graph Artifacts")
    
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    data = utils.load_parquet_files()
    if data and any(not df.empty for df in data.values()):
        st.success(f"Loaded graph data from: {utils.get_latest_output_dir()}")
        
        # Entities
        with st.expander("🟦 Entities", expanded=True):
            if 'entities' in data and not data['entities'].empty:
                df = data['entities']
                display_cols = [c for c in ['title', 'type', 'description'] if c in df.columns]
                if display_cols:
                    st.dataframe(df[display_cols], use_container_width=True, height=300)
                else:
                    st.dataframe(df, use_container_width=True, height=300)
                st.caption(f"Total: {len(df)} entities")
            else:
                st.info("No entities found.")

        # Relationships
        with st.expander("🔗 Relationships"):
            if 'relationships' in data and not data['relationships'].empty:
                df = data['relationships']
                display_cols = [c for c in ['source', 'target', 'description', 'weight'] if c in df.columns]
                if display_cols:
                    st.dataframe(df[display_cols], use_container_width=True, height=300)
                else:
                    st.dataframe(df, use_container_width=True, height=300)
                st.caption(f"Total: {len(df)} relationships")
            else:
                st.info("No relationships found.")
                
        # Community Reports - Clickable
        with st.expander("🏘️ Community Reports", expanded=True):
            if 'reports' in data and not data['reports'].empty:
                df = data['reports']
                
                for idx, row in df.iterrows():
                    title = row.get('title', f'Community {idx}')
                    summary = str(row.get('summary', 'No summary'))[:150] + "..."
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button(f"📖 View", key=f"report_{idx}"):
                            st.session_state.selected_report = idx
                    with col2:
                        st.markdown(f"**{title}**")
                        st.caption(summary)
                
                if st.session_state.selected_report is not None:
                    idx = st.session_state.selected_report
                    if idx < len(df):
                        row = df.iloc[idx]
                        st.divider()
                        st.markdown("### 📋 Full Report")
                        st.markdown(f"**Title:** {row.get('title', 'N/A')}")
                        st.info(row.get('summary', 'No summary'))
                        
                        if 'full_content' in row and row['full_content']:
                            with st.expander("📄 Full Content"):
                                st.text(row['full_content'])
                        
                        if st.button("✖️ Close Report"):
                            st.session_state.selected_report = None
                            st.rerun()
                
                st.caption(f"Total: {len(df)} communities")
            else:
                st.info("No community reports found.")
    else:
        st.warning("No graph data available. Please run indexing first.")

# ============== Tab 3: Query ==============
with tab3:
    st.header("Query the Knowledge Graph")
    
    method = st.radio(
        "Search Method", 
        ["local", "global"], 
        index=0,
        horizontal=True,
        help="Local: Best for specific facts. Global: Best for summaries."
    )
    
    query = st.text_area("Enter your question:", placeholder="e.g., What are IIMs and WAT PI?")
    
    if st.button("🔍 Ask", type="primary"):
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner(f"Running {method} search (may take a few minutes)..."):
                result = utils.run_query(query, method=method)
                
                # Save to database (persistent!)
                db.save_query(query, method, result)
                
                st.markdown("### Answer")
                st.markdown(result)
                st.success("✅ Saved to query history!")

# ============== Tab 4: Query History (Persistent) ==============
with tab4:
    st.header("📜 Query History (Persistent)")
    st.caption("Stored in SQLite database - survives page refresh")
    
    queries = db.get_all_queries()
    
    if queries:
        for item in queries:
            with st.container():
                st.markdown(f"""
                <div class="history-item">
                    <small>🕐 {item['timestamp']} | ID: {item['id']} | Method: {item['method']}</small>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander(f"**Q:** {item['query'][:80]}{'...' if len(item['query']) > 80 else ''}", expanded=False):
                    st.markdown("**Question:**")
                    st.info(item['query'])
                    st.markdown("**Answer:**")
                    st.success(item['response'])
                    
                    if st.button(f"🔄 Re-run query", key=f"rerun_{item['id']}"):
                        with st.spinner("Re-running..."):
                            result = utils.run_query(item['query'], method=item['method'])
                            db.save_query(item['query'], item['method'], result)
                            st.rerun()
                
                st.divider()
        
        # Export
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export as Text"):
                export_text = ""
                for item in queries:
                    export_text += f"ID: {item['id']}\n"
                    export_text += f"Time: {item['timestamp']}\n"
                    export_text += f"Method: {item['method']}\n"
                    export_text += f"Query: {item['query']}\n"
                    export_text += f"Response: {item['response']}\n"
                    export_text += "-" * 50 + "\n\n"
                st.download_button(
                    "Download",
                    export_text,
                    file_name="query_history.txt",
                    mime="text/plain"
                )
    else:
        st.info("No queries yet. Go to the Query tab to ask questions!")

# ============== Tab 5: Logs ==============
with tab5:
    st.header("Process Logs")
    if st.session_state.logs:
        st.text_area("Indexing Logs", st.session_state.logs, height=500)
        if st.button("Clear Logs"):
            st.session_state.logs = ""
            st.rerun()
    else:
        st.info("No logs available. Run indexing to see logs here.")
