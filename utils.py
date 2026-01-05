"""
GraphRAG Utilities - Configured for Ollama Local LLMs
Tested and verified with llama3.2 and nomic-embed-text
"""
import os
import sys
import shutil
import requests
import pandas as pd

# ============== Configuration ==============
RAG_DIR = "rag_project"
INPUT_DIR = os.path.join(RAG_DIR, "input")

# Dynamic command detection for Cloud/Linux compatibility
GRAPHRAG_CMD = shutil.which("graphrag")
if not GRAPHRAG_CMD:
    # Fallback to local user bin if not in PATH
    GRAPHRAG_CMD = os.path.join(os.path.dirname(sys.executable), "graphrag")

# ============== Initialization ==============
def init_graphrag():
    """Initializes the GraphRAG project structure."""
    if not os.path.exists(RAG_DIR):
        os.makedirs(RAG_DIR)
    
    if not GRAPHRAG_CMD:
        return False, "GraphRAG executable not found. Please install: pip install graphrag"

    try:
        subprocess.run([GRAPHRAG_CMD, "init", "--root", RAG_DIR], check=True)
        return True, "Initialized GraphRAG project."
    except Exception as e:
        return False, f"Failed to init: {e}"

def check_ollama_status(url="http://localhost:11434"):
    """Quickly checks if Ollama is reachable."""
    try:
        response = requests.get(url, timeout=1.0)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def update_settings(model_name="gpt-oss:20b-cloud"):
    """Updates settings.yaml to use Ollama with optimized settings for local LLMs."""
    settings_path = os.path.join(RAG_DIR, "settings.yaml")
    
    if not os.path.exists(settings_path):
        return False, "settings.yaml not found. Run Init first."
    
    try:
        with open(settings_path, 'r') as f:
            params = yaml.safe_load(f)
        
        # ===== CHAT MODEL CONFIG =====
        if 'models' in params and 'default_chat_model' in params['models']:
            chat = params['models']['default_chat_model']
            chat['model_provider'] = 'openai'  # Ollama uses OpenAI-compatible API
            chat['auth_type'] = 'api_key'
            chat['api_key'] = 'ollama'  # Dummy key, not used
            chat['api_base'] = 'http://localhost:11434/v1'
            chat['model'] = model_name
            chat['concurrent_requests'] = 1  # Sequential for local LLMs
            chat['max_retries'] = 10
            chat['request_timeout'] = 3600.0  # 1 hour timeout for cloud LLMs
            
        # ===== EMBEDDING MODEL CONFIG =====
        if 'models' in params and 'default_embedding_model' in params['models']:
            embed = params['models']['default_embedding_model']
            embed['model_provider'] = 'openai'
            embed['auth_type'] = 'api_key'
            embed['api_key'] = 'ollama'
            embed['api_base'] = 'http://localhost:11434/v1'
            embed['model'] = 'nomic-embed-text'  # Best embedding model for Ollama
            embed['request_timeout'] = 600.0  # 10 minutes for embeddings
            
        # PERFORMANCE OPTIMIZATIONS
        # Disable Gleanings for Speed
        if 'extract_graph' in params:
            params['extract_graph']['max_gleanings'] = 0
            
        # Increase chunk size to reduce number of LLM calls (default: 1200)
        if 'chunks' in params:
            params['chunks']['size'] = 2000  # Larger chunks = fewer LLM calls
            params['chunks']['overlap'] = 200  # Disable second pass for speed
            
        with open(settings_path, 'w') as f:
            yaml.dump(params, f)
            
        return True, "Updated settings.yaml for Ollama."
        
    except Exception as e:
        return False, f"Failed to update settings: {e}"

# ============== Indexing ==============
def run_indexing():
    """Runs the indexing process with unbuffered output for real-time logs."""
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        
        process = subprocess.Popen(
            [GRAPHRAG_CMD, "index", "--root", RAG_DIR, "--verbose"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge for real-time logging
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )
        return process
    except Exception as e:
        return None

# ============== Querying ==============
def run_query(query, method="local"):
    """Runs a query against the graph. Use 'local' for specific facts, 'global' for summaries."""
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        result = subprocess.run(
            [GRAPHRAG_CMD, "query", "--root", RAG_DIR, "--method", method, "--query", query],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for slow LLMs
            env=env
        )
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Query timed out. Local LLMs can be slow - try a simpler question or wait longer."
    except Exception as e:
        return f"Exception: {e}"

# ============== Data Loading ==============
def get_latest_output_dir():
    """Gets the output directory from indexing."""
    output_base = os.path.join(RAG_DIR, "output")
    if not os.path.exists(output_base):
        return None
    
    # Check for parquet files directly in output (newer GraphRAG structure)
    if os.path.exists(os.path.join(output_base, "entities.parquet")):
        return output_base
    
    # Fallback: Look for timestamped subdirs
    subdirs = [os.path.join(output_base, d) for d in os.listdir(output_base) 
               if os.path.isdir(os.path.join(output_base, d))]
    
    if not subdirs:
        return None
        
    latest_subdir = max(subdirs, key=os.path.getmtime)
    artifacts_path = os.path.join(latest_subdir, "artifacts")
    
    return artifacts_path if os.path.exists(artifacts_path) else latest_subdir

def load_parquet_files():
    """Loads entities, relationships, and community reports."""
    artifacts_dir = get_latest_output_dir()
    if not artifacts_dir:
        return None
    
    data = {}
    
    # File patterns to load (handles both old and new naming)
    file_patterns = {
        "entities": ["entities.parquet", "create_final_entities.parquet"],
        "relationships": ["relationships.parquet", "create_final_relationships.parquet"],
        "reports": ["community_reports.parquet", "create_final_community_reports.parquet"]
    }
    
    for key, patterns in file_patterns.items():
        for pattern in patterns:
            path = os.path.join(artifacts_dir, pattern)
            if os.path.exists(path):
                try:
                    data[key] = pd.read_parquet(path)
                    break
                except Exception:
                    pass
        if key not in data:
            data[key] = pd.DataFrame()
    
    return data

# ============== File Management ==============
def save_uploaded_file(uploaded_file):
    """Saves uploaded file to input directory, converting PDF to text if needed."""
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        
    file_path = os.path.join(INPUT_DIR, uploaded_file.name)
    
    if uploaded_file.name.lower().endswith('.pdf'):
        import pdfplumber
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                text = "".join(page.extract_text() or "" for page in pdf.pages)
            
            txt_filename = os.path.splitext(uploaded_file.name)[0] + ".txt"
            txt_path = os.path.join(INPUT_DIR, txt_filename)
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            return True, f"Saved & Converted PDF to {txt_filename}"
        except Exception as e:
            return False, str(e)
    else:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True, f"Saved {uploaded_file.name}"

def clear_project():
    """Clears cache and output for a fresh run."""
    import shutil
    cache_dir = os.path.join(RAG_DIR, "cache")
    output_dir = os.path.join(RAG_DIR, "output")
    
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    return True, "Cleared cache and output."
