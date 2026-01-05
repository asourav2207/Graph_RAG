"""
Database module for persistent storage of query history and metadata.
Uses SQLite for lightweight, file-based persistence.
"""
import sqlite3
import json
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "rag_project/graphrag_data.db"

def init_db():
    """Initialize the database with required tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Query history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                method TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Metadata table for app settings/state
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()

@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# ============== Query History ==============

def save_query(query: str, method: str, response: str) -> int:
    """Save a query to the history. Returns the query ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO query_history (timestamp, query, method, response) 
               VALUES (?, ?, ?, ?)""",
            (datetime.now().strftime("%Y-%m-%d %H:%M"), query, method, response)
        )
        conn.commit()
        return cursor.lastrowid

def get_all_queries():
    """Get all queries from history, newest first."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, timestamp, query, method, response 
               FROM query_history 
               ORDER BY id DESC"""
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_query_by_id(query_id: int):
    """Get a specific query by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM query_history WHERE id = ?", (query_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

def clear_history():
    """Clear all query history."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM query_history")
        conn.commit()

def get_query_count():
    """Get total number of queries."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM query_history")
        return cursor.fetchone()[0]

# ============== Metadata ==============

def set_metadata(key: str, value: str):
    """Save a metadata key-value pair."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO metadata (key, value, updated_at) 
               VALUES (?, ?, ?)""",
            (key, value, datetime.now().isoformat())
        )
        conn.commit()

def get_metadata(key: str, default=None):
    """Get a metadata value by key."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else default

# Initialize database on module import
init_db()
