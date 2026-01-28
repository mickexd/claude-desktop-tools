# AI Tools - Supercharge Your Claude Desktop

Welcome! This project provides a set of "brain upgrades" for your Claude Desktop application. These tools help Claude remember things forever and handle massive documents without getting confused or "hallucinating" (making things up).

**Now supporting both Windows and Mac (Apple Silicon M-series chipsets)!**

## Why do I need this?

Normally, AI models like Claude have a "short-term memory" limit. As your conversation gets longer, the AI starts to forget the beginning of the chat (**Context Rot**) or might start making mistakes because it's overwhelmed with too much information (**Hallucinations**).

This project solves those problems using two specialized servers:

### 1. The "Long-Term Memory" (LanceDB Memory Server)

Think of this as a digital notebook that Claude can write in and read from at any time.

- **The Benefit:** Claude can remember details from a conversation you had last week or even last month.
- **How it stops Hallucinations:** Instead of guessing what you said before, Claude searches its database to find the _exact_ facts.
- **Example:** You tell Claude your favorite coding style on Monday. On Friday, you start a new chat, and Claude still knows exactly how you like your code formatted.

### 2. The "Smart Document Reader" (Smart Context Manager)

Think of this as a professional researcher who reads a 500-page book and creates a perfectly organized table of contents and summaries for Claude.

- **The Benefit:** You can give Claude massive files (books, long logs, entire codebases) that would normally be "too big" to process.
- **How it stops Context Rot:** It breaks the document into small, manageable pieces and only shows Claude the parts that are relevant to your current question.
- **Example:** You upload a 200-page manual. Instead of Claude trying to read the whole thing at once (and getting confused), it uses this tool to find the _one specific paragraph_ that answers your question.

---

## Simple Setup Guide

You don't need to be a pro programmer to get this working. Just follow these steps:

### Step 1: Install the "Engine" (Python)

Make sure you have **Python 3.12 or newer** installed on your computer. You can download it from [python.org](https://www.python.org/).

### System Requirements

To run the AI models locally, your computer needs sufficient GPU VRAM (Windows/Linux) or unified RAM (Mac):

- **Windows/Linux (GPU-accelerated):**
  - **Minimum:** 8 GB VRAM (CUDA or ROCm support required)
  - **Recommended:** 16 GB+ VRAM for optimal performance
  - `ministral-3` (8B GGUF): ~6 GB VRAM
  - `nomic-embed-text` embedding model: ~2 GB VRAM
- **Mac (Apple Silicon M-series):**
  - **Minimum:** 16 GB unified RAM
  - **Recommended:** 24 GB+ unified RAM for optimal performance
  - Uses Metal Performance Shaders (MPS) for GPU acceleration on M1/M2/M3/M4 chips

### Step 2: Install the "Brain Power" (Ollama)

For the Smart Context Manager to work, you need **Ollama**. It's a free app that runs AI models on your own computer.

1. Download it from [ollama.com](https://ollama.com/).
2. Open your terminal (Command Prompt/PowerShell on Windows, or Terminal on Mac) and run these two commands:
   ```bash
   ollama pull ministral-3:latest
   ollama pull nomic-embed-text:latest
   ```

### Step 3: Download and Install the Tools

1. Download this project folder to your computer.
2. Open your terminal in that folder and run this command to install the necessary "parts":
   ```bash
   pip install mcp fastmcp lancedb sentence-transformers torch pyarrow pandas requests numpy
   ```

### Step 4: Connect to Claude Desktop

You need to tell Claude where to find these new tools.

1. Open your Claude Desktop configuration file:
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
   - **Mac:** `~/Library/Application Support/Claude/claude_desktop_config.json`

2. Paste the appropriate configuration below (replace the path with the actual path where you saved the files):

**For Windows:**

```json
{
  "mcpServers": {
    "lancedb-memory": {
      "command": "python",
      "args": [
        "C:/Path/To/Your/Folder/Claude Desktop Windows/RAG (LanceDB)/lancedb_memory.py"
      ]
    },
    "smart-context": {
      "command": "python",
      "args": [
        "C:/Path/To/Your/Folder/Claude Desktop Windows/RLM + Comic Text Emebeding/smart_context_mcp.py"
      ]
    }
  }
}
```

**For Mac (Apple Silicon M-series):**

```json
{
  "mcpServers": {
    "lancedb-memory": {
      "command": "python3",
      "args": [
        "/Path/To/Your/Folder/Claude Desktop Mac/RAG (LanceDB)/lancedb_memory.py"
      ]
    },
    "smart-context": {
      "command": "python3",
      "args": [
        "/Path/To/Your/Folder/Claude Desktop Mac/RLM + Comic Text Emebeding/smart_context_mcp.py"
      ]
    }
  }
}
```

---

## How to use it

Once you restart Claude Desktop, you'll see new tools available. You can just talk to Claude normally:

### Memory Management (LanceDB)

- **To store a memory:** "Hey Claude, use store_memory to remember that my project deadline is March 15th."
- **To search memories:** "Use search_memory to find what I told you about deadlines."
- **To get recent memories:** "Use get_recent_memories to show me our last 10 conversation turns."
- **To update a memory:** "Use update_memory to change the memory with ID '1234567_user' to say 'deadline is March 20th'."
- **To delete a specific memory:** "Use delete_memory to remove the memory with ID '1234567_user'."
- **To clear all memories:** "Use clear_all_memories with confirm=true to delete everything." (⚠️ This is irreversible!)
- **To get memory statistics:** "Use get_memory_stats to show me how many memories are stored."

### Document Processing (Smart Context - TRUE RLM)

The Smart Context Manager now features a **True RLM (Recursive Language Model)** implementation based on the research paper ["Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models"](https://arxiv.org/abs/2308.15022). This revolutionary approach allows Claude to programmatically decompose complex tasks using a sandboxed Python REPL environment.

#### What is RLM?

RLM is a paradigm where the language model writes and executes code to solve problems iteratively:

1. **LM generates Python code** in ```repl blocks to examine and process context
2. **Code executes** in a sandboxed environment with access to powerful functions
3. **Results feed back** to the LM, which refines its approach
4. **Iterates** until the LM provides a final answer using `FINAL()` or `FINAL_VAR()`

#### RLM Capabilities

The REPL environment provides these powerful functions:

- **`llm_query(prompt)`** - Recursively call a sub-LLM to handle subtasks
- **`llm_query_batched(prompts)`** - Process multiple queries concurrently (much faster!)
- **`semantic_search(query, top_k=5)`** - Find relevant sections using embeddings
- **`SHOW_VARS()`** - List all variables created in the REPL
- **`print()`** - Debug and view intermediate results

#### How to Use RLM

**Basic RLM Query:**

```
Use rlm_query with:
- query: "What are the main security vulnerabilities in this codebase?"
- context: <your document or data>
- max_iterations: 30 (default)
```

**RLM with File:**

```
Use rlm_query_file with:
- query: "Extract all API endpoints and their authentication requirements"
- file_path: "api_documentation.txt"
- enable_semantic_search: true
```

#### Why RLM is Revolutionary

**Traditional Approach:**

- Fixed chunking strategy (always splits at 50K characters)
- Processes all chunks even if irrelevant
- No ability to adapt to query complexity
- Limited by single-pass processing

**RLM Approach:**

- **Query-aware**: Decides how to chunk based on your specific question
- **Adaptive**: Uses semantic search to find only relevant sections
- **Recursive**: Can call itself to handle massive contexts hierarchically
- **Iterative**: Refines approach based on intermediate results
- **Concurrent**: Processes multiple chunks in parallel for speed

#### Example RLM Strategies

The LM might use different strategies depending on your query:

**For "Summarize this document":**

```python
# Chunk intelligently
chunks = [context[i:i+100000] for i in range(0, len(context), 100000)]

# Summarize in parallel
prompts = [f"Summarize: {chunk}" for chunk in chunks]
summaries = llm_query_batched(prompts)

# Combine results
final = llm_query(f"Combine these summaries: {summaries}")
```

**For "Find security issues":**

```python
# Use semantic search to find relevant sections
relevant = semantic_search("security vulnerability authentication", top_k=10)

# Analyze only relevant sections
issues = []
for section in relevant:
    analysis = llm_query(f"Find security issues in: {section['content']}")
    issues.append(analysis)
```

#### Legacy Tools Still Available

For simpler use cases, the original tools remain available:

- **`chunk_document_smart`** - Split documents into manageable chunks
- **`get_chunk`** - Retrieve specific chunks
- **`process_document_recursive`** - Hierarchical summarization with fixed strategy
- **`query_recursive`** - Semantic search on processed documents
- **`semantic_search_document`** - Find relevant sections using embeddings
- **`summarize_with_local_llm`** - Quick summaries using local Ollama
- **`extract_key_info_local`** - Extract entities, action items, etc.

## Project Structure (For Techies)

### Windows Versions

- **[`Claude Desktop Windows/RAG (LanceDB)/lancedb_memory.py`](<Claude%20Desktop%20Windows/RAG%20(LanceDB)/lancedb_memory.py>):** Uses LanceDB and Sentence-Transformers for vector-based memory.
- **[`Claude Desktop Windows/RLM + Comic Text Emebeding/smart_context_mcp.py`](Claude%20Desktop%20Windows/RLM%20+%20Comic%20Text%20Emebeding/smart_context_mcp.py):** Uses FastMCP and Ollama for hierarchical document processing and RAG.

### Mac Versions (Apple Silicon M-series)

- **[`Claude Desktop Mac/RAG (LanceDB)/lancedb_memory.py`](<Claude%20Desktop%20Mac/RAG%20(LanceDB)/lancedb_memory.py>):** Mac-optimized version using LanceDB and Sentence-Transformers for vector-based memory.
- **[`Claude Desktop Mac/RLM + Comic Text Emebeding/smart_context_mcp.py`](Claude%20Desktop%20Mac/RLM%20+%20Comic%20Text%20Emebeding/smart_context_mcp.py):** Mac-optimized version using FastMCP and Ollama for hierarchical document processing and RAG.

---

## Memory Management Features

The LanceDB Memory Server provides powerful tools to manage Claude's long-term memory:

### Available Tools

1. **`store_memory`** - Store new conversation turns
   - Stores text with semantic embeddings for intelligent search
   - Supports role tagging (user/assistant) and conversation IDs
   - Optional metadata for additional context

2. **`search_memory`** - Semantic search across all memories
   - Find relevant memories using natural language queries
   - Filter by role (user/assistant/both)
   - Adjustable similarity threshold and result count
   - Returns ranked results with similarity scores

3. **`get_recent_memories`** - Retrieve recent conversation history
   - Get the most recent N memories
   - Optional filtering by conversation ID
   - Sorted by timestamp (newest first)

4. **`update_memory`** - Modify existing memories
   - Update text content and/or metadata
   - Automatically regenerates embeddings when text changes
   - Preserves original timestamp and role
   - Requires memory ID (visible in search results)

5. **`delete_memory`** - Remove specific memories
   - Delete individual memories by ID
   - Does not affect other memories
   - Useful for removing outdated or incorrect information

6. **`clear_all_memories`** - Clear entire memory database
   - ⚠️ **WARNING:** This is irreversible!
   - Requires explicit confirmation (confirm=true)
   - Use only when you want to start fresh

7. **`get_memory_stats`** - View memory statistics
   - Total memory count
   - Breakdown by role (user/assistant)
   - Conversation statistics
   - Storage usage information

### Memory IDs

Each memory has a unique ID in the format `timestamp_role` (e.g., `1738195234.567_user`). You can find memory IDs in:

- Search results from `search_memory`
- Recent memories from `get_recent_memories`
- Memory statistics from `get_memory_stats`

Use these IDs with `update_memory` or `delete_memory` to manage specific memories.

---

## Platform Notes

### Windows

- Use `python` in the configuration file
- Paths use backslashes (`\`) or forward slashes (`/`)
- Configuration file location: `%APPDATA%\Claude\claude_desktop_config.json`

### Mac (Apple Silicon M1/M2/M3/M4)

- Use `python3` in the configuration file
- Paths use forward slashes (`/`)
- Configuration file location: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Both Intel and Apple Silicon Macs are supported
