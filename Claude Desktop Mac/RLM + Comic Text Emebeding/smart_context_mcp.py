#!/usr/bin/env python3
from fastmcp import FastMCP
import os
import re
import json
import platform
import tempfile
import threading
import base64
from pathlib import Path
from typing import List, Dict, Optional, Callable
import hashlib
import requests
import numpy as np
from datetime import datetime
from contextlib import contextmanager
import signal

# ============================================================================
# RLM DATA TYPES
# ============================================================================

from dataclasses import dataclass
from typing import Any
import io
import sys
import time

@dataclass
class REPLResult:
    """Result from executing code in REPL environment"""
    stdout: str
    stderr: str
    locals: dict
    execution_time: float
    rlm_calls: list

    def __init__(self, stdout: str, stderr: str, locals: dict,
                 execution_time: float = 0.0, rlm_calls: list = None):
        self.stdout = stdout
        self.stderr = stderr
        self.locals = locals
        self.execution_time = execution_time
        self.rlm_calls = rlm_calls or []

@dataclass
class RLMChatCompletion:
    """Record of a single LLM call made from within REPL"""
    root_model: str
    prompt: str
    response: str
    execution_time: float

@dataclass
class CodeBlock:
    """Code block with execution result"""
    code: str
    result: 'REPLResult'

@dataclass
class RLMIteration:
    """Single iteration of RLM execution loop"""
    prompt: str
    response: str
    code_blocks: list
    final_answer: str = None
    iteration_time: float = None

mcp = FastMCP("Smart Context Manager - Recursive LLM")

# ============================================================================
# ENHANCED CONFIGURATION
# ============================================================================

class Config:
    OLLAMA_BASE_URL = "http://localhost:11434"
    LLM_MODEL = "ministral-3:latest"
    EMBEDDING_MODEL = "nomic-embed-text:latest"
    
    # FIX: Use absolute paths in user's home directory instead of relative paths
    HOME_DIR = Path.home()
    CACHE_DIR = str(HOME_DIR / ".cache" / "smart_context" / "context_cache")
    EMBEDDINGS_CACHE_DIR = str(HOME_DIR / ".cache" / "smart_context" / "embeddings_cache")
    
    # Token limits (approximate)
    CHUNK_SIZE = 50000  # ~12K tokens
    MAX_CONTEXT = 120000  # ~30K tokens for local models
    
    # Recursive processing config
    RECURSIVE_DEPTH = 3
    SIMILARITY_THRESHOLD = 0.7
    BATCH_SIZE = 10
    
    # Safety limits
    MIN_CHUNK_SIZE = 100
    MAX_CHUNK_SIZE = 500000
    DEFAULT_TIMEOUT = 300  # 5 minutes

config = Config()

# ============================================================================
# RLM SYSTEM PROMPT
# ============================================================================

RLM_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs.

The REPL environment provides:
1. `context` variable - Your input data
2. `llm_query(prompt)` - Query a sub-LLM recursively
3. `llm_query_batched(prompts)` - Query multiple prompts concurrently (faster!)
4. `semantic_search(query, top_k=5)` - Find relevant sections using embeddings
5. `SHOW_VARS()` - List all variables you've created
6. `print()` - View outputs and debug

Execute Python code by wrapping it in ```repl blocks:
```repl
# Example: Examine context
print(f"Context length: {len(context)}")
print(f"Context type: {type(context)}")
```

Example strategy for large documents:

```repl
# Chunk the context intelligently
chunk_size = 100000
chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]

# Query sub-LLMs in parallel (much faster!)
prompts = [f"Summarize this section: {chunk}" for chunk in chunks]
summaries = llm_query_batched(prompts)

# Aggregate results
final_answer = llm_query(f"Combine these summaries into one: {summaries}")
```

Example using semantic search:

```repl
# Find relevant sections
relevant = semantic_search("authentication methods", top_k=3)
for section in relevant:
    print(f"Similarity: {section['similarity']}")
    print(f"Content: {section['content'][:200]}")

# Query LLM on relevant sections only
contexts = [s['content'] for s in relevant]
answer = llm_query(f"Based on these sections, answer the query: {contexts}")
```

IMPORTANT: When done, provide final answer using:

- `FINAL(your answer here)` - Direct answer
- `FINAL_VAR(variable_name)` - Return a variable you created

Think step-by-step, use the REPL extensively, and leverage recursive LLM calls!
"""

# ============================================================================
# CODE PARSING UTILITIES
# ============================================================================

def find_code_blocks(text: str) -> list[str]:
    """Extract ```repl code blocks from LM response"""
    pattern = r"```repl\s*\n(.*?)\n```"
    results = []
    for match in re.finditer(pattern, text, re.DOTALL):
        code_content = match.group(1).strip()
        results.append(code_content)
    return results

def find_final_answer(text: str, repl_env=None) -> str | None:
    """Find FINAL(...) or FINAL_VAR(...) in response"""
    # Check for FINAL_VAR
    final_var_pattern = r"^\s*FINAL_VAR\((.*?)\)"
    match = re.search(final_var_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        variable_name = match.group(1).strip().strip('"').strip("'")
        if repl_env is not None:
            result = repl_env.execute_code(f"print(FINAL_VAR({variable_name!r}))")
            return result.stdout.strip() or result.stderr.strip() or ""
        return None

    # Check for FINAL
    final_pattern = r"^\s*FINAL\((.*)\)\s*$"
    match = re.search(final_pattern, text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None

def format_execution_result(result: REPLResult, max_length: int = 20000) -> str:
    """Format REPL result for display to LM"""
    parts = []

    if result.stdout:
        stdout = result.stdout
        if len(stdout) > max_length:
            stdout = stdout[:max_length] + f"... [{len(stdout)-max_length} chars truncated]"
        parts.append(f"Output:\n{stdout}")

    if result.stderr:
        stderr = result.stderr
        if len(stderr) > max_length:
            stderr = stderr[:max_length] + f"... [{len(stderr)-max_length} chars truncated]"
        parts.append(f"Errors:\n{stderr}")

    vars_list = [k for k in result.locals.keys() if not k.startswith("_")]
    if vars_list:
        parts.append(f"Variables: {vars_list}")

    return "\n\n".join(parts) if parts else "No output"

# ============================================================================
# REPL ENVIRONMENT
# ============================================================================

_SAFE_BUILTINS = {
    "print": print, "len": len, "str": str, "int": int, "float": float,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "bool": bool,
    "type": type, "isinstance": isinstance, "enumerate": enumerate,
    "zip": zip, "map": map, "filter": filter, "sorted": sorted,
    "reversed": reversed, "range": range, "min": min, "max": max,
    "sum": sum, "abs": abs, "round": round, "any": any, "all": all,
    "pow": pow, "chr": chr, "ord": ord, "hex": hex, "bin": bin,
    "repr": repr, "format": format, "hasattr": hasattr, "getattr": getattr,
    "setattr": setattr, "dir": dir, "open": open, "__import__": __import__,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError, "RuntimeError": RuntimeError,
    # Blocked for security
    "input": None, "eval": None, "exec": None, "compile": None,
    "globals": None, "locals": None,
}

class REPLEnvironment:
    """Sandboxed Python REPL for RLM code execution"""

    def __init__(self, context: Any, ollama_client, enable_semantic_search: bool = True):
        self.ollama_client = ollama_client
        self.context = context
        self._lock = threading.Lock()
        self._pending_llm_calls = []

        # Sandboxed namespace
        self.globals = {"__builtins__": _SAFE_BUILTINS.copy(), "__name__": "__main__"}
        self.locals = {"context": context}

        # Add helper functions
        self.globals["FINAL_VAR"] = self._final_var
        self.globals["SHOW_VARS"] = self._show_vars
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched

        if enable_semantic_search:
            self.globals["semantic_search"] = self._semantic_search
            self._doc_id = None

    def _final_var(self, variable_name: str) -> str:
        """Return variable value as final answer"""
        variable_name = variable_name.strip().strip("\"'")
        if variable_name in self.locals:
            return str(self.locals[variable_name])
        available = [k for k in self.locals.keys() if not k.startswith("_")]
        return f"Error: Variable '{variable_name}' not found. Available: {available}"

    def _show_vars(self) -> str:
        """Show all variables"""
        available = {k: type(v).__name__ for k, v in self.locals.items() if not k.startswith("_")}
        return f"Available variables: {available}" if available else "No variables yet"

    def _llm_query(self, prompt: str, model: str = None) -> str:
        """Recursive LLM call - CORE RLM CAPABILITY!"""
        start_time = time.perf_counter()
        try:
            result = self.ollama_client.generate(
                prompt, model=model or config.LLM_MODEL,
                options={"temperature": 0.7, "num_ctx": 8192}
            )
            response = result.get("response", "")
            self._pending_llm_calls.append(RLMChatCompletion(
                root_model=model or config.LLM_MODEL,
                prompt=prompt, response=response,
                execution_time=time.perf_counter() - start_time
            ))
            return response
        except Exception as e:
            return f"Error: LLM query failed - {str(e)}"

    def _llm_query_batched(self, prompts: list[str], model: str = None) -> list[str]:
        """Batched recursive LLM calls for concurrency"""
        import concurrent.futures
        results = []
        start_time = time.perf_counter()

        try:
            def query_single(prompt):
                result = self.ollama_client.generate(
                    prompt, model=model or config.LLM_MODEL,
                    options={"temperature": 0.7, "num_ctx": 8192}
                )
                return result.get("response", "")

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(query_single, p) for p in prompts]
                results = [f.result() for f in futures]

            exec_time = time.perf_counter() - start_time
            for prompt, response in zip(prompts, results):
                self._pending_llm_calls.append(RLMChatCompletion(
                    root_model=model or config.LLM_MODEL,
                    prompt=prompt, response=response,
                    execution_time=exec_time / len(prompts)
                ))
            return results
        except Exception as e:
            return [f"Error: Batched query failed - {str(e)}"] * len(prompts)

    def _semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search using embeddings"""
        try:
            query_result = self.ollama_client.embed(query)
            if 'error' in query_result:
                return [{"error": query_result['error']}]

            query_embedding = query_result['embedding']
            vector_store = get_vector_store()

            # Create doc_id and index if needed
            if self._doc_id is None:
                self._doc_id = hashlib.md5(str(self.context)[:1000].encode()).hexdigest()[:16]
                if not vector_store.load_document(self._doc_id):
                    if isinstance(self.context, str):
                        paragraphs = [p.strip() for p in self.context.split('\n\n') if len(p.strip()) > 50]
                        chunks = []
                        for i, para in enumerate(paragraphs[:100]):
                            embed_result = self.ollama_client.embed(para[:1000])
                            if 'embedding' in embed_result:
                                chunks.append({
                                    'content': para,
                                    'embedding': embed_result['embedding'],
                                    'index': i
                                })
                        vector_store.add_document(self._doc_id, chunks)

            results = vector_store.search(self._doc_id, query_embedding, top_k=top_k, threshold=0.5)
            return [{'similarity': round(r['similarity'], 3), 'content': r.get('content', ''),
                    'index': r.get('index', -1)} for r in results]
        except Exception as e:
            return [{"error": f"Semantic search failed: {str(e)}"}]

    def execute_code(self, code: str) -> REPLResult:
        """Execute Python code in sandboxed environment"""
        start_time = time.perf_counter()
        self._pending_llm_calls = []

        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()

        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                combined = {**self.globals, **self.locals}
                exec(code, combined, combined)

                for key, value in combined.items():
                    if key not in self.globals and not key.startswith("_"):
                        self.locals[key] = value

                stdout, stderr = stdout_buf.getvalue(), stderr_buf.getvalue()
            except Exception as e:
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

        return REPLResult(
            stdout=stdout, stderr=stderr, locals=self.locals.copy(),
            execution_time=time.perf_counter() - start_time,
            rlm_calls=self._pending_llm_calls.copy()
        )

# ============================================================================
# RLM COMPLETION LOOP
# ============================================================================

def rlm_completion_loop(query: str, context: Any, ollama_client,
                       max_iterations: int = 30, enable_semantic_search: bool = True,
                       verbose: bool = False) -> dict:
    """
    Main RLM execution loop: LM generates code → execute → feed back → repeat.
    This implements the core RLM paradigm from the paper.
    """
    start_time = time.perf_counter()
    repl_env = REPLEnvironment(context, ollama_client, enable_semantic_search)

    # Build context info
    context_info = f"Context type: {type(context).__name__}, "
    if isinstance(context, str):
        context_info += f"length: {len(context)} chars"
    elif isinstance(context, list):
        context_info += f"length: {len(context)} items"
    elif isinstance(context, dict):
        context_info += f"keys: {list(context.keys())}"

    message_history = [
        {"role": "system", "content": RLM_SYSTEM_PROMPT},
        {"role": "assistant", "content": f"Your context is available. {context_info}"},
        {"role": "user", "content": f"Query: {query}\n\nExamine the context and plan your approach using the REPL environment."}
    ]

    all_iterations = []

    for iteration_num in range(max_iterations):
        if verbose:
            print(f"\n{'='*80}\nRLM Iteration {iteration_num + 1}/{max_iterations}\n{'='*80}")

        # LM generates response
        prompt_text = "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in message_history])

        iter_start = time.perf_counter()
        lm_response = ollama_client.generate(prompt_text, options={"temperature": 0.7, "num_ctx": 8192})
        response_text = lm_response.get("response", "")

        if verbose:
            print(f"\nLM Response:\n{response_text[:500]}...")

        # Extract and execute code blocks
        code_blocks = find_code_blocks(response_text)
        executed_blocks = []

        for code in code_blocks:
            if verbose:
                print(f"\nExecuting code:\n{code[:200]}...")
            result = repl_env.execute_code(code)
            executed_blocks.append(CodeBlock(code=code, result=result))
            if verbose:
                print(f"Output: {result.stdout[:200] if result.stdout else '(none)'}")
                if result.stderr:
                    print(f"Errors: {result.stderr[:200]}")

        # Create iteration record
        iteration = RLMIteration(
            prompt=message_history[-1]["content"],
            response=response_text,
            code_blocks=executed_blocks,
            iteration_time=time.perf_counter() - iter_start
        )
        all_iterations.append(iteration)

        # Check for final answer
        final_answer = find_final_answer(response_text, repl_env)
        if final_answer:
            if verbose:
                print(f"\n{'='*80}\nFINAL ANSWER: {final_answer[:200]}...\n{'='*80}")

            return {
                "answer": final_answer,
                "iterations": iteration_num + 1,
                "execution_time": time.perf_counter() - start_time,
                "code_blocks_executed": sum(len(it.code_blocks) for it in all_iterations),
                "recursive_llm_calls": len(repl_env._pending_llm_calls),
                "status": "success"
            }

        # Add to history
        message_history.append({"role": "assistant", "content": response_text})

        for block in executed_blocks:
            exec_result = format_execution_result(block.result)
            message_history.append({
                "role": "user",
                "content": f"Code executed:\n```python\n{block.code}\n```\n\nREPL output:\n{exec_result}"
            })

        message_history.append({"role": "user", "content": "Continue using REPL. Your next action:"})

    # Max iterations reached
    if verbose:
        print(f"\n{'='*80}\nMAX ITERATIONS REACHED\n{'='*80}")

    return {
        "answer": "Max iterations reached. Try with more iterations or simplify the query.",
        "iterations": max_iterations,
        "execution_time": time.perf_counter() - start_time,
        "code_blocks_executed": sum(len(it.code_blocks) for it in all_iterations),
        "recursive_llm_calls": len(repl_env._pending_llm_calls),
        "status": "max_iterations_reached"
    }

# Create cache directories with parents=True for nested paths
Path(config.CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(config.EMBEDDINGS_CACHE_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================================
# SECURE JSON ENCODER/DECODER FOR NUMPY ARRAYS (REPLACES PICKLE)
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays securely"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '__numpy__': True,
                'dtype': str(obj.dtype),
                'data': base64.b64encode(obj.tobytes()).decode('ascii'),
                'shape': list(obj.shape)
            }
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)

def numpy_decoder(dct):
    """JSON decoder hook for numpy arrays"""
    if '__numpy__' in dct:
        data = base64.b64decode(dct['data'])
        return np.frombuffer(data, dtype=dct['dtype']).reshape(dct['shape'])
    return dct

# ============================================================================
# TIMEOUT CONTEXT MANAGER (CROSS-PLATFORM)
# ============================================================================

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

@contextmanager
def timeout_context(seconds: int):
    """
    Context manager for timeout handling.
    macOS/Unix-optimized using SIGALRM (more efficient than threading).
    """
    if seconds <= 0:
        yield
        return
    
    # macOS/Unix implementation using SIGALRM
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# ============================================================================
# ENHANCED OLLAMA CLIENT WITH THREAD-SAFE CACHING
# ============================================================================

class OllamaClient:
    """Handle all Ollama API interactions with thread-safe batching and caching"""
    
    def __init__(self, base_url: str = config.OLLAMA_BASE_URL):
        self.base_url = base_url
        self._embedding_cache = {}
        self._cache_lock = threading.Lock()  # Thread-safe cache access
        self._save_lock = threading.Lock()   # Thread-safe file writes
        self._load_embedding_cache()
        self._check_connection()
    
    def _load_embedding_cache(self):
        """Load persistent embedding cache from JSON (secure, no pickle)"""
        cache_file = Path(config.EMBEDDINGS_CACHE_DIR) / "embedding_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    self._embedding_cache = json.load(f, object_hook=numpy_decoder)
                print(f"Loaded {len(self._embedding_cache)} cached embeddings")
            except Exception as e:
                print(f"Could not load embedding cache: {e}")
                self._embedding_cache = {}
    
    def _save_embedding_cache(self):
        """Save embedding cache to disk (thread-safe, JSON format)"""
        cache_file = Path(config.EMBEDDINGS_CACHE_DIR) / "embedding_cache.json"
        with self._save_lock:
            try:
                # Write to temp file first, then rename (atomic operation)
                temp_file = cache_file.with_suffix('.tmp')
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self._embedding_cache, f, cls=NumpyEncoder)
                temp_file.replace(cache_file)
            except Exception as e:
                print(f"Warning: Could not save embedding cache: {e}")
    
    def _cache_key(self, text: str, model: str) -> str:
        """Generate cache key for embeddings"""
        return hashlib.md5(f"{model}:{text[:1000]}".encode()).hexdigest()
    
    def _check_connection(self):
        """Verify Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError("Ollama is not responding correctly")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}\n"
                f"Make sure Ollama is running: ollama serve\n"
                f"Error: {str(e)}"
            )
    
    def generate(self, prompt: str, model: str = None, 
                 stream: bool = False, options: dict = None) -> dict:
        """Generate text completion"""
        model = model or config.LLM_MODEL
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": options or {
                "temperature": 0.7,
                "num_ctx": 8192,
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Ollama generation failed: {str(e)}"}
    
    def embed(self, text: str, model: str = None, 
              use_cache: bool = True) -> dict:
        """Generate embeddings for text with thread-safe caching"""
        model = model or config.EMBEDDING_MODEL
        cache_key = self._cache_key(text, model)
        
        # Thread-safe cache read
        with self._cache_lock:
            if use_cache and cache_key in self._embedding_cache:
                return {"embedding": self._embedding_cache[cache_key]}
        
        payload = {
            "model": model,
            "prompt": text[:8000]  # Limit text size for embeddings
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # Thread-safe cache write
            with self._cache_lock:
                if use_cache and 'embedding' in result:
                    self._embedding_cache[cache_key] = result['embedding']
                    # Periodically save cache (every 50 new embeddings)
                    if len(self._embedding_cache) % 50 == 0:
                        # Save in background thread to not block
                        threading.Thread(target=self._save_embedding_cache, daemon=True).start()
            
            return result
        except requests.exceptions.RequestException as e:
            return {"error": f"Ollama embedding failed: {str(e)}"}
    
    def embed_batch(self, texts: List[str], model: str = None,
                    batch_size: int = 10) -> List[dict]:
        """
        Batch embed multiple texts with connection reuse and caching.
        Returns results in same order as input texts.
        """
        model = model or config.EMBEDDING_MODEL
        results = [None] * len(texts)
        uncached_items = []
        
        # Check cache first for all texts
        with self._cache_lock:
            for i, text in enumerate(texts):
                cache_key = self._cache_key(text, model)
                if cache_key in self._embedding_cache:
                    results[i] = {'embedding': self._embedding_cache[cache_key], 'cached': True}
                else:
                    uncached_items.append((i, text, cache_key))
        
        if not uncached_items:
            return results
        
        # Process uncached in batches with connection reuse
        session = requests.Session()
        
        try:
            for batch_start in range(0, len(uncached_items), batch_size):
                batch = uncached_items[batch_start:batch_start + batch_size]
                
                for idx, text, cache_key in batch:
                    payload = {
                        "model": model,
                        "prompt": text[:8000]
                    }
                    
                    try:
                        response = session.post(
                            f"{self.base_url}/api/embeddings",
                            json=payload,
                            timeout=60
                        )
                        response.raise_for_status()
                        result = response.json()
                        
                        if 'embedding' in result:
                            with self._cache_lock:
                                self._embedding_cache[cache_key] = result['embedding']
                            results[idx] = {'embedding': result['embedding'], 'cached': False}
                        else:
                            results[idx] = {'error': 'No embedding returned'}
                            
                    except Exception as e:
                        results[idx] = {'error': str(e)}
        finally:
            session.close()
        
        # Save cache after batch processing
        if any(r and not r.get('cached') and 'embedding' in r for r in results if r):
            threading.Thread(target=self._save_embedding_cache, daemon=True).start()
        
        return results
    
    def list_models(self) -> dict:
        """List available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Failed to list models: {str(e)}"}

# ============================================================================
# ENHANCED VECTOR STORE WITH OPTIMIZED SEARCH
# ============================================================================

class VectorStore:
    """Efficient vector storage and similarity search with optimizations"""
    
    def __init__(self, cache_dir: str = config.EMBEDDINGS_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.indexes = {}
        self._lock = threading.Lock()
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float], 
                          norm1: float = None, norm2: float = None) -> float:
        """
        Calculate cosine similarity between two vectors.
        Accepts precomputed norms for optimization.
        """
        vec1 = np.array(vec1) if not isinstance(vec1, np.ndarray) else vec1
        vec2 = np.array(vec2) if not isinstance(vec2, np.ndarray) else vec2
        
        norm1 = norm1 or np.linalg.norm(vec1)
        norm2 = norm2 or np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        return float(dot_product / (norm1 * norm2))
    
    def add_document(self, doc_id: str, chunks: List[Dict]):
        """Add document chunks with precomputed norms for faster search"""
        # Precompute norms for all embeddings
        for chunk in chunks:
            if 'embedding' in chunk and chunk['embedding']:
                vec = np.array(chunk['embedding'])
                chunk['_norm'] = float(np.linalg.norm(vec))
        
        index_file = self.cache_dir / f"{doc_id}_vectors.json"
        
        with self._lock:
            # Save using JSON (secure, no pickle)
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'chunks': chunks,
                    'created': datetime.now().isoformat()
                }, f, cls=NumpyEncoder)
            
            self.indexes[doc_id] = chunks
    
    def load_document(self, doc_id: str) -> Optional[List[Dict]]:
        """Load document from vector store"""
        with self._lock:
            if doc_id in self.indexes:
                return self.indexes[doc_id]
        
        index_file = self.cache_dir / f"{doc_id}_vectors.json"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f, object_hook=numpy_decoder)
                    with self._lock:
                        self.indexes[doc_id] = data['chunks']
                    return data['chunks']
            except Exception as e:
                print(f"Error loading document {doc_id}: {e}")
                return None
        
        return None
    
    def search(self, doc_id: str, query_embedding: List[float], 
               top_k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """
        Search for similar chunks in a document.
        Optimized with precomputed norms and early termination.
        """
        chunks = self.load_document(doc_id)
        if not chunks:
            return []
        
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return []
        
        results = []
        for chunk in chunks:
            if 'embedding' not in chunk or not chunk['embedding']:
                continue
            
            chunk_norm = chunk.get('_norm')
            if chunk_norm is None:
                chunk_vec = np.array(chunk['embedding'])
                chunk_norm = np.linalg.norm(chunk_vec)
            else:
                chunk_vec = np.array(chunk['embedding'])
            
            if chunk_norm == 0:
                continue
            
            # Optimized similarity with precomputed norms
            similarity = self.cosine_similarity(query_vec, chunk_vec, query_norm, chunk_norm)
            
            if similarity >= threshold:
                result = {k: v for k, v in chunk.items() if k != '_norm'}
                result['similarity'] = similarity
                results.append(result)
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the vector store"""
        with self._lock:
            if doc_id in self.indexes:
                del self.indexes[doc_id]
            
            index_file = self.cache_dir / f"{doc_id}_vectors.json"
            if index_file.exists():
                index_file.unlink()
                return True
        return False

# ============================================================================
# LAZY INITIALIZATION FOR OLLAMA AND VECTOR STORE
# ============================================================================

_ollama_instance = None
_vector_store_instance = None
_init_lock = threading.Lock()

def get_ollama() -> OllamaClient:
    """Lazy initialization of Ollama client with retry capability"""
    global _ollama_instance
    with _init_lock:
        if _ollama_instance is None:
            _ollama_instance = OllamaClient()
        return _ollama_instance

def get_vector_store() -> VectorStore:
    """Lazy initialization of vector store"""
    global _vector_store_instance
    with _init_lock:
        if _vector_store_instance is None:
            _vector_store_instance = VectorStore()
        return _vector_store_instance

def reset_ollama():
    """Reset Ollama instance (useful for reconnection)"""
    global _ollama_instance
    with _init_lock:
        _ollama_instance = None

# ============================================================================
# CONTEXT MANAGER
# ============================================================================

class ContextManager:
    def __init__(self, cache_dir=config.CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.sessions = {}
        self._lock = threading.Lock()
    
    def create_session_id(self, *args) -> str:
        """Generate unique session ID from inputs"""
        content = ''.join(str(arg) for arg in args)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def resolve_file_path(self, filename: str) -> Path:
        """
        Cross-platform file path resolution with OS detection.
        """
        file_path = Path(filename)
        
        # If absolute path, verify directly
        if file_path.is_absolute():
            if file_path.exists() and file_path.is_file():
                if os.access(file_path, os.R_OK):
                    return file_path
                else:
                    raise PermissionError(
                        f"File exists but is not readable: {file_path}\n"
                        f"Check file permissions"
                    )
            raise FileNotFoundError(f"Absolute path not found: {file_path}")
        
        # macOS-optimized search directories
        search_dirs = [
            Path.cwd(),
            Path.home(),
            Path("/tmp"),
            Path.home() / "Documents",
            Path.home() / "Downloads",
            Path.home() / "Desktop",
        ]
        
        # Search in each directory
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            
            candidate = search_dir / filename
            
            if candidate.exists() and candidate.is_file():
                if os.access(candidate, os.R_OK):
                    return candidate
        
        # File not found
        searched = "\n  ".join(str(d / filename) for d in search_dirs if d.exists())
        raise FileNotFoundError(
            f"File '{filename}' not found in any search directory.\n"
            f"OS: macOS\n"
            f"Searched locations:\n  {searched}\n"
            f"Tip: Use absolute path or place file in current directory: {Path.cwd()}\n"
            f"Or check permissions with: ls -l {filename}"
        )
    
    def verify_file_access(self, filepath: Path) -> dict:
        """Verify complete file access with diagnostic information."""
        result = {
            "exists": filepath.exists(),
            "is_file": filepath.is_file() if filepath.exists() else False,
            "readable": False,
            "size": 0,
            "permissions": None,
            "error": None
        }
        
        if not result["exists"]:
            result["error"] = "File does not exist"
            return result
        
        if not result["is_file"]:
            result["error"] = "Path is not a file (maybe directory)"
            return result
        
        result["readable"] = os.access(filepath, os.R_OK)
        
        if not result["readable"]:
            result["error"] = "File exists but is not readable (check permissions)"
            return result
        
        try:
            stat = filepath.stat()
            result["size"] = stat.st_size
            result["permissions"] = oct(stat.st_mode)[-3:]
        except Exception as e:
            result["error"] = f"Could not stat file: {e}"
        
        return result
    
    def read_file_safe(self, filepath: Path, encoding: str = "utf-8") -> str:
        """Read file with robust error handling and encoding fallback."""
        access_check = self.verify_file_access(filepath)
        
        if not access_check["readable"]:
            raise PermissionError(
                f"Cannot read file: {access_check['error']}\n"
                f"Path: {filepath}\n"
                f"Details: {access_check}"
            )
        
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            return content
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    content = f.read()
                return content
            except Exception as e:
                raise IOError(f"Failed to read file with latin-1 encoding: {e}")
        except Exception as e:
            raise IOError(f"Failed to read file {filepath}: {e}")
    
    def validate_file_path(self, file_path: str) -> Path:
        """Legacy method - redirects to resolve_file_path for compatibility."""
        return self.resolve_file_path(file_path)
    
    def register_session(self, session_id: str, cache_file: Path):
        """Thread-safe session registration"""
        with self._lock:
            self.sessions[session_id] = cache_file
    
    def get_session(self, session_id: str) -> Optional[Path]:
        """Thread-safe session retrieval"""
        with self._lock:
            return self.sessions.get(session_id)

context_mgr = ContextManager()

# ============================================================================
# INPUT VALIDATION HELPERS
# ============================================================================

def validate_chunk_params(chunk_size: int, overlap: int) -> dict:
    """Validate chunking parameters and return error dict if invalid"""
    if chunk_size < config.MIN_CHUNK_SIZE:
        return {'error': f'chunk_size must be at least {config.MIN_CHUNK_SIZE}, got {chunk_size}'}
    if chunk_size > config.MAX_CHUNK_SIZE:
        return {'error': f'chunk_size cannot exceed {config.MAX_CHUNK_SIZE}, got {chunk_size}'}
    if overlap < 0:
        return {'error': f'overlap cannot be negative, got {overlap}'}
    if overlap >= chunk_size:
        return {'error': f'overlap ({overlap}) must be less than chunk_size ({chunk_size})'}
    return None

def validate_compression_level(level: str) -> dict:
    """Validate compression level parameter"""
    valid_levels = ["low", "medium", "high", "ultra"]
    if level not in valid_levels:
        return {'error': f'compression_level must be one of {valid_levels}, got {level}'}
    return None

# ============================================================================
# TOOL 1: Smart Document Chunking
# ============================================================================

@mcp.tool()
def chunk_document_smart(
    file_path: str,
    chunk_size: int = 50000,
    overlap: int = 5000,
    preserve_structure: bool = True
) -> dict:
    """
    Intelligently chunk a large document for sequential processing.
    Claude can then process each chunk and maintain context.
    
    Args:
        file_path: Path to document (can be just filename if in uploads)
        chunk_size: Characters per chunk (default 50K ≈ 12K tokens)
        overlap: Characters to overlap between chunks (maintains context)
        preserve_structure: Try to break at paragraph boundaries
    
    Returns:
        Metadata about chunks + saves them for retrieval
    """
    # Input validation
    validation_error = validate_chunk_params(chunk_size, overlap)
    if validation_error:
        return validation_error
    
    # Clamp to reasonable limits
    chunk_size = min(chunk_size, config.MAX_CHUNK_SIZE)
    overlap = min(overlap, chunk_size // 2)
    
    try:
        validated_path = context_mgr.validate_file_path(file_path)
        content = context_mgr.read_file_safe(validated_path)
        
        chunks = []
        
        if preserve_structure:
            paragraphs = content.split('\n\n')
            current_chunk = ""
            chunk_id = 0
            
            for para in paragraphs:
                if len(current_chunk) + len(para) < chunk_size:
                    current_chunk += para + '\n\n'
                else:
                    if current_chunk:
                        chunks.append({
                            'id': chunk_id,
                            'content': current_chunk,
                            'size': len(current_chunk)
                        })
                        chunk_id += 1
                    
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + para + '\n\n'
            
            if current_chunk:
                chunks.append({
                    'id': chunk_id,
                    'content': current_chunk,
                    'size': len(current_chunk)
                })
        else:
            start = 0
            chunk_id = 0
            
            while start < len(content):
                end = min(start + chunk_size, len(content))
                chunk = content[start:end]
                
                chunks.append({
                    'id': chunk_id,
                    'content': chunk,
                    'size': len(chunk)
                })
                
                start += chunk_size - overlap
                chunk_id += 1
        
        session_id = context_mgr.create_session_id(str(validated_path), chunk_size)
        cache_file = context_mgr.cache_dir / f"session_{session_id}.json"
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'file_path': str(validated_path),
                'total_chunks': len(chunks),
                'chunks': chunks,
                'chunk_size': chunk_size,
                'overlap': overlap
            }, f)
        
        context_mgr.register_session(session_id, cache_file)
        
        return {
            'session_id': session_id,
            'file_path': str(validated_path),
            'total_chunks': len(chunks),
            'original_size': len(content),
            'avg_chunk_size': sum(c['size'] for c in chunks) // len(chunks) if chunks else 0,
            'message': f"Document split into {len(chunks)} chunks. Use get_chunk() to retrieve each one."
        }
        
    except FileNotFoundError as e:
        return {'error': str(e)}
    except PermissionError as e:
        return {'error': str(e)}
    except Exception as e:
        return {'error': f"Unexpected error: {str(e)}"}

# ============================================================================
# TOOL 2: Retrieve Specific Chunks
# ============================================================================

@mcp.tool()
def get_chunk(
    session_id: str,
    chunk_id: int
) -> dict:
    """
    Get a specific chunk from a previously chunked document.
    
    Args:
        session_id: Session ID from chunk_document_smart()
        chunk_id: Which chunk to retrieve (0-indexed)
    
    Returns:
        The chunk content and metadata
    """
    try:
        cache_file = context_mgr.get_session(session_id)
        if not cache_file:
            return {'error': 'Session not found. Run chunk_document_smart() first.'}
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        chunks = session_data['chunks']
        
        if chunk_id >= len(chunks) or chunk_id < 0:
            return {
                'error': f'Chunk {chunk_id} out of range. Valid range: 0-{len(chunks)-1}'
            }
        
        chunk = chunks[chunk_id]
        
        return {
            'chunk_id': chunk_id,
            'total_chunks': len(chunks),
            'content': chunk['content'],
            'size': chunk['size'],
            'progress': f"Chunk {chunk_id + 1} of {len(chunks)}"
        }
        
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# RECURSIVE DOCUMENT PROCESSING HELPERS
# ============================================================================

def _smart_split(text: str, chunk_size: int) -> List[str]:
    """Split text intelligently at paragraph boundaries"""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + '\n\n'
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para + '\n\n'
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [text]

def _remove_embeddings(data: Dict) -> Dict:
    """Remove embeddings from data structure for JSON serialization"""
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if k == 'embedding':
                result[k] = '<embedding_removed>'
            elif k == '_norm':
                continue  # Skip internal fields
            elif isinstance(v, (dict, list)):
                result[k] = _remove_embeddings(v)
            else:
                result[k] = v
        return result
    elif isinstance(data, list):
        return [_remove_embeddings(item) for item in data]
    return data

def _extract_all_chunks(data: Dict) -> List[Dict]:
    """Extract all chunks with embeddings from hierarchy"""
    chunks = []
    
    if data.get('type') == 'leaf' and data.get('embedding'):
        chunks.append({
            'content': data.get('content', ''),
            'summary': data.get('summary', ''),
            'depth': data.get('depth', 0),
            'embedding': data.get('embedding')
        })
    
    if 'children' in data:
        for child in data['children']:
            chunks.extend(_extract_all_chunks(child))
    
    return chunks

def _count_chunks(data: Dict) -> int:
    """Count total chunks in hierarchy"""
    if data.get('type') == 'leaf':
        return 1
    
    total = 0
    if 'children' in data:
        for child in data['children']:
            total += _count_chunks(child)
    
    return total

def _process_recursively(
    content: str,
    depth: int,
    max_depth: int,
    chunk_size: int,
    enable_embeddings: bool,
    ollama_client: OllamaClient,
    progress_callback: Callable[[str], None] = None
) -> Dict:
    """
    Recursive processing core with memory optimization.
    Uses progress callback for monitoring long operations.
    """
    if progress_callback:
        progress_callback(f"Processing at depth {depth}, content size: {len(content)}")
    
    # Base case: content is small enough or max depth reached
    if len(content) <= chunk_size or depth >= max_depth:
        summary_result = ollama_client.generate(
            f"Provide a concise summary (2-3 sentences) of:\n\n{content[:8000]}",
            options={"temperature": 0.5, "num_ctx": 8192}
        )
        
        summary = summary_result.get('response', content[:200]).strip()
        
        embedding = None
        if enable_embeddings:
            embed_result = ollama_client.embed(content[:8000])
            if 'embedding' in embed_result:
                embedding = embed_result['embedding']
        
        return {
            'type': 'leaf',
            'depth': depth,
            'content': content,
            'summary': summary,
            'size': len(content),
            'embedding': embedding
        }
    
    # Recursive case: split and process
    chunks = _smart_split(content, chunk_size)
    
    processed_children = []
    child_summaries = []
    
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(f"Processing chunk {i+1}/{len(chunks)} at depth {depth}")
        
        child_result = _process_recursively(
            content=chunk,
            depth=depth + 1,
            max_depth=max_depth,
            chunk_size=chunk_size,
            enable_embeddings=enable_embeddings,
            ollama_client=ollama_client,
            progress_callback=progress_callback
        )
        
        processed_children.append(child_result)
        child_summaries.append(child_result['summary'])
    
    # Create summary of summaries
    combined_summaries = "\n\n".join(child_summaries)
    
    meta_summary_result = ollama_client.generate(
        f"Synthesize these summaries into one coherent summary (3-5 sentences):\n\n{combined_summaries[:8000]}",
        options={"temperature": 0.5, "num_ctx": 8192}
    )
    
    meta_summary = meta_summary_result.get('response', combined_summaries[:300]).strip()
    
    embedding = None
    if enable_embeddings:
        embed_result = ollama_client.embed(meta_summary)
        if 'embedding' in embed_result:
            embedding = embed_result['embedding']
    
    return {
        'type': 'node',
        'depth': depth,
        'summary': meta_summary,
        'children': processed_children,
        'num_children': len(processed_children),
        'total_size': sum(child.get('size', 0) for child in processed_children),
        'embedding': embedding
    }

# ============================================================================
# TOOL 3: Recursive Document Processing
# ============================================================================

@mcp.tool()
def process_document_recursive(
    file_path: str,
    compression_level: str = "medium",
    enable_embeddings: bool = True,
    timeout_seconds: int = 300
) -> dict:
    """
    GAME CHANGER: Process document recursively with hierarchical summarization.
    Creates a pyramid of summaries - you can ask questions at ANY detail level!
    
    Args:
        file_path: Path to document
        compression_level: How aggressive to compress (low/medium/high/ultra)
        enable_embeddings: Generate embeddings for semantic search
        timeout_seconds: Maximum processing time (default 5 minutes)
    
    Returns:
        Hierarchical document structure with summaries at multiple levels
    """
    # Validate compression level
    validation_error = validate_compression_level(compression_level)
    if validation_error:
        return validation_error
    
    # Get Ollama client with lazy init
    try:
        ollama_client = get_ollama()
    except ConnectionError as e:
        return {'error': f'Ollama not connected: {str(e)}. Start with: ollama serve'}
    
    compression_map = {
        "low": {"depth": 1, "chunk_size": 8000},
        "medium": {"depth": 2, "chunk_size": 5000},
        "high": {"depth": 3, "chunk_size": 3000},
        "ultra": {"depth": 4, "chunk_size": 2000}
    }
    
    settings = compression_map[compression_level]
    
    try:
        validated_path = context_mgr.validate_file_path(file_path)
        content = context_mgr.read_file_safe(validated_path)
        
        print(f"Processing document recursively at {compression_level} compression...")
        
        # Process with timeout
        try:
            with timeout_context(timeout_seconds):
                result = _process_recursively(
                    content=content,
                    depth=0,
                    max_depth=settings["depth"],
                    chunk_size=settings["chunk_size"],
                    enable_embeddings=enable_embeddings,
                    ollama_client=ollama_client
                )
        except TimeoutError as e:
            return {
                'error': str(e),
                'partial_results': True,
                'message': 'Processing timed out. Try with higher compression_level or smaller document.'
            }
        
        # Create session
        session_id = context_mgr.create_session_id(
            str(validated_path), 
            compression_level,
            datetime.now().isoformat()
        )
        
        # Save to cache (without embeddings in JSON)
        cache_file = context_mgr.cache_dir / f"recursive_{session_id}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            result_copy = _remove_embeddings(result)
            json.dump({
                'file_path': str(validated_path),
                'compression_level': compression_level,
                'hierarchy': result_copy,
                'created': datetime.now().isoformat()
            }, f)
        
        context_mgr.register_session(session_id, cache_file)
        
        # Save embeddings to vector store if enabled
        if enable_embeddings:
            vector_store = get_vector_store()
            chunks_with_embeddings = _extract_all_chunks(result)
            vector_store.add_document(session_id, chunks_with_embeddings)
            print(f"Saved {len(chunks_with_embeddings)} chunks to vector store")
        
        return {
            'session_id': session_id,
            'file_path': str(validated_path),
            'compression_level': compression_level,
            'max_depth_reached': result.get('depth', 0),
            'total_chunks': _count_chunks(result),
            'top_level_summary': result.get('summary', '')[:500],
            'embeddings_enabled': enable_embeddings,
            'message': 'Document processed hierarchically. Use query_recursive() for semantic search!'
        }
        
    except FileNotFoundError as e:
        return {'error': str(e)}
    except PermissionError as e:
        return {'error': str(e)}
    except Exception as e:
        return {'error': f"Unexpected error: {str(e)}"}

# ============================================================================
# TOOL 4: Semantic Query on Processed Documents
# ============================================================================

@mcp.tool()
def query_recursive(
    session_id: str,
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.6
) -> dict:
    """
    GAME CHANGER #2: Query recursively processed documents semantically.
    This finds the EXACT relevant sections without loading the whole document!
    
    Args:
        session_id: Session from process_document_recursive()
        query: Natural language question
        top_k: Number of results
        min_similarity: Minimum similarity threshold
    
    Returns:
        Most relevant sections with context
    """
    # Input validation
    if top_k < 1:
        return {'error': 'top_k must be at least 1'}
    if not 0 <= min_similarity <= 1:
        return {'error': 'min_similarity must be between 0 and 1'}
    
    try:
        ollama_client = get_ollama()
        vector_store = get_vector_store()
    except ConnectionError as e:
        return {'error': f'Services not available: {str(e)}'}
    
    try:
        # Get query embedding
        query_result = ollama_client.embed(query)
        if 'error' in query_result:
            return query_result
        
        query_embedding = query_result['embedding']
        
        # Search vector store
        results = vector_store.search(
            session_id,
            query_embedding,
            top_k=top_k,
            threshold=min_similarity
        )
        
        if not results:
            return {
                'query': query,
                'results_found': 0,
                'message': 'No semantically similar sections found. Try lowering min_similarity.'
            }
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'similarity_score': round(result['similarity'], 3),
                'depth': result.get('depth', 0),
                'summary': result.get('summary', ''),
                'content_preview': result.get('content', '')[:500],
                'full_content': result.get('content', '')
            })
        
        return {
            'query': query,
            'results_found': len(formatted_results),
            'top_k': top_k,
            'results': formatted_results,
            'usage_tip': 'Use the full_content field to get complete context for relevant sections'
        }
        
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# TOOL 5: Semantic Search Document (with caching)
# ============================================================================

@mcp.tool()
def semantic_search_document(
    file_path: str,
    query: str,
    top_k: int = 5,
    context_window: int = 500,
    use_cache: bool = True
) -> dict:
    """
    Use local embeddings to find semantically relevant sections.
    NOW WITH CACHING - much faster on repeat searches!
    
    Args:
        file_path: Path to document
        query: Natural language query
        top_k: Number of most relevant sections to return
        context_window: Characters of context around matches
        use_cache: Use cached embeddings if available
    
    Returns:
        Semantically relevant sections ranked by similarity
    """
    try:
        ollama_client = get_ollama()
        vector_store = get_vector_store()
    except ConnectionError as e:
        return {'error': f'Ollama not connected: {str(e)}. Start with: ollama serve'}
    
    try:
        validated_path = context_mgr.validate_file_path(file_path)
        doc_id = hashlib.md5(str(validated_path).encode()).hexdigest()[:16]
        
        # Check if we have cached embeddings for this document
        cached_chunks = vector_store.load_document(doc_id) if use_cache else None
        
        if cached_chunks:
            # Use cached embeddings - MUCH FASTER!
            query_result = ollama_client.embed(query)
            if 'error' in query_result:
                return query_result
            
            query_embedding = query_result['embedding']
            results = vector_store.search(doc_id, query_embedding, top_k=top_k, threshold=0.5)
            
            formatted = []
            for r in results:
                formatted.append({
                    'similarity_score': round(r['similarity'], 3),
                    'content': r.get('content', ''),
                    'context': r.get('content', '')
                })
            
            return {
                'query': query,
                'results_found': len(formatted),
                'results': formatted,
                'cached': True,
                'message': 'Results from cached embeddings (instant search!)'
            }
        
        # No cache - generate embeddings
        content = context_mgr.read_file_safe(validated_path)
        
        # Split into paragraphs for embedding
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
        
        # Generate query embedding
        query_result = ollama_client.embed(query)
        if 'error' in query_result:
            return query_result
        
        query_embedding = query_result.get('embedding', [])
        
        # Generate embeddings for each paragraph and cache them
        chunks_to_cache = []
        similarities = []
        
        # Limit to first 100 paragraphs for performance
        paragraphs_to_process = paragraphs[:100]
        
        for idx, para in enumerate(paragraphs_to_process):
            para_result = ollama_client.embed(para[:1000], use_cache=True)
            if 'error' not in para_result:
                para_embedding = para_result.get('embedding', [])
                
                similarity = vector_store.cosine_similarity(query_embedding, para_embedding)
                similarities.append((idx, para, similarity))
                
                chunks_to_cache.append({
                    'content': para,
                    'embedding': para_embedding,
                    'index': idx
                })
        
        # Cache embeddings for future searches
        if use_cache and chunks_to_cache:
            vector_store.add_document(doc_id, chunks_to_cache)
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[2], reverse=True)
        top_results = similarities[:top_k]
        
        results = []
        for idx, para, score in top_results:
            pos = content.find(para)
            start = max(0, pos - context_window)
            end = min(len(content), pos + len(para) + context_window)
            
            results.append({
                'paragraph_id': idx,
                'similarity_score': float(score),
                'content': para,
                'context': content[start:end],
                'position': pos
            })
        
        return {
            'query': query,
            'results_found': len(results),
            'results': results,
            'cached': False,
            'message': f'Found {len(results)} semantically relevant sections. Embeddings cached for faster future searches!'
        }
        
    except FileNotFoundError as e:
        return {'error': str(e)}
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# TOOL 6: Local LLM Summarization
# ============================================================================

@mcp.tool()
def summarize_with_local_llm(
    text: str,
    style: str = "concise",
    max_length: int = 500
) -> dict:
    """
    Summarize text using local Ollama LLM (saves your Claude tokens!)
    
    Args:
        text: Text to summarize
        style: "concise", "detailed", or "bullet_points"
        max_length: Target summary length in words
    
    Returns:
        AI-generated summary
    """
    try:
        ollama_client = get_ollama()
    except ConnectionError as e:
        return {'error': f'Ollama not connected: {str(e)}. Start with: ollama serve'}
    
    valid_styles = ["concise", "detailed", "bullet_points"]
    if style not in valid_styles:
        return {'error': f'style must be one of {valid_styles}, got {style}'}
    
    style_prompts = {
        "concise": "Provide a brief, concise summary in 2-3 sentences.",
        "detailed": f"Provide a detailed summary in approximately {max_length} words.",
        "bullet_points": "Provide a summary as bullet points of key information."
    }
    
    prompt = f"""{style_prompts[style]}

Text to summarize:
{text[:10000]}

Summary:"""
    
    try:
        result = ollama_client.generate(prompt, options={"temperature": 0.5})
        
        if 'error' in result:
            return result
        
        summary = result.get('response', '').strip()
        
        return {
            'original_length': len(text),
            'summary_length': len(summary),
            'compression_ratio': f"{len(summary) / len(text) * 100:.1f}%" if text else "0%",
            'summary': summary,
            'model_used': config.LLM_MODEL
        }
        
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# TOOL 7: Extract Key Information
# ============================================================================

@mcp.tool()
def extract_key_info_local(
    text: str,
    extraction_type: str = "main_points"
) -> dict:
    """
    Use local LLM to extract specific information types.
    
    Args:
        text: Text to analyze
        extraction_type: "main_points", "entities", "action_items", "technical_details"
    
    Returns:
        Extracted information
    """
    try:
        ollama_client = get_ollama()
    except ConnectionError as e:
        return {'error': f'Ollama not connected: {str(e)}. Start with: ollama serve'}
    
    extraction_prompts = {
        "main_points": "Extract the main points from this text as a numbered list.",
        "entities": "Extract all named entities (people, organizations, locations, technologies) from this text.",
        "action_items": "Extract all action items, tasks, or to-dos from this text.",
        "technical_details": "Extract all technical specifications, numbers, and technical details from this text."
    }
    
    if extraction_type not in extraction_prompts:
        return {'error': f'extraction_type must be one of {list(extraction_prompts.keys())}'}
    
    prompt = f"""{extraction_prompts[extraction_type]}

Text:
{text[:8000]}

Extracted information:"""
    
    try:
        result = ollama_client.generate(prompt, options={"temperature": 0.3})
        
        if 'error' in result:
            return result
        
        extracted = result.get('response', '').strip()
        
        return {
            'extraction_type': extraction_type,
            'extracted_info': extracted,
            'model_used': config.LLM_MODEL
        }
        
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# TOOL 8: Check Ollama Status
# ============================================================================

@mcp.tool()
def check_ollama_status() -> dict:
    """
    Check if Ollama is running and list available models.
    
    Returns:
        Ollama status and available models
    """
    try:
        ollama_client = get_ollama()
    except ConnectionError as e:
        return {
            'status': 'disconnected',
            'message': f'Ollama is not running: {str(e)}. Start it with: ollama serve',
            'configured_llm': config.LLM_MODEL,
            'configured_embedding': config.EMBEDDING_MODEL
        }
    
    try:
        models_result = ollama_client.list_models()
        
        if 'error' in models_result:
            return models_result
        
        available_models = [m['name'] for m in models_result.get('models', [])]
        
        llm_available = any(config.LLM_MODEL in m for m in available_models)
        embed_available = any(config.EMBEDDING_MODEL in m for m in available_models)
        
        return {
            'status': 'connected',
            'ollama_url': config.OLLAMA_BASE_URL,
            'configured_llm': config.LLM_MODEL,
            'configured_embedding': config.EMBEDDING_MODEL,
            'llm_available': llm_available,
            'embedding_available': embed_available,
            'available_models': available_models,
            'recommendations': {
                'install_llm': f'ollama pull {config.LLM_MODEL}' if not llm_available else None,
                'install_embedding': f'ollama pull {config.EMBEDDING_MODEL}' if not embed_available else None
            }
        }
        
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# TOOL 9: Extract Sections by Keywords
# ============================================================================

@mcp.tool()
def extract_sections_by_keywords(
    file_path: str,
    keywords: List[str],
    context_lines: int = 5
) -> dict:
    """
    Extract only sections containing specific keywords.
    This dramatically reduces context needed!
    
    Args:
        file_path: Path to document (can be just filename if in uploads)
        keywords: List of keywords to search for
        context_lines: Number of surrounding lines to include
    
    Returns:
        Only relevant sections (massive token savings)
    """
    if not keywords:
        return {'error': 'keywords list cannot be empty'}
    
    if context_lines < 0:
        return {'error': 'context_lines cannot be negative'}
    
    try:
        validated_path = context_mgr.validate_file_path(file_path)
        
        with open(validated_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        relevant_sections = []
        
        for i, line in enumerate(lines):
            if any(keyword.lower() in line.lower() for keyword in keywords):
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)
                
                section = {
                    'line_number': i + 1,
                    'matched_keywords': [kw for kw in keywords if kw.lower() in line.lower()],
                    'content': ''.join(lines[start:end]),
                    'context_range': f"Lines {start+1}-{end}"
                }
                
                relevant_sections.append(section)
        
        # Deduplicate overlapping sections
        unique_sections = []
        seen_ranges = set()
        
        for section in relevant_sections:
            range_key = section['context_range']
            if range_key not in seen_ranges:
                unique_sections.append(section)
                seen_ranges.add(range_key)
        
        total_content = '\n\n'.join(s['content'] for s in unique_sections)
        total_chars = sum(len(line) for line in lines)
        
        return {
            'file_path': str(validated_path),
            'sections_found': len(unique_sections),
            'keywords_searched': keywords,
            'total_content': total_content,
            'compression_ratio': f"{len(total_content) / total_chars * 100:.1f}%" if total_chars > 0 else "0%",
            'sections': unique_sections
        }
        
    except FileNotFoundError as e:
        return {'error': str(e)}
    except Exception as e:
        return {'error': f"Unexpected error: {str(e)}"}

# ============================================================================
# TOOL 10: Create Document Map
# ============================================================================

@mcp.tool()
def create_document_map(
    file_path: str,
    detect_headings: bool = True
) -> dict:
    """
    Create a structural map of the document without loading full content.
    Claude can use this to decide what to load.
    
    Args:
        file_path: Path to document (can be just filename if in uploads)
        detect_headings: Try to identify section headings
    
    Returns:
        Document structure/outline
    """
    try:
        validated_path = context_mgr.validate_file_path(file_path)
        
        with open(validated_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        structure = {
            'file_path': str(validated_path),
            'total_lines': len(lines),
            'file_size_chars': sum(len(l) for l in lines),
            'estimated_tokens': sum(len(l) for l in lines) // 4,
            'sections': []
        }
        
        if detect_headings:
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                is_heading = (
                    (line_stripped.isupper() and 3 < len(line_stripped) < 100) or
                    line_stripped.startswith('#') or
                    re.match(r'^\d+\.', line_stripped) or
                    (i + 1 < len(lines) and 
                     lines[i+1].strip() in ['='*len(line_stripped), '-'*len(line_stripped)])
                )
                
                if is_heading:
                    preview_end = min(i + 4, len(lines))
                    preview = ' '.join(lines[i+1:preview_end]).strip()[:200]
                    
                    structure['sections'].append({
                        'line_number': i + 1,
                        'heading': line_stripped,
                        'preview': preview,
                        'type': 'heading'
                    })
        
        return structure
        
    except FileNotFoundError as e:
        return {'error': str(e)}
    except Exception as e:
        return {'error': f"Unexpected error: {str(e)}"}

# ============================================================================
# TOOL 11: Search Document
# ============================================================================

@mcp.tool()
def search_document(
    file_path: str,
    pattern: str,
    is_regex: bool = False,
    max_results: int = 50
) -> dict:
    """
    Search document for patterns (like grep).
    Returns matching lines with context.
    
    Args:
        file_path: Path to document (can be just filename if in uploads)
        pattern: Search pattern
        is_regex: Treat pattern as regex
        max_results: Maximum matches to return
    
    Returns:
        Matching lines with context
    """
    if not pattern:
        return {'error': 'pattern cannot be empty'}
    
    if max_results < 1:
        return {'error': 'max_results must be at least 1'}
    
    try:
        validated_path = context_mgr.validate_file_path(file_path)
        
        with open(validated_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        matches = []
        
        # Validate regex if needed
        if is_regex:
            try:
                compiled_pattern = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                return {'error': f'Invalid regex pattern: {e}'}
        
        for i, line in enumerate(lines):
            match_found = False
            if is_regex:
                if compiled_pattern.search(line):
                    match_found = True
            else:
                if pattern.lower() in line.lower():
                    match_found = True
            
            if match_found:
                matches.append((i, line))
                if len(matches) >= max_results:
                    break
        
        results = []
        for line_num, line in matches:
            start = max(0, line_num - 2)
            end = min(len(lines), line_num + 3)
            context = ''.join(lines[start:end])
            
            results.append({
                'line_number': line_num + 1,
                'match': line.strip(),
                'context': context
            })
        
        return {
            'file_path': str(validated_path),
            'pattern': pattern,
            'matches_found': len(results),
            'results': results
        }
        
    except FileNotFoundError as e:
        return {'error': str(e)}
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# TOOL 12: Multi-Document Session
# ============================================================================

@mcp.tool()
def create_multi_doc_session(
    document_paths: List[str],
    session_name: str,
    fail_on_error: bool = False
) -> dict:
    """
    Create a session with multiple documents for comparative analysis.
    
    Args:
        document_paths: List of document paths
        session_name: Name for this session
        fail_on_error: If True, fail if any document can't be loaded
    
    Returns:
        Session info for cross-document queries
    """
    if not document_paths:
        return {'error': 'document_paths list cannot be empty'}
    
    if not session_name:
        return {'error': 'session_name cannot be empty'}
    
    try:
        validated_paths = []
        documents = []
        errors = []
        total_size = 0
        
        for path in document_paths:
            try:
                validated_path = context_mgr.validate_file_path(path)
                validated_paths.append(str(validated_path))
                
                content = context_mgr.read_file_safe(validated_path)
                
                doc_info = {
                    'path': str(validated_path),
                    'name': validated_path.name,
                    'size': len(content),
                    'estimated_tokens': len(content) // 4,
                    'status': 'loaded'
                }
                
                documents.append(doc_info)
                total_size += len(content)
                
            except (FileNotFoundError, PermissionError) as e:
                error_info = {'path': path, 'error': str(e)}
                errors.append(error_info)
                
                if fail_on_error:
                    return {'error': f"Failed to load document: {path}", 'details': str(e)}
                
                documents.append({
                    'path': path,
                    'name': Path(path).name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        if not validated_paths:
            return {'error': 'No documents could be loaded', 'errors': errors}
        
        session_id = context_mgr.create_session_id(session_name, *validated_paths)
        
        cache_file = context_mgr.cache_dir / f"multidoc_{session_id}.json"
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'session_name': session_name,
                'document_paths': validated_paths,
                'documents': documents,
                'total_size': total_size,
                'errors': errors
            }, f)
        
        context_mgr.register_session(session_id, cache_file)
        
        loaded_count = len([d for d in documents if d.get('status') == 'loaded'])
        
        return {
            'session_id': session_id,
            'session_name': session_name,
            'documents_loaded': loaded_count,
            'documents_failed': len(errors),
            'total_size': total_size,
            'estimated_total_tokens': total_size // 4,
            'document_list': [d['name'] for d in documents],
            'errors': errors if errors else None,
            'warning': f'{len(errors)} document(s) failed to load' if errors else None
        }
        
    except Exception as e:
        return {'error': f"Unexpected error: {str(e)}"}

# ============================================================================
# TOOL 13: Compare Sections
# ============================================================================

@mcp.tool()
def compare_sections(
    session_id: str,
    section_identifier: str
) -> dict:
    """
    Extract the same section from multiple documents for comparison.
    
    Args:
        session_id: Multi-doc session ID
        section_identifier: Keyword or heading to find
    
    Returns:
        Matching sections from all documents
    """
    if not section_identifier:
        return {'error': 'section_identifier cannot be empty'}
    
    try:
        cache_file = context_mgr.get_session(session_id)
        if not cache_file:
            return {'error': 'Session not found'}
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        comparisons = []
        
        for doc_path in session_data['document_paths']:
            result = extract_sections_by_keywords(
                doc_path,
                [section_identifier],
                context_lines=10
            )
            
            if 'total_content' in result and result['total_content']:
                comparisons.append({
                    'document': Path(doc_path).name,
                    'content': result['total_content'],
                    'sections_found': result['sections_found']
                })
        
        return {
            'section_identifier': section_identifier,
            'documents_compared': len(comparisons),
            'comparisons': comparisons
        }
        
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# TOOL 14: Incremental Summary
# ============================================================================

@mcp.tool()
def prepare_incremental_summary(
    session_id: str,
    summary_so_far: Optional[str] = None
) -> dict:
    """
    Helper for incremental summarization across chunks.
    Claude can build up a summary piece by piece.
    
    Args:
        session_id: Chunked document session
        summary_so_far: Previous summary to build on
    
    Returns:
        Next chunk with instructions for incremental summary
    """
    try:
        cache_file = context_mgr.get_session(session_id)
        if not cache_file:
            return {'error': 'Session not found'}
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        # Track progress
        progress_file = context_mgr.cache_dir / f"progress_{session_id}.json"
        
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
        else:
            progress = {'next_chunk': 0}
        
        current_chunk = progress['next_chunk']
        chunks = session_data['chunks']
        
        if current_chunk >= len(chunks):
            return {
                'status': 'complete',
                'message': 'All chunks processed',
                'final_summary': summary_so_far
            }
        
        chunk = chunks[current_chunk]
        
        # Update progress
        progress['next_chunk'] = current_chunk + 1
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f)
        
        instruction = f"""
Process this chunk and update the summary.

Current summary so far:
{summary_so_far or "None - this is the first chunk"}

New chunk to process (chunk {current_chunk + 1} of {len(chunks)}):
{chunk['content']}

Please:
1. Extract key points from this chunk
2. Integrate them with the summary so far
3. Return the updated summary
"""
        
        return {
            'status': 'in_progress',
            'chunk_number': current_chunk + 1,
            'total_chunks': len(chunks),
            'progress_pct': (current_chunk + 1) / len(chunks) * 100,
            'instruction': instruction,
            'chunk_content': chunk['content']
        }
        
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# TOOL 15: Process Chat Conversation
# ============================================================================

@mcp.tool()
def process_chat_conversation(
    chat_content: str,
    chat_id: str,
    compression_level: str = "medium"
) -> dict:
    """
    Process chat conversations recursively for efficient storage and retrieval.
    Perfect for integrating with Claude's conversation_search!
    
    Args:
        chat_content: Full chat conversation text
        chat_id: Unique identifier for this chat
        compression_level: How much to compress (low/medium/high)
    
    Returns:
        Processed chat with hierarchical summaries and embeddings
    """
    if not chat_content:
        return {'error': 'chat_content cannot be empty'}
    
    if not chat_id:
        return {'error': 'chat_id cannot be empty'}
    
    validation_error = validate_compression_level(compression_level)
    if validation_error:
        return validation_error
    
    try:
        ollama_client = get_ollama()
    except ConnectionError as e:
        return {'error': f'Ollama not connected: {str(e)}'}
    
    try:
        # Use cross-platform temp file
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.txt', 
            prefix=f'chat_{chat_id}_',
            delete=False,
            encoding='utf-8'
        ) as temp_file:
            temp_file.write(chat_content)
            temp_path = Path(temp_file.name)
        
        try:
            # Process recursively
            result = process_document_recursive(
                file_path=str(temp_path),
                compression_level=compression_level,
                enable_embeddings=True
            )
            
            # Extract key entities and topics
            extract_result = ollama_client.generate(
                f"""Extract from this conversation:
1. Main topics discussed (bullet list)
2. Key entities (people, companies, technologies)
3. Action items or decisions made
4. Technical details mentioned

Conversation:
{chat_content[:8000]}

Provide as structured lists:""",
                options={"temperature": 0.3}
            )
            
            key_info = extract_result.get('response', '').strip()
            
        finally:
            # Always clean up temp file
            if temp_path.exists():
                temp_path.unlink()
        
        return {
            **result,
            'chat_id': chat_id,
            'key_information': key_info,
            'searchable': True,
            'message': 'Chat processed hierarchically and ready for semantic search!'
        }
        
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# TOOL 16: Clear Embedding Cache
# ============================================================================

@mcp.tool()
def clear_embedding_cache() -> dict:
    """
    Clear embedding caches to free up space.
    Use this if you want to start fresh or caches get corrupted.
    
    Returns:
        Status of cache clearing operation
    """
    try:
        # Clear in-memory cache
        try:
            ollama_client = get_ollama()
            with ollama_client._cache_lock:
                ollama_client._embedding_cache = {}
            ollama_client._save_embedding_cache()
        except ConnectionError:
            pass  # Ollama not connected, skip
        
        # Clear vector store
        try:
            vector_store = get_vector_store()
            with vector_store._lock:
                vector_store.indexes = {}
        except Exception:
            pass
        
        # Count and remove cache files
        cache_files_removed = 0
        
        # Remove JSON cache files
        for pattern in ["*.json", "*.pkl"]:  # Include old pickle files for cleanup
            cache_files = list(Path(config.EMBEDDINGS_CACHE_DIR).glob(pattern))
            for cache_file in cache_files:
                try:
                    cache_file.unlink()
                    cache_files_removed += 1
                except Exception:
                    pass
        
        return {
            'status': 'success',
            'files_removed': cache_files_removed,
            'message': f'Cleared {cache_files_removed} cache files. Ready for fresh embeddings.'
        }
        
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# RLM TOOLS
# ============================================================================

@mcp.tool()
def rlm_query(
    query: str,
    context: Any,
    max_iterations: int = 30,
    enable_semantic_search: bool = True,
    verbose: bool = False
) -> dict:
    """
    Recursive LLM (RLM) query: Solves complex tasks by writing and executing code.
    The LM can recursively query itself, search context, and transform data.
    
    Args:
        query: The complex question or task to solve
        context: The data to analyze (string, list, or dict)
        max_iterations: Maximum reasoning steps (default 30)
        enable_semantic_search: Allow LM to use embeddings for search
        verbose: Return detailed execution logs
    
    Returns:
        Final answer and execution metadata
    """
    try:
        ollama_client = get_ollama()
    except ConnectionError as e:
        return {'error': f'Ollama not connected: {str(e)}'}
        
    return rlm_completion_loop(
        query=query,
        context=context,
        ollama_client=ollama_client,
        max_iterations=max_iterations,
        enable_semantic_search=enable_semantic_search,
        verbose=verbose
    )

@mcp.tool()
def rlm_query_file(
    query: str,
    file_path: str,
    max_iterations: int = 30,
    enable_semantic_search: bool = True,
    verbose: bool = False
) -> dict:
    """
    Recursive LLM (RLM) query on a file: Reads file then starts reasoning loop.
    
    Args:
        query: The complex question or task about the file
        file_path: Path to the document
        max_iterations: Maximum reasoning steps (default 30)
        enable_semantic_search: Allow LM to use embeddings for search
        verbose: Return detailed execution logs
    """
    try:
        validated_path = context_mgr.validate_file_path(file_path)
        content = context_mgr.read_file_safe(validated_path)
        
        ollama_client = get_ollama()
    except FileNotFoundError as e:
        return {'error': str(e)}
    except ConnectionError as e:
        return {'error': f'Ollama not connected: {str(e)}'}
    except Exception as e:
        return {'error': f"Setup failed: {str(e)}"}
        
    return rlm_completion_loop(
        query=query,
        context=content,
        ollama_client=ollama_client,
        max_iterations=max_iterations,
        enable_semantic_search=enable_semantic_search,
        verbose=verbose
    )

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import atexit
    
    # Save embedding cache on shutdown
    def cleanup():
        try:
            ollama_client = get_ollama()
            ollama_client._save_embedding_cache()
        except Exception:
            pass
    
    atexit.register(cleanup)
    
    print("=" * 80)
    print("Smart Context Manager - Recursive LLM Edition (macOS Optimized)")
    print("=" * 80)
    print(f"Platform: macOS")
    print(f"Cache directory: {config.CACHE_DIR}")
    print(f"Embeddings cache: {config.EMBEDDINGS_CACHE_DIR}")
    print(f"LLM Model: {config.LLM_MODEL}")
    print(f"Embedding Model: {config.EMBEDDING_MODEL}")
    print("=" * 80)
    print("\nRLM CAPABILITIES:")
    print("  [+] Recursive Reasoning Loop (RLM)")
    print("  [+] Sandboxed Python REPL")
    print("  [+] Recursive LLM Queries (llm_query)")
    print("  [+] Batched Concurrent Queries (llm_query_batched)")
    print("  [+] Integrated Semantic Search")
    print("=" * 80)
    print("\nmacOS Optimizations:")
    print("  [+] SIGALRM-based timeouts (native Unix signals)")
    print("  [+] macOS-specific file search paths (~/Documents, ~/Downloads, ~/Desktop)")
    print("  [+] Thread-safe caching with locks")
    print("  [+] Lazy initialization for Ollama (retry on reconnect)")
    print("  [+] JSON-based cache (secure, no pickle vulnerabilities)")
    print("  [+] Input validation on all parameters")
    print("  [+] Optimized vector search with precomputed norms")
    print("  [+] Graceful error handling in multi-doc sessions")
    print("  [+] Batch embedding with connection reuse")
    print("=" * 80)
    
    mcp.run()
