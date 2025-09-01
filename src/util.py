import os, re, time
import psycopg2
import requests
import tiktoken
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Embeddings: 1536-dim to fit IVFFlat index limits
EMB_MODEL = "text-embedding-3-small"
ENC = tiktoken.get_encoding("cl100k_base")

# --- Database ---
def get_db():
    """Open a new autocommit connection to Supabase/Postgres."""
    url = os.environ["SUPABASE_DB_URL"]
    # Disable server-side prepared statements to avoid collisions like
    # psycopg.errors.DuplicatePreparedStatement: prepared statement "_pg3_0" already exists
    return psycopg2.connect(url, prepare_threshold=None)


# --- Embeddings ---
def embed_batch(texts):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def tok_count(s):
    return len(ENC.encode(s))

# --- Chunking ---
def _sliding(text: str, target=900, overlap=100):
    toks = ENC.encode(text); step = max(1, target - overlap); out=[]
    for i in range(0, len(toks), step):
        window = toks[i:i+target]
        if not window: break
        out.append(ENC.decode(window))
        if i + target >= len(toks): break
    return out

def chunk_markdown(md: str, target=900, overlap=100):
    """Return list of dicts: {text, heading, page_num}."""
    # Try PDF 'Page N' markers first
    page_marks = [(m.start(), int(m.group(1))) for m in re.finditer(r"(?im)^\s*page\s+(\d+)\b", md)]
    if page_marks:
        chunks=[]; idxs=[i for i,_ in page_marks]+[len(md)]; pages=[p for _,p in page_marks]
        for k in range(len(pages)):
            seg = md[idxs[k]:idxs[k+1]].strip()
            for s in _sliding(seg, target, overlap):
                chunks.append({"text": s, "heading": None, "page_num": pages[k]})
        return chunks

    # Otherwise split by headings (#..###)
    parts = re.split(r"(?m)^(#{1,3}\s.*)$", md)
    chunks=[]
    pre = parts[0].strip()
    if pre:
        for s in _sliding(pre, target, overlap):
            chunks.append({"text": s, "heading": None, "page_num": None})
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = (parts[i+1] if i+1 < len(parts) else "").strip()
        sec = (heading + "\n\n" + body).strip()
        htxt = heading.lstrip("# ").strip()
        if tok_count(sec) <= target + 100:
            chunks.append({"text": sec, "heading": htxt, "page_num": None})
        else:
            for s in _sliding(sec, target, overlap):
                chunks.append({"text": s, "heading": htxt, "page_num": None})
    return chunks

# --- Firecrawl helpers ---
def fc_session():
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {os.environ['FIRECRAWL_API_KEY']}"})
    retry = Retry(
        total=8, read=8, connect=8,
        backoff_factor=0.8,
        status_forcelist=(429,500,502,503,504),
        allowed_methods=frozenset(["GET","POST"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

FIRECRAWL_BASE = "https://api.firecrawl.dev/v2"
