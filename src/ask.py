import os, sys
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client

load_dotenv()
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_ANON_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)
STATE = "CT"

SYS = (
    "You are SAMBOT. Answer strictly from the provided passages. "
    "Be concise. Do NOT generate a Sources section — the system will add one."
)

def rewrite_query(q: str) -> str:
    """Use GPT to rewrite the query into a clearer search query for compliance/provider manuals."""
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Rewrite the user's query into a precise, domain-specific search query for regulatory/provider manuals. Avoid answering; only rewrite."},
            {"role": "user", "content": q}
        ],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

def embed(q: str):
    return client.embeddings.create(
        model="text-embedding-3-small", input=q
    ).data[0].embedding

def retrieve(vec, state=STATE, k=24):
    # Try vector search first
    try:
        result = supabase.rpc('match_chunks', {
            'query_embedding': vec,
            'match_state': state,
            'match_count': k
        }).execute()
        
        if result.data and len(result.data) > 0:
            rows = []
            for chunk in result.data:
                rows.append((
                    chunk.get('url', ''),
                    chunk.get('title', ''),
                    chunk.get('text', ''),
                    chunk.get('page_num')
                ))
            return rows
    except:
        pass
    
    # Fallback to regular query
    result = supabase.table('chunks') \
        .select('text, page_num, heading, documents!inner(url, title)') \
        .eq('documents.state', state) \
        .limit(k) \
        .execute()
    
    rows = []
    for chunk in result.data:
        doc = chunk.get('documents', {})
        rows.append((
            doc.get('url', ''),
            doc.get('title', ''),
            chunk.get('text', ''),
            chunk.get('page_num')
        ))
    
    return rows

def answer(q: str, state=STATE):
    # Step 1: Rewrite
    print(f"Original query: {q}")
    rewritten = rewrite_query(q)
    print(f"New query: {rewritten}")

    # Step 2: Embed rewritten query
    print("Vectorizing...")
    v = embed(rewritten)

    # Step 3: Retrieve context
    rows = retrieve(v, state, 24)

    ctx, cites = [], []
    for (url, title, text, page) in rows[:8]:
        ctx.append(f"[{title}] {text}")
        cite = f"{url}#page={page}" if page else url
        if cite not in cites:
            cites.append(cite)

    # Step 4: Ask LLM for the answer only
    prompt = (
        f"{SYS}\n\nQuestion: {q}\n\nOptimized Query: {rewritten}\n\nPassages:\n" +
        "\n\n---\n\n".join(ctx[:6]) +
        "\n\nWrite only the answer."
    )

    print("Sending to LLM")
    chat = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    out = chat.choices[0].message.content.strip()

    # Step 5: Append deterministic sources
    if cites:
        out += "\n\nSources:\n" + "\n".join(f"- {u}" for u in cites[:4])
    else:
        out += "\n\n(Sources not available)"
    return out

if __name__ == "__main__":
    q = "What training is required for providers?" if len(sys.argv) < 2 else " ".join(sys.argv[1:])
    print(answer(q))
