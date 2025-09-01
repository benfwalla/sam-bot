import os, sys
import psycopg
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
DB = os.environ["SUPABASE_DB_URL"]
STATE = os.environ.get("STATE", "NJ")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYS = (
    "You are SAMBOT. Answer strictly from the provided passages. "
    "Be concise. Do NOT generate a Sources section — the system will add one."
)

def rewrite_query(q: str) -> str:
    """Use GPT to rewrite the query into a clearer search query for compliance/provider manuals."""
    print(f"[rewrite] Original query: {q}")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Rewrite the user's query into a precise, domain-specific search query for regulatory/provider manuals. Avoid answering; only rewrite."},
            {"role": "user", "content": q}
        ],
        temperature=0
    )
    rewritten = resp.choices[0].message.content.strip()
    print(f"[rewrite] Rewritten query: {rewritten}")
    return rewritten

def embed(q: str):
    print(f"[embed] Embedding query: {q[:60]}{'...' if len(q) > 60 else ''}")
    return client.embeddings.create(
        model="text-embedding-3-small", input=q
    ).data[0].embedding

def retrieve(vec, state=STATE, k=24):
    print(f"[db] Retrieving top {k} rows for state={state}")
    with psycopg.connect(DB) as con, con.cursor() as cur:
        cur.execute("""
          SELECT d.url,
                 COALESCE(d.title,''),
                 c.text,
                 c.page_num
          FROM chunks c
          JOIN documents d ON d.doc_id = c.doc_id
          WHERE d.state = %s
          ORDER BY c.embedding <=> %s::vector
          LIMIT %s
        """, (state, vec, k))
        rows = cur.fetchall()
        print(f"[db] Retrieved {len(rows)} rows")
        return rows

def answer(q: str, state=STATE):
    # Step 1: Rewrite
    rewritten = rewrite_query(q)

    # Step 2: Embed rewritten query
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

    print("[llm] Sending to GPT for final answer")
    chat = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": SYS},
                  {"role": "user", "content": prompt}],
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
