import os, sys, uuid, time
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from util import get_db, fc_session, FIRECRAWL_BASE, chunk_markdown, embed_batch

load_dotenv()
STATE = os.environ.get("STATE", "NJ")

def _clean(s):
    if s is None:
        return None
    # Remove NUL bytes which Postgres text/varchar cannot store
    return str(s).replace("\x00", "")

def fetch_all(job_id: str):
    s = fc_session()
    next_url = f"{FIRECRAWL_BASE}/crawl/{job_id}"
    all_rows = []
    while next_url:
        r = s.get(next_url, timeout=90)
        r.raise_for_status()
        payload = r.json()
        data = payload.get("data") or []
        all_rows.extend(data)
        status = payload.get("status")
        next_url = payload.get("next")
        completed = payload.get("completed"); total = payload.get("total")
        if completed and total:
            print(f"[firecrawl] {completed}/{total} • accumulated {len(all_rows)}")
        if status != "completed" and not next_url:
            time.sleep(2.0)
            next_url = f"{FIRECRAWL_BASE}/crawl/{job_id}"
    return all_rows

def upsert(rows):
    con = get_db()
    docs = []
    chunks = []

    print(f"[upsert] Starting ingestion for {len(rows)} rows")

    for idx, row in enumerate(rows, 1):
        md = _clean((row.get("markdown") or "").strip())
        meta = row.get("metadata") or {}
        url = _clean(meta.get("sourceURL") or meta.get("ogUrl"))
        if not url:
            continue
        title = _clean(meta.get("title"))
        doc_type = _clean("pdf" if url.lower().endswith(".pdf") else "html")
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, url))

        docs.append((doc_id, url, title, STATE, doc_type))

        if not md:
            continue

        chunks_list = chunk_markdown(md, target=900, overlap=100)
        if not chunks_list:
            continue

        B = 16
        for i in range(0, len(chunks_list), B):
            batch = chunks_list[i:i+B]
            print(f"[embed] Doc {idx}/{len(rows)} • embedding batch {i//B+1}/{(len(chunks_list)+B-1)//B}")
            vecs = embed_batch([c["text"] for c in batch])
            for j, (c, emb) in enumerate(zip(batch, vecs)):
                chunk_id = f"{i+j:06d}"
                text = _clean(c.get("text"))
                heading = _clean(c.get("heading"))
                page_num = c.get("page_num")
                chunks.append((doc_id, chunk_id, i+j, text, page_num, heading, emb))

        if idx % 50 == 0:
            print(f"[progress] Processed {idx}/{len(rows)} docs so far…")

    with con, con.cursor() as cur:
        if docs:
            print(f"[db] Inserting {len(docs)} documents")
            execute_values(cur, """
                INSERT INTO documents (doc_id, url, title, state, doc_type)
                VALUES %s
                ON CONFLICT (doc_id) DO UPDATE
                SET title=EXCLUDED.title,
                    state=EXCLUDED.state,
                    doc_type=EXCLUDED.doc_type;
            """, docs, page_size=500)

        if chunks:
            print(f"[db] Inserting {len(chunks)} chunks")
            execute_values(cur, """
                INSERT INTO chunks (doc_id, chunk_id, seq, text, page_num, heading, embedding)
                VALUES %s
                ON CONFLICT (doc_id, chunk_id) DO UPDATE
                SET text=EXCLUDED.text,
                    page_num=EXCLUDED.page_num,
                    heading=EXCLUDED.heading,
                    embedding=EXCLUDED.embedding;
            """, chunks, page_size=500)

    print("[upsert] Finished inserts")

def main():
    job_id = "c34c12b5-e21d-46fe-bdb2-094e46ab1c10"
    rows = fetch_all(job_id)
    print(f"Fetched {len(rows)} items. Ingesting…")
    upsert(rows)
    print("Done.")

if __name__ == "__main__":
    main()
