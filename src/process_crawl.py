import os
import uuid
import re
from dotenv import load_dotenv
from firecrawl import Firecrawl
from supabase import create_client, Client
from openai import OpenAI
import tiktoken

load_dotenv()

# PASTE JOB ID HERE
JOB_ID = "f3189288-ce7c-472e-9d99-bbb8ee2aa591"
STATE = "CT"

FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Chunk size for processing documents (in tokens)
MAX_CHUNK_SIZE = 1200

firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
tokenizer = tiktoken.get_encoding("cl100k_base")

def clean_text(text: str) -> str:
    """Clean text by removing null bytes and other problematic characters."""
    if not text:
        return ""
    # Remove null bytes and other control characters except newlines and tabs
    cleaned = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')
    return cleaned.strip()

def chunk_markdown(markdown: str) -> list[dict]:
    chunks = []
    header_pattern = r'^(#{1,3}\s+.+)$'
    sections = re.split(header_pattern, markdown, flags=re.MULTILINE)
    
    current_heading = None
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        if re.match(header_pattern, section):
            current_heading = section.lstrip('# ').strip()
            continue
            
        if len(tokenizer.encode(section)) <= MAX_CHUNK_SIZE:
            chunks.append({
                'text': clean_text(section),
                'heading': clean_text(current_heading) if current_heading else None,
                'page_num': None
            })
        else:
            # Split by paragraphs if too big
            paragraphs = section.split('\n\n')
            current_chunk = ""
            
            for para in paragraphs:
                test_chunk = current_chunk + '\n\n' + para if current_chunk else para
                
                if len(tokenizer.encode(test_chunk)) <= MAX_CHUNK_SIZE:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append({
                            'text': clean_text(current_chunk.strip()),
                            'heading': clean_text(current_heading) if current_heading else None,
                            'page_num': None
                        })
                    current_chunk = para
            
            if current_chunk:
                chunks.append({
                    'text': clean_text(current_chunk.strip()),
                    'heading': clean_text(current_heading) if current_heading else None,
                    'page_num': None
                })
    
    return chunks

def embed_texts_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in a single API call for efficiency."""
    if not texts:
        return []
    
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        if "maximum context length" in str(e):
            print(f"[embed] Batch too large ({len(texts)} texts), falling back to individual embedding")
            # Fall back to individual embedding for each text
            embeddings = []
            for text in texts:
                try:
                    response = openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=[text]
                    )
                    embeddings.append(response.data[0].embedding)
                except Exception as single_error:
                    print(f"[embed] Failed to embed single text: {str(single_error)[:100]}...")
                    # Use a zero vector as fallback
                    embeddings.append([0.0] * 1536)
            return embeddings
        else:
            raise e


if __name__ == "__main__":
    # Get crawl results
    status = firecrawl.get_crawl_status(JOB_ID)
    if status.status != 'completed':
        print(f"Crawl not complete: {status.status}")
        exit()

    pages = status.data
    print(f"Processing {len(pages)} pages")

    # Batch processing settings
    BATCH_SIZE = 100  # Process chunks in batches
    EMBEDDING_BATCH_SIZE = 10  # OpenAI embedding batch size (reduced to avoid token limits)

    # Collect all documents and chunks first
    all_documents = []
    all_chunks = []
    all_texts_to_embed = []

    print("Preparing documents and chunks...")
    for i, page in enumerate(pages):
        url = page.metadata.url
        title = page.metadata.title or ''
        markdown = page.markdown

        if not url or not markdown:
            continue

        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, url))
        doc_type = 'pdf' if url.lower().endswith('.pdf') else 'html'

        # Prepare document for batch insert
        all_documents.append({
            'doc_id': doc_id,
            'url': clean_text(url),
            'title': clean_text(title),
            'state': STATE,
            'doc_type': doc_type
        })

        # Chunk and prepare for batch processing
        chunks = chunk_markdown(markdown)

        for j, chunk in enumerate(chunks):
            chunk_data = {
                'doc_id': doc_id,
                'chunk_id': f"{j:06d}",
                'seq': j,
                'text': chunk['text'],
                'page_num': chunk['page_num'],
                'heading': chunk['heading']
            }
            all_chunks.append(chunk_data)
            all_texts_to_embed.append(chunk['text'])

        if (i + 1) % 10 == 0:
            print(f"Prepared {i+1}/{len(pages)} pages")

    print(f"Total: {len(all_documents)} documents, {len(all_chunks)} chunks")

    # Batch insert documents
    print("Inserting documents...")
    supabase.table('documents').upsert(all_documents).execute()
    print(f"✓ Inserted {len(all_documents)} documents")

    # Batch embed and insert chunks
    print("Embedding and inserting chunks...")
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_chunks = all_chunks[i:i + BATCH_SIZE]
        batch_texts = all_texts_to_embed[i:i + BATCH_SIZE]

        # Embed texts in smaller batches (OpenAI limit)
        embeddings = []
        for j in range(0, len(batch_texts), EMBEDDING_BATCH_SIZE):
            embed_batch = batch_texts[j:j + EMBEDDING_BATCH_SIZE]
            batch_embeddings = embed_texts_batch(embed_batch)
            embeddings.extend(batch_embeddings)

        # Add embeddings to chunk data
        for chunk, embedding in zip(batch_chunks, embeddings):
            chunk['embedding'] = embedding

        # Batch insert chunks
        supabase.table('chunks').upsert(batch_chunks).execute()

        print(f"✓ Processed {min(i + BATCH_SIZE, len(all_chunks))}/{len(all_chunks)} chunks")

    print("✓ Done! All documents and chunks processed.")
