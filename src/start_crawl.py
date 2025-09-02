import os
import json
from dotenv import load_dotenv
from firecrawl import Firecrawl

load_dotenv()

FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY")
firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)

# Start crawl
STATE = "CT"
SEED_URL = "https://portal.ct.gov/dds"
LIMIT = 100

crawl_job = firecrawl.start_crawl(
    url=SEED_URL, 
    limit=LIMIT,
    max_discovery_depth=4,
    exclude_paths=[
        ".*\\.jpg$", ".*\\.png$", ".*\\.gif$", ".*\\.jpeg$"  # Skip image files
    ]
)
job_id = crawl_job.id if hasattr(crawl_job, 'id') else str(crawl_job)

# Save job info
jobs = {
    "state": STATE,
    "seed_url": SEED_URL,
    "job_id": job_id,
    "limit": LIMIT
}

with open("crawl_jobs.json", 'w') as f:
    json.dump(jobs, f, indent=2)

print(f"Started crawl. Job ID: {job_id}")

