import nest_asyncio
import asyncio
from langchain_community.document_loaders import PlaywrightURLLoader

# Patch Jupyter's running loop
nest_asyncio.apply()

async def fetch_docs(url):
    loader = PlaywrightURLLoader(
        urls=[url],
        remove_selectors=["script", "style"],
        headless=True
    )
    docs = await loader.aload()
    return docs

# Helper to call async function in notebook
def get_docs(url):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(fetch_docs(url))


