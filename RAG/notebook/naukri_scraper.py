import nest_asyncio
import asyncio
from langchain_community.document_loaders import PlaywrightURLLoader

# Patch the running loop so we can use await inside Jupyter
nest_asyncio.apply()

async def main():
    page_url = "https://www.investorgain.com/report/live-ipo-gmp/331/ipo/"
    loader = PlaywrightURLLoader(
        urls=[page_url],
        remove_selectors=["script", "style"],
        headless=True
    )
    docs = await loader.aload()  # async load
    return docs 

# Get the running loop and run the async function
loop = asyncio.get_event_loop()
loop.run_until_complete(main())
