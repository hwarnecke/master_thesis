import requests
from bs4 import BeautifulSoup as bs
from GetURLs import get_urls
from llama_index.core import (
    VectorStoreIndex,
    Document,)
class WebCrawler:
    """
    Can be used to crawl a website and it's subpoges for data to fill a vector store with.
    Will create a VectorStoreIndex from the data and save it to a directory.
    Default is the service.osnabrueck.de website.
    """

    @staticmethod
    def crawl(URL="https://service.osnabrueck.de/", ignore=["anmeldung", "#", ".pdf", ".PDF", "de/de/"], PERSIST_DIR="./webcrawl_data"):
        """
        Crawl the starting URL and all subpages for data.
        The default settings are set to crawl the service.osnabrueck.de website with a few ignore parameters to avoid duplicate content.
        :param URL: String - the URL to start with
        :param ignore: List of Strings - URLs containing these strings will be ignored
        :param PERSIST_DIR: String - the directory where the Llamaindex VectorStoreIndex will be stored
        :return: None
        """

        print("Fetching URLs from: " + URL)
        # Get all links from the given URL
        links = get_urls(URL, ignore=ignore)
        print("Found " + str(len(links)) + " URLs")
        print("Load Data from URLs...")

        text_list = []
        for link in links:
            try:
                server = requests.get(link, timeout=3)
                soup = bs(server.text, 'html.parser')
                # the text should contain the URL and the content of the page
                content = soup.get_text()
                #combined = f"URL: {link}\nContent: {content}" # I experimented with adding the URL to the content but it didn't work well
                text_list.append(content)
            except:
                print("Error with URL: " + link)
                continue
        documents = [Document(text=t) for t in text_list]

        print("Creating Index...")
        index = VectorStoreIndex.from_documents(documents)
        print("Saving Index...")
        index.storage_context.persist(PERSIST_DIR)