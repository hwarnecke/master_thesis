import requests
from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
import re
import tqdm

class ServiceScraper():
    """
    This is a class to scrape the service.osnabrueck.de website.
    It is highly specific to the structure of the website.
    It focusses on the actual Services listed on the Dienstleistungen A-Z page.
    It also extracts some metadata from the tags of the list entries.
    """

    def __CollectServiceURLs(self, url = "https://service.osnabrueck.de/dienstleistungen?search=&kategorie="):
        """
        Collects all the URLs of the services from the Dienstleistungen A-Z page.
        together with some metadata.
        :param url: String - the URL of the Dienstleistungen A-Z page
        :return: Dict - a dictionary with the service names as keys and the metadata as values
        The values are the url to the service, the category, the starting letter and some synonyms
        """
        server = requests.get(url, timeout=3)
        soup = BeautifulSoup(server.text, 'html.parser')
        list_entries = soup.find_all('li', attrs={'class': 'list-group-item dlitem'})

        services = {}
        for entry in list_entries:
            a_tag = entry.find('a')
            service_name = a_tag.text
            services[service_name] = {
                'href': a_tag['href'] if 'href' in a_tag.attrs else None,
                'data-kategorie': entry['data-kategorie'] if 'data-kategorie' in entry.attrs else None,
                'data-letter': entry['data-letter'] if 'data-letter' in entry.attrs else None,
                'data-synonyme': [synonym.strip() for synonym in
                                  entry['data-synonyme'].split(',')] if 'data-synonyme' in entry.attrs else None
            }

        return services


    def __CollectServiceInformation(self, url):
        """
        Collects the text from the service page and cuts some unnecessary parts.
        :param URL: String - the URL of the service page
        :return: content: Dict - contains text and metadata of each page
        """
        server = requests.get(url, timeout=3)
        soup = BeautifulSoup(server.text, 'html.parser')
        content = soup.find('div', attrs={'class': 'service-detail'}).get_text()

        # we don't need the text after the 'Amt/Fachbereich' section
        # but we still want to get the Fachbereich as metadata
        cutoff = content.find('Amt/Fachbereich')
        last_part = content[cutoff:]
        fachbereich = self.__CollectFachbereichData(last_part)

        # most services have a contact associated with it, we want to leverage that as metadata
        contact_data = self.__CollectContactData(soup)

        # combine the two dictionaries into one and add the text as a value
        metadata = dict(fachbereich, **contact_data)
        content = {"text": content[:cutoff]}
        content.update(metadata)

        return content


    def __CollectContactData(self, soup):
        """
        Collects the contact data from the service page.
        :param soup: BeautifulSoup - the soup object of the service page
        :return: Dict - the contact data
        """
        contact_div = soup.find('div', attrs={'class': 'contact'})
        contact_name = ""
        contact_url = ""
        name = None
        url = None

        # I know this is not the nicest way of
        if contact_div is not None:
            name = contact_div.find('span')
            url = contact_div.find('a', attrs={'title': True})

        if name is not None:
            contact_name = name.get_text()

        if url is not None:
            contact_url = url['href']

        return {'Kontakt': contact_name, 'Kontakt URL': contact_url}



    def __CollectFachbereichData(self, text):
        """

        :param text:
        :return: Dict - the Fachbereich data
        """
        pattern = r'FB\d{2}'
        match = re.search(pattern, text)
        Fachbereich = ""
        if match is not None:
            Fachbereich = match.group()
        return {'Fachbereich': Fachbereich}

    def __CreateNodes(self, services):
        """
        Creates TextNodes from the text of the services and adds the respective metadata.
        :param services: Dict - the dictionary of all the services
        :return: List of TextNodes
        """
        all_nodes=[]
        for service_name, service_data in tqdm.tqdm(services.items()):

            # Create TextNodes
            splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
            text_nodes = splitter.get_nodes_from_documents([Document(text=service_data['text'])])

            for node in text_nodes:
                node.metadata = {
                    'Typ': 'Dienstleistung',
                    'Name': service_name,
                    'URL': service_data['href'],
                    'Kategorie': service_data['data-kategorie'],
                    'Anfangsbuchstabe': service_data['data-letter'],
                    'Synonyme': service_data['data-synonyme'],
                    'Fachbereich': service_data['Fachbereich'],
                    'Kontakt': service_data['Kontakt'],
                    'Kontakt URL': service_data['Kontakt URL']
                    }
            all_nodes.extend(text_nodes)

        return all_nodes





    def ScrapeServicePage(self, url = "https://service.osnabrueck.de/dienstleistungen?search=&kategorie="):
        """
        Scrapes the service.osnabrueck.de website.
        Collects the URLs of the services from the Dienstleistungen A-Z page.
        Collects the text from the service pages.
        Collects the contact data from the service pages.
        Creates TextNodes from the text of the services and adds the respective metadata.
        :param url: String - the URL of the Dienstleistungen A-Z page
        :return: List of TextNodes
        """
        print("\n01/03: Fetching URLs from: " + url)
        services = self.__CollectServiceURLs(url)
        print("\nFound " + str(len(services)) + " URLs")

        print("\n02/03: Load Data from URLs...")
        for service_name, service_data in tqdm.tqdm(services.items()):
            content = self.__CollectServiceInformation(service_data['href'])
            services[service_name].update(content)

        print("\n03/03: Creating TextNodes...")
        text_nodes = self.__CreateNodes(services)
        print("\nCreated " + str(len(text_nodes)) + " TextNodes")

        return text_nodes


# if __name__ == "__main__":
#     scraper = ServiceScraper()
#     text_nodes = scraper.ScrapeServicePage()
#     print(text_nodes[0])