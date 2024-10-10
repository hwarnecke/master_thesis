import json

import requests
from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
import re
import tqdm

class ServiceScraper:
    """
    This is a class to scrape the service.osnabrueck.de website.
    It is highly specific to the structure of the website.
    It focuses on the actual Services and Contacts listed on the Dienstleistungen A-Z page.
    It also extracts some metadata from the tags of the list entries and the content of the pages.
    """

    def __CollectServiceURLs(self,
                             url: str = "https://service.osnabrueck.de/dienstleistungen?search=&kategorie="
                             ) -> dict[str,any]:
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


    def __CollectServiceInformation(self, url: str) -> dict[str,any]:
        """
        Collects the text from the service page and cuts some unnecessary parts.
        :param url: String - the URL of the service page
        :return: content: Dict - contains text and metadata of each page
        """
        server = requests.get(url, timeout=3)
        soup = BeautifulSoup(server.text, 'html.parser')
        content: str = soup.find('div', attrs={'class': 'service-detail'}).get_text()

        # everything after the 'Amt/Fachbereich' section is poorly structured for plain text extraction, so we cut it
        cutoff = content.find('Amt/Fachbereich')
        # but we do need to include the contact person in the text otherwise some multi-hop question do not make sense
        # so we retrieve that information, rewrite it slightly and append to the rest of the page
        additional_information: str = self.__AppendContactData(soup)

        # but we still want to get the Fachbereich as metadata
        last_part = content[cutoff:]
        fachbereich = self.__CollectFachbereichData(last_part)

        # most services have a contact associated with it, we want to leverage that as metadata
        contact_data = self.__CollectContactData(soup)

        # combine the two dictionaries into one and add the text as a value
        metadata = dict(fachbereich, **contact_data)
        content = content[:cutoff] + additional_information
        content_dict = {"text": content}
        content_dict.update(metadata)

        return content_dict

    def __AppendContactData(self, soup) -> str:
        """
        it is important that the contact data is part of the text, otherwise the multi-hop questions don't make sense.
        This will extract the information if possible and format it accordingly.
        Other than the __CollectContactData, this will get all the entries for people and institutions.
        :param soup:
        :return:
        """
        # there are different types of contacts: people, institutions and similar services. I don't need the latter
        contact_people = soup.findAll('div', attrs={'class': 'kontaktperson-name'})
        contact_institution = soup.find('div', attrs={'class': 'kontakteinrichtung-name'})

        additional_text: str = ""
        # create a fitting string for the institution
        if contact_institution:
            additional_text = "\nZuständige Einrichtung: " + contact_institution.get_text()

        # there might be multiple people and some might have associated roles
        for person in contact_people:
            position_div = person.find('span', attrs={'class': 'kontaktperson-position'})
            if position_div:
                position = position_div.get_text()
            # the name comes last, this will find the name even if there is no position
            name = person.findAll('span')[-1].get_text()
            name = re.sub(r'\s+', ' ', name)  # somehow weird whitespaces are added to the names
            if position_div:
                person_text = "Zuständig als " + position + ": " + name
            else:
                person_text = "Zuständig: " + name

            additional_text = additional_text + "\n" + person_text

        return additional_text


    def __CollectContactData(self, soup) -> dict[str,str]:
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



    def __CollectFachbereichData(self, text: str) -> dict[str,str]:
        """
        collect the information to which 'Fachbereich' the service belongs
        :param text: content of the page
        :return: Dict - the Fachbereich data
        """
        pattern = r'FB\d{2}'
        match = re.search(pattern, text)
        Fachbereich = ""
        if match is not None:
            Fachbereich = match.group()
        return {'Fachbereich': Fachbereich}

    def __CreateServiceNodes(self, services:dict, chunk_size: int = 512) -> list:
        """
        Creates TextNodes from the text of the services and adds the respective metadata.
        :param services: Dict - the dictionary of all the services
        :return: List of TextNodes
        """
        all_nodes=[]
        for service_name, service_data in tqdm.tqdm(services.items()):

            # Create TextNodes
            splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=20)
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


    def __CollectContactURLs(self, services: dict) -> dict[str,dict[str,str]]:
        """
        Collects all links that point towards institutions or contact persons
        :param services: the urls of the services
        :return: a dict containing the name as key and the metadata(href and position) as values
        """
        contacts = {}
        for service_name, service_data in services.items():

            url = service_data['href']
            server = requests.get(url, timeout=3)
            soup = BeautifulSoup(server.text, 'html.parser')
            contact_divs = soup.find_all('div', attrs={'class': 'contact'})

            links: list[str] = []
            for div in contact_divs:
                # each new contact is in its own list
                # with a 'span' containing the name and an 'a' for the link
                lis = div.find_all('li')
                for li in lis:
                    spans = li.find_all('span')
                    a_tag = li.find('a')  # the url is always before the phone number

                    # some 'Kontaktpersonen' have their position specified.
                    # if so, they have to spans instead of one
                    if len(spans) == 1:
                        name: str = spans[0].text
                        position: str = None
                    else:
                        name: str = spans[1].text
                        position: str = spans[0].text

                    href: str = a_tag['href']

                    # exclusion criteria are if it points towards a 'dienstleistung' or if we already visited it
                    function = "einrichtung" if "einrichtung" in href else "mitarbeiter" if "mitarbeiter" in href else None
                    not_visited = href not in links

                    if function and not_visited:
                        links.append(href)
                        contacts[name] = {
                            "href": href,
                            "position": position,
                            "typ": function
                        }
        return contacts

    def __CollectContactInformation(self, url: str) -> dict[str,any]:
        """
        collecting the content and metadata for the given URL
        this is specific for the 'Einrichtungen' or 'Mitarbeiter' (so contacts instead of services)
        :param url: the url to extract from
        :return: the dict containing the text and the metadata, with the name of the contact as key
        """
        server = requests.get(url, timeout=3)
        soup = BeautifulSoup(server.text, 'html.parser')
        content = soup.find('div', attrs={'class': 'col-12'}).get_text()
        info = {"text": content}
        metadata: dict = self.__ExtractRelatedServices(soup)
        # TODO: I could extract more metadata if wanted:
        #       which 'Einrichtung' a person works for
        #       which 'Einrichtung' another 'Einrichtung' is connected to
        info.update(metadata)
        return info

    def __ExtractRelatedServices(self, soup) -> dict[str,list]:
        """
        extracts the information about related services if there are any
        :param soup: the soup object for that site
        :return: {'Dienstleistung': list}
        """
        div = soup.find('div', attrs={'id': 'allservices'})
        related_services: list[str] = []
        if div:
            services = div.find_all('li')
            for service in services:
                name: str = service.find('a').get_text()
                related_services.append(name)
        return {"Dienstleistungen": related_services}


    def __CreateContactNodes(self, contacts:dict, chunk_size: int = 512) -> list:
        """
        creating nodes for the contact data.
        It differs from the service nodes only in the type of metadata that is saved
        :param contacts: the dict containing all the information about the contacts
        :param chunk_size: how small to split the text nodes
        :return: the list of TextNodes
        """
        all_nodes = []
        for contact_name, contact_data in tqdm.tqdm(contacts.items()):

            # Create TextNodes
            splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=20)
            text_nodes = splitter.get_nodes_from_documents([Document(text=contact_data['text'])])

            for node in text_nodes:
                node.metadata = {
                    'Typ': contact_data['typ'],
                    'Name': contact_name,
                    'URL': contact_data['href'],
                    'Dienstleistungen': contact_data['Dienstleistungen']
                }
            all_nodes.extend(text_nodes)

        return all_nodes

    def __SaveToDisc(self, path: str, content: dict):
        with open(path, 'w') as file:
            json.dump(content, file)


    def ScrapeServicePage(self, url: str = "https://service.osnabrueck.de/dienstleistungen?search=&kategorie=",
                          chunk_size: int = 512, load_from_disc: bool = False, service_path: str = "service.json",
                          contact_path: str = "contact.json"):
        """
        Scrapes the service.osnabrueck.de website.
        Collects the URLs of the services from the Dienstleistungen A-Z page.
        Collects the text from the service pages.
        Collects the contact data from the service pages.
        Creates TextNodes from the text of the services and adds the respective metadata.
        :param url: String - the URL of the Dienstleistungen A-Z page
        :param chunk_size: The chunk size of the nodes that are created. Some embeddings might not work with 1024.
        :return: List of TextNodes
        """
        if not load_from_disc:
            print("\n01/03: Fetching URLs from: " + url)
            services: dict[str,dict] = self.__CollectServiceURLs(url)
            print("\nFound " + str(len(services)) + " URLs.")

            print("\nFetching contact URLs")
            contacts: dict[str,dict] = self.__CollectContactURLs(services)
            print("\nFound: " + str(len(contacts)) + " URLs.")

            print("\n02/03: Load Data from URLs...")
            for service_name, service_data in tqdm.tqdm(services.items()):
                content = self.__CollectServiceInformation(service_data['href'])
                services[service_name].update(content)

            for contact_name, contact_data in tqdm.tqdm(contacts.items()):
                content = self.__CollectContactInformation(contact_data['href'])
                contacts[contact_name].update(content)

            # save the data to disc, so I can load it later instead of having to scrape everything again
            # I wanted to save the splitted version already but TextNodes don't seem serializable
            self.__SaveToDisc("service.json", services)
            self.__SaveToDisc("contact.json", contacts)
            print("\nData successfully saved.")

        else:
            print("\nSkipping Webscraper, Loading files from Disc.")
            with open(service_path, 'r') as file:
                services = json.load(file)
            with open(contact_path, 'r') as file:
                contacts = json.load(file)

        print("\n03/03: Creating TextNodes...")
        service_nodes: list = self.__CreateServiceNodes(services, chunk_size=chunk_size)
        contact_nodes: list = self.__CreateContactNodes(contacts, chunk_size=chunk_size)
        nodes: list = service_nodes + contact_nodes
        print("\nCreated " + str(len(nodes)) + " TextNodes")

        return nodes


if __name__ == "__main__":
    scraper = ServiceScraper()
    text_nodes = scraper.ScrapeServicePage(load_from_disc=True)
    print(text_nodes[0])