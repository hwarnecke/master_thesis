import requests
from bs4 import BeautifulSoup
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
import re

"""
Some specific tests for scraping the service.osnabrueck.de website.
It seems like most things of interest for me are reachable from the 
Dienstleistungen A-Z page.

Also some metadata can be extracted from the tags of the list entries.
"""
def main():
    url = "https://service.osnabrueck.de/dienstleistungen?search=&kategorie="

    server = requests.get(url, timeout=3)

    soup = BeautifulSoup(server.text, 'html.parser')

    # find all list entries that point towards a service
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

    print(len(services))

    # scrape data and create documents

    all_nodes = []
    all_contact_urls = []
    for service_name, service_data in services.items():
        if service_data['href'] is not None:
            server = requests.get(service_data['href'], timeout=3)
            soup = BeautifulSoup(server.text, 'html.parser')
            # content = soup.get_text() # this would get the text of the whole page which is too much
            content = soup.find('div', attrs={'class': 'service-detail'}).get_text()

            # we don't need anything after the 'Amt/Fachbereich' section
            # but we can use the Fachbereich as metadata
            cutoff = content.find('Amt/Fachbereich')
            last_part = content[cutoff:]
            pattern = r'FB\d{2}'
            match = re.search(pattern, last_part)
            Fachbereich = ""
            if match is not None:
                Fachbereich = match.group()

            # we should also get the Kontakt as metadata
            # I guess scraping the actual contact page would also be a good idea
            contact_div = soup.find('div', attrs={'class': 'contact'})
            contact_name = ""
            contact_url = ""
            name = None

            # this is the ugliest type check I have ever written
            if contact_div is not None:
                name = contact_div.find('span')
                url = contact_div.find('a', attrs={'title': True})

            if name is not None:
                contact_name = name.get_text()

            if url is not None:
                contact_url = url['href']
                all_contact_urls.append(contact_url)

            # remove unnecessary rest of the page
            content = content[:cutoff]

            splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
            text_nodes = splitter.get_nodes_from_documents([Document(text=content)])
            for node in text_nodes:
                node.metadata = {
                    'Typ': 'Dienstleistung',
                    'URL': service_data['href'],
                    'Name': service_name,
                    'Kategorie': service_data['data-kategorie'],
                    'Buchstabe': service_data['data-letter'],
                    'Synonyme': service_data['data-synonyme'],
                    'Fachbereich': Fachbereich,
                    'Kontakt': contact_name,
                    'Kontakt URL': contact_url
                }
            all_nodes.extend(text_nodes)


    print (len(all_nodes))

def additional(url):
    server = requests.get(url, timeout=3)
    soup = BeautifulSoup(server.text, 'html.parser')
    #content = soup.find('div', attrs={'class': 'service-detail'}).get_text()
    #cutoff = content.find('Amt/Fachbereich')
    #last_part = content[cutoff:]
    contact = appendContactData(soup)
    #print(contact)


def __CollectContactData(soup) -> dict[str,str]:
    """
    Collects the contact data from the service page.
    :param soup: BeautifulSoup - the soup object of the service page
    :return: Dict - the contact data
    """
    #contact_divs = soup.findAll('div', attrs={'class': 'contact'})
    contact_divs = soup.findAll('div', attrs={'class': 'kontaktperson-name'})
    #print(contact_divs)
    names = []
    for contact_div in contact_divs:
        contact_name = ""
        contact_url = ""
        name = None
        url = None

        if contact_div is not None:
            #names = contact
            name = contact_div.findAll('span')
            url = contact_div.find('a', attrs={'title': True})
            category = contact_div.find('h2', attrs={'class': "h3"})
            category_text = category.get_text().replace(" ", "")
            #print(category_text)


        if name is not None and (category_text == "Kontakt" or category_text == "Kontaktpersonen"):
            contact_name = name[-1].get_text()
            contact_name = re.sub(r'\s+', ' ', contact_name)
            print(contact_name)
            name = category_text + ":" + " " + contact_name
            names.append(name)

        if url is not None:
            contact_url = url['href']

    print(names)
    #return {'Kontakt': contact_name, 'Kontakt URL': contact_url}


def appendContactData(soup) -> str:
    """
    it is important that the contact data is part of the text, otherwise the multi-hop questions don't make sense.
    This will extract the information if possible and format it accordingly
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
        name = re.sub(r'\s+', ' ', name) # somehow weird whitespaces are added to the names
        if position_div:
            person_text = "Zuständig als " + position + ": " + name
        else:
            person_text = "Zuständig: " + name

        additional_text = additional_text + "\n" + person_text

    print(additional_text)
    return additional_text


def re_tests():
    contact_name = "T H"
    contact_name = re.sub(r'\s+', '', contact_name)
    print(contact_name)

if __name__ == "__main__":
    #additional(url = "https://service.osnabrueck.de/dienstleistungen/-/egov-bis-detail/dienstleistung/4157/show")
    additional("https://service.osnabrueck.de/dienstleistungen/-/egov-bis-detail/dienstleistung/3630/show")
    #additional("https://service.osnabrueck.de/dienstleistungen/-/egov-bis-detail/einrichtung/6205/show")