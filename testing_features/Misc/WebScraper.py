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

url = "https://service.osnabrueck.de/dienstleistungen?search=&kategorie="

server = requests.get(url, timeout=3)

soup = BeautifulSoup(server.text, 'html.parser')

# find all list entries that point towards a service
list_entries = soup.find_all('li', attrs={'class': 'list-group-item dlitem'})

# # extract some metadata from the tags
# data_kategorie = []
# data_letter = []
# data_synonyme = []
#
# # extract the href and the name of each service
# service_links = []
# service_names = []
#
# for entry in list_entries:
#     data_kategorie.append(entry['data-kategorie'] if 'data-kategorie' in entry.attrs else None)
#     data_letter.append(entry['data-letter'] if 'data-letter' in entry.attrs else None)
#     data_synonyme.append(entry['data-synonyme'] if 'data-synonyme' in entry.attrs else None)
#
#     a_tag = entry.find('a')
#     service_links.append(a_tag['href'] if 'href' in a_tag.attrs else None)
#     service_names.append(a_tag.text)
#
#
# print(list_entries[0])
# print(service_links[0])
# print(service_names[0])


services = {}

for entry in list_entries:
    a_tag = entry.find('a')
    service_name = a_tag.text
    services[service_name] = {
        'href': a_tag['href'] if 'href' in a_tag.attrs else None,
        'data-kategorie': entry['data-kategorie'] if 'data-kategorie' in entry.attrs else None,
        'data-letter': entry['data-letter'] if 'data-letter' in entry.attrs else None,
        'data-synonyme': [synonym.strip() for synonym in entry['data-synonyme'].split(',')] if 'data-synonyme' in entry.attrs else None
    }

print(len(services))


# scrape data and create documents

all_nodes = []
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

        # remove unnecessary rest of the page
        content = content[:cutoff]

        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        text_nodes = splitter.get_nodes_from_documents([Document(text=content)])
        for node in text_nodes:
            node.metadata = {
                'Name': service_name,
                'Kategorie': service_data['data-kategorie'],
                'Buchstabe': service_data['data-letter'],
                'Synonyme': service_data['data-synonyme'],
                'Fachbereich': Fachbereich
            }
        all_nodes.extend(text_nodes)

print (len(all_nodes))