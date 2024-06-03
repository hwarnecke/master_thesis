import requests
from bs4 import BeautifulSoup
import re
import tqdm

"""
I want to include the contact information from the service page in the data base.
For that I need to test how to collect the information properly and in a good format

Usually, each service has a contact and/or contact person field on the right side.
I already use that to collect some metadata.
These fields contain some link to the website of that contact.
All I need to do is grab that link and scrape the page behind it.
I should make sure to not scrape the some page twice though.

Things I need to exclude:
any url I already visited
any url that leads to a dienstleistung instead of a contact
-> or the other way around, only links that contain 'einrichtung' or 'mitarbeiter'
any href containing a telephone number
-> is already solved by the above

"""

def main():
    # the url to an example service that has both the contact and the contact person field
    #url: str = "https://service.osnabrueck.de/dienstleistungen/-/egov-bis-detail/dienstleistung/26094/show"
    #url = "https://service.osnabrueck.de/dienstleistungen/-/egov-bis-detail/dienstleistung/451814/show"
    url = "https://service.osnabrueck.de/dienstleistungen/-/egov-bis-detail/dienstleistung/5184/show"

    # extract the urls
    server = requests.get(url, timeout=3)
    soup = BeautifulSoup(server.text, 'html.parser')
    contact_divs = soup.find_all('div', attrs={'class': 'contact'})

    links: list[str] = []
    for div in contact_divs:
        # find the 'a' tags
        a_tags = div.find_all('a')
        spans = div.find_all('span')
        for a_tag in a_tags:
            contact = a_tag.text
            href:str = a_tag['href']

            if href.__contains__("einrichtung") or href.__contains__("mitarbeiter"):
                links.append(href)
                print(f"{contact}: {href}")

    print(len(links))


def main_dict():
    # the url to an example service that has both the contact and the contact person field
    url: str = "https://service.osnabrueck.de/dienstleistungen/-/egov-bis-detail/dienstleistung/26094/show"
    # url = "https://service.osnabrueck.de/dienstleistungen/-/egov-bis-detail/dienstleistung/451814/show"
    # url = "https://service.osnabrueck.de/dienstleistungen/-/egov-bis-detail/dienstleistung/5184/show"

    # extract the urls
    server = requests.get(url, timeout=3)
    soup = BeautifulSoup(server.text, 'html.parser')
    contact_divs = soup.find_all('div', attrs={'class': 'contact'})

    links: list[str] = []
    contacts = {}
    for div in contact_divs:
        # find the 'a' tags
        lis = div.find_all('li')
        print(len(lis))
        for li in lis:
            spans = li.find_all('span')
            a_tag = li.find('a')   # the url is always before the phone number

            if len(spans) == 1:
                name: str = spans[0].text
                position = None
            else:
                name: str = spans[1].text
                position: str = spans[0].text

            href: str = a_tag['href']

            print(f"{name}: {href}")

            # exclusion criteria are if it points towards a 'dienstleistung' or if we already visited it
            valid_link = href.__contains__("einrichtung") or href.__contains__("mitarbeiter")
            not_visited = href not in links
            print(valid_link,not_visited)

            if valid_link and not_visited:
                links.append(href)
                contacts[name]= {
                    "href": href,
                    "position": position
                }

    print(len(links))
    print(contacts)


def extract_content():
    # url = "https://service.osnabrueck.de/dienstleistungen/-/egov-bis-detail/einrichtung/5083/show"
    url = "https://service.osnabrueck.de/dienstleistungen/-/egov-bis-detail/mitarbeiter/25593/show"
    server = requests.get(url, timeout=3)
    soup = BeautifulSoup(server.text, 'html.parser')
    content = soup.find('div', attrs={'class': 'col-12'}).get_text()
    print(content)

def __ExtractRelatedServices(soup) -> dict[str,list]:
    div = soup.find('div', attrs={'id': 'allservices'})
    related_services: list[str] = []

    if not div:
        div = soup.find('div', attrs={'id': 'dienstleistungen-der-einrichtung'})

    if div:
        print("found div")
        services = div.find_all('li')
        for service in services:
            name: str = service.find('a').get_text()
            related_services.append(name)
    else:
        print("found no div")
    return {"Dienstleistungen": related_services}

def test_extract():
    # url = "https://service.osnabrueck.de/dienstleistungen/-/egov-bis-detail/mitarbeiter/25593/show"
    url = "https://service.osnabrueck.de/dienstleistungen/-/egov-bis-detail/einrichtung/5083/show"
    server = requests.get(url, timeout=3)
    soup = BeautifulSoup(server.text, 'html.parser')
    dict = __ExtractRelatedServices(soup)
    print(dict)

if __name__ == "__main__":
    test_extract()
