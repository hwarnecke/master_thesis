import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlsplit

"""
old, slow and deprecated code for getting all links from a website
"""

def get_on_site_links(url, ignore=[]):
    """
    Get all links from a given url that are on the same domain
    :param url: string
        the URL to get the links from
    :param ignore: list of strings
        every URL containing one of the strings in this list will be ignored
    :return: list of strings
    """
    server = requests.get(url, timeout=3)
    try:
        soup = BeautifulSoup(server.text, 'html.parser')
    except:
        return []
    links = soup.find_all('a')
    valid = []

    # check if the <a> object has a href attribute
    for l in links:
        try:
            href = l['href']
        except:
            continue

        # create the absolute url
        href_abs = urljoin(url, href)

        # check if the link is valid and should be added to the list
        valid_flag = False

        # only accept links with the same netloc (i.e. the same domain)
        if urlsplit(url).netloc == urlsplit(href_abs).netloc:
            valid_flag = True

        # links containing one of the strings in the ignore list will be ignored
        for i in ignore:
            if i in href_abs:
                valid_flag = False


        if valid_flag:
            valid.append(href_abs)

    # remove duplicates (might not catch some that are not exactly the same but lead to the same page)
    valid = list(set(valid))

    return valid

def get_urls(start_url, limit=100_000, ignore=[]):
    """
    Get all links, including those on subpages, from a given start URL
    :param url: string
        the start URL
    :param limit: int
        the maximum number of subpages to visit
    :param ignore: list of strings
        every URL containing one of the strings in this list will be ignored
    :return: list of strings
    """
    # initialize a stack and a list of visited sites
    stack = [start_url]
    visited = []

    last_change = 0
    # check each URL in the stack for new links
    while stack and len(visited) <= limit:

        # report progress every 50 subpages
        subpage_count = len(visited)
        if subpage_count % 50 == 0 and subpage_count != last_change:
            print(str(subpage_count) + " Subpages visited.")
            print(str(len(stack)) + " Subpages currently in stack.")
            last_change = subpage_count

        url = stack.pop()
        if url in visited:
            continue
        else:
            # add new URL to the visited and get all the links on the page
            visited.append(url)
            links = get_on_site_links(url, ignore=ignore)
            for link in links:
                stack.append(link)


    return visited