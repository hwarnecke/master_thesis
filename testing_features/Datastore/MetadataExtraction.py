import os
import re
import pdfplumber
from llama_index.core.schema import TextNode
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter

"""
I need to extract some metadata from the Bürgerforen pdfs.
The metadata is stored in the title of the pdf.
I wrote some functions that extract the metadata and the text from the pdf. 
The text is the split into smaller chunks and stored in TextNodes with the respective metadata added

Afterwards some tests of the functions are done.
The whole script is a test of features, later this needs to be put into a separate class or something.
"""


# most metadata is stored in the title of the pdf
# so we need a list of all the pdfs in order to extract the metadata
start_path = "../../data/Buergerforen"
disctricts = os.listdir(start_path)


# we need to extract the district, the date and the number of the protocol
def find_districts(text):
    pattern = r'Bürgerforum (.*?) \('
    matches = re.findall(pattern, text)
    matches = [word.strip() for match in matches for word in match.split(',')]
    for i in range(1, len(matches)):
        if matches[i].startswith('-'):
            matches[i] = matches[i-1] + matches[i]
    return matches

def find_date(text):
    pattern = r'\d{2}\.\d{2}\.\d{4}'
    dates = re.findall(pattern, text)
    return dates

def find_number(text):
    pattern = r'\(\d{2}\)'
    numbers = re.findall(pattern, text)
    # Remove parentheses
    numbers = [number.strip('()') for number in numbers]
    return numbers

# now we need to extract the text from the pdf and split it into smaller chunks
# the chunks are then stored in TextNodes with the metadata added
def create_text_nodes_from_pdf(pdf_path):
    # Extract text from PDF
    with pdfplumber.open(pdf_path) as pdf:
        text = ' '.join(page.extract_text() for page in pdf.pages)

    # Extract metadata from PDF title
    title = os.path.basename(pdf_path)
    districts = find_districts(title)
    dates = find_date(title)
    numbers = find_number(title)

    # Create TextNodes
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
    #chunks = splitter.split_text_into_chunks(text)
    #text_nodes = [TextNode(content=chunk, metadata={'districts': districts, 'dates': dates, 'numbers': numbers}) for chunk in chunks]
    text_nodes = splitter.get_nodes_from_documents([Document(text=text)])
    for node in text_nodes:
        node.metadata = {'districts': districts, 'dates': dates, 'numbers': numbers}

    return text_nodes


# now we need to process all the pdfs in a directory
# a list of all the TextNodes is returned
def process_all_pdfs(directory):
    all_nodes = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                nodes = create_text_nodes_from_pdf(pdf_path)
                all_nodes.extend(nodes)
    return all_nodes

# test the functions
path = start_path + "/" + disctricts[5]
test_pdf = os.listdir(path)[0]
district_matches = find_districts(test_pdf)
date_matches = find_date(test_pdf)
number_matches = find_number(test_pdf)
#print(district_matches)
#print(date_matches)
#print(number_matches)


# now we test how llamaindex data_loader splits a single pdf into multiple documents
# and how I can turn those into a TextNode

# load the pdf
pdf_path = path + "/" + test_pdf
#print(os.path.isfile(pdf_path))
documents = SimpleDirectoryReader(input_dir=path, num_files_limit=2).load_data()

# get the first document
first_doc = documents[4]
#print(first_doc)
#print(len(documents))

# create TextNodes from the pdf
# text_nodes = create_text_nodes_from_pdf(pdf_path)
# print(len(text_nodes))
# print(text_nodes[0])

# test the process_all_pdfs function
all_nodes = process_all_pdfs(start_path)

print(len(all_nodes))
print(all_nodes[-1].metadata)