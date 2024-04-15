import os
import re
import pdfplumber
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from tqdm import tqdm

class BuergerforenPDFLoader():
    """
    This class is highly specific to the Bürgerforen PDFs and to the way they are named.
    It is used to extract metadata from the title of the PDFs and to split the text into smaller chunks.
    """
    @staticmethod
    def load_data(directory):
        """
        Recursively goes through the directory and turn the pdfs into TextNodes with the respecitve metadata.
        :param directory: string path to the directory
        :return: list of TextNodes
        """
        all_nodes = []
        for root, dirs, files in tqdm(os.walk(directory)):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    nodes = BuergerforenPDFLoader.__create_text_nodes_from_pdf(pdf_path)
                    all_nodes.extend(nodes)
        return all_nodes

    @staticmethod
    def __find_districts(text):
        pattern = r'Bürgerforum (.*?) \('
        matches = re.findall(pattern, text)
        matches = [word.strip() for match in matches for word in match.split(',')]
        for i in range(1, len(matches)):
            if matches[i].startswith('-'):
                matches[i] = matches[i - 1] + matches[i]
        return matches

    @staticmethod
    def __find_date(text):
        pattern = r'\d{2}\.\d{2}\.\d{4}'
        dates = re.findall(pattern, text)
        return dates

    @staticmethod
    def __find_number(text):
        pattern = r'\(\d{2}\)'
        numbers = re.findall(pattern, text)
        # Remove parentheses
        numbers = [number.strip('()') for number in numbers]
        return numbers

    @staticmethod
    def __create_text_nodes_from_pdf(pdf_path):
        # Extract text from PDF
        with pdfplumber.open(pdf_path) as pdf:
            text = ' '.join(page.extract_text() for page in pdf.pages)

        # Extract metadata from PDF title
        title = os.path.basename(pdf_path)
        districts = BuergerforenPDFLoader.__find_districts(title)
        dates = BuergerforenPDFLoader.__find_date(title)
        numbers = BuergerforenPDFLoader.__find_number(title)

        # Create TextNodes
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        # chunks = splitter.split_text_into_chunks(text)
        # text_nodes = [TextNode(content=chunk, metadata={'districts': districts, 'dates': dates, 'numbers': numbers}) for chunk in chunks]
        text_nodes = splitter.get_nodes_from_documents([Document(text=text)])
        for node in text_nodes:
            node.metadata = {'districts': districts, 'dates': dates, 'numbers': numbers}

        return text_nodes