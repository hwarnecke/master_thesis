import csv
import os
class DataLogging:
    """
    Class for logging data to a CSV file.
    Is initialized with a file path.
    Data is written as a dictionary, where the keys are the header names and the values are the data.
    the first time the write_csv method is called, a header is created.
    One such object should be created for each query engine and used for all questions.

    """

    def __init__(self, file_path):
        """
        Initializes the DataLogging object with the given file path and RAG system ID.
        the file path should be a csv file.
        :param file_path: String - path to the csv file
        """
        self.file_path = file_path
        self.header_filled = False
        self.add_header_filled = False


    def write_csv(self, data: dict, path: str = None):
        """
        Writes the given data to a CSV file.
        Creates a header the first time this method is called, regardless of the content of the file.
        If you open the same file with a different logger object, it will create a new header!
        Specifying the path can be helpful to log the additional values
        :param data: Dict containing the name of the header and the value
        :param path: str defaults to the file_path specified in the beginning
        :return: None
        """
        if path is None:
            path = self.file_path

        # in case the directory doesn't exist
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # 'w' creates a new file but would also delete the content if it already exists
        if os.path.exists(path):
            mode = "a"
        else:
            mode = "w"

        with open(path, mode) as file:
            fieldnames = data.keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=';')

            # write header if file is new
            if not self.header_filled:
                self.header_filled = True
                writer.writeheader()

            # since the data should be a single dictionary, we can directly write it
            writer.writerow(data)

