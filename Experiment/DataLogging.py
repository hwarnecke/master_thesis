import csv
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


    def write_csv(self, data):
        """
        Writes the given data to a CSV file.
        Creates a header the first time this method is called, regardless of the content of the file.
        If you open the same file with a different logger object, it will create a new header!
        :param data: Dict containing the name of the header and the value
        :return: None
        """

        with open(self.file_path, "a") as file:
            fieldnames = data.keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # write header if file is new
            if not self.header_filled:
                self.header_filled = True
                writer.writeheader()

            # since the data should be a single dictionary, we can directly write it
            writer.writerow(data)