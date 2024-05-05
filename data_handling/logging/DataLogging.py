import csv
class DataLogging:
    """
    Is initialized with a file path and the RAG system ID.
    It will write the logs as a csv file to the given path.
    It has a log function to store the data in this object and a write function to finally write the data to the file.
    The logs are stored as strings or list of strings.

    The system ID should be a unique identifier that contains the following information:
    - the type of the system (which advanced technique is used)
    - the type of LLM used
    - the type of embedding used
    the exact format is TBI.

    It will log the following information:
    (this list will be updated as the project progresses)
    - the query string
    - the response string
    - the evaluation scores & explanations
    - the time scores
    - the tokens used & the cost value
    - the retrieved nodes
    """

    def __init__(self, file_path, rag_system_id):
        """
        Initializes the DataLogging object with the given file path and RAG system ID.
        the file path should be a csv file.
        :param file_path: String - path to the csv file
        :param rag_system_id: String - Identifier for the RAG system (format TBI)
        """
        self.file_path = file_path
        self.rag_system_id = rag_system_id

        self.query = ""
        self.response = ""
        self.evaluation_scores = []
        self.time_scores = []
        self.tokens = []
        self.retrieved_nodes = []


    def log(self, value, log_type):
        """
        Logs the given value with the given log type.
        Possible log types are:
        - query
        - response
        - evaluation_scores
        - time_scores
        - tokens
        - retrieved_nodes

        :param value: the value to log
        :param log_type: the type of the log
        :return: None
        """

        # not pretty but it works
        if log_type == "query":
            self.query = value
        elif log_type == "response":
            self.response = value
        elif log_type == "evaluation_scores":
            self.evaluation_scores = value
        elif log_type == "time_scores":
            self.time_scores = value
        elif log_type == "tokens":
            self.tokens = value
        elif log_type == "retrieved_nodes":
            self.retrieved_nodes = value
        else:
            raise ValueError("Unknown log type.")

    def write(self):
        """
        Writes the log to the file.
        :return: None
        """
        # should be called after all the logs are set.
        # Potentially, I could add a check that he automatically logs the data once all values are set
        # but I am not sure if that makes it more or less save, tbh.

        #TODO: test if he handles list of strings correctly and if this is the correct way to write CSV files
        #TODO: Do I even want CSV? Why not just a text file with a specific format that I can easily import as CSV?

        with open(self.file_path, "a") as file:
            file.write(f"{self.rag_system_id}, {self.query}, {self.response}, {self.evaluation_scores}, {self.time_scores}, {self.tokens}, {self.retrieved_nodes}\n")

        # reset the values, so the object can be reused for the next question
        # if anything goes wrong, and he does not write a specific value, then it will be empty and not use an old value
        self.query = ""
        self.response = ""
        self.evaluation_scores = []
        self.time_scores = []
        self.tokens = []
        self.retrieved_nodes = []


    def get(self, log_type):
        """
        Returns the log of the given type.
        :param log_type: String - the type
        :return: String or List of String - the log value

        Possible log types are:
        - query
        - response
        - evaluation_scores
        - time_scores
        - tokens
        - retrieved_nodes
        """
        if log_type == "query":
            return self.query
        elif log_type == "response":
            return self.response
        elif log_type == "evaluation_scores":
            return self.evaluation_scores
        elif log_type == "time_scores":
            return self.time_scores
        elif log_type == "tokens":
            return self.tokens
        elif log_type == "retrieved_nodes":
            return self.retrieved_nodes
        else:
            raise ValueError("Unknown log type.")