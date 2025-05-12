import openml

DATASET_ID = 1049


class EDA:
    def __init__(self):
        self.dataset = None

    def load_dataset(self) -> None:
        """
        Load a dataset from OpenML.
        """
        # Load the dataset from local repository
        self.dataset = openml.datasets.get_dataset(DATASET_ID)
