import yaml
from sklearn import datasets
import pandas as pd

class InputParser:

    FILE_NAME = "config.yaml"
    INPUT_DATA = "../Data/LolStats.csv"
    settings = None

    enumMap = {
        "Fighter": 1,
        "Mage": 2,
        "Assassin": 3,
        "Marksman": 4,
        "Tank": 5,
        "Support": 6,
        "TOP": 1,
        "MID": 2,
        "ADC": 3,
        "SUPPORT": 4,
        "JUNGLE": 5,
        "D": 0,
        "C": 1,
        "B": 2,
        "A": 3,
        "S": 4,
        "God": 5
    }

    def __init__(self):
        self.data_x = []
        self.data_y = []

    def read_config_file(self):

        with open(self.FILE_NAME, 'r') as f:
            self.settings = yaml.safe_load(f)

    def prepare_random_data(self, rand_state: int):

        self.data_x, self.data_y = datasets.make_classification(
            n_samples = 300,
            random_state = rand_state,
            n_features = 2,
            n_classes = 3,
            n_clusters_per_class = 1,
            n_informative = 2,
            n_redundant = 0,
            n_repeated = 0
        )

    def prepare_tabular_data(self):

        frame = pd.read_csv(self.INPUT_DATA, header = 0)
        self.data_x = frame[frame.columns[1:-1]].to_numpy()
        self.data_y = frame[frame.columns[-1]].to_numpy()

        # Zamieniamy kolumny tekstowe na liczbowe (klasa postaci, rola):
        for i in [0, 1]:
            self.text_columns_to_enum(i)

        # Zamieniamy tiery na liczby:
        self.tiers_to_enum()
        self.data_y = list(self.data_y)

    def text_columns_to_enum(self, column_number: int):

        for data in self.data_x:
            text_data = data[column_number]
            try:
                data[column_number] = self.enumMap[text_data]
            except(KeyError):
                data[column_number] = 0

    def tiers_to_enum(self):

        for i in range(len(self.data_y)):
            self.data_y[i] = self.enumMap[self.data_y[i]]


    def get_prepared_data(self):
        return self.data_x.copy(), self.data_y.copy()
