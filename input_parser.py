import yaml
from sklearn import datasets

class InputParser:

    FILE_NAME = "config.yaml"
    settings = None

    def __init__(self):
        self.data_x = []
        self.data_y = []

    def read_config_file(self):
        
        with open(self.FILE_NAME, 'r') as f:
            self.settings = yaml.safe_load(f)

    def prepare_random_data(self):
        
        self.data_x, self.data_y = datasets.make_classification(
            n_samples = 300,
            random_state = 2137 #firmowy znak Łukasza XD
        )

    def prepare_tabular_data(self):
        pass

    def get_prepared_data(self):
        return self.data_x.copy(), self.data_y.copy()