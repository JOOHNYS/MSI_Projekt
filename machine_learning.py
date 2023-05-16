from input_parser import InputParser
from output_data import OutputData, ArgScores
from support_vector_machines import SupportVectorMachines
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold

from copy import deepcopy
import numpy as np

"""
Klasa MachineLearning zawiera wszelkie metody potrzebne do przeprowadzenia ekspermentów.
"""
class MachineLearning:

    RANDOM_STATE = 2137
    SPLITS_NUM = 5
    REPEATS = 2

    inputParser = InputParser()
    outputData = OutputData()

    """
    Metoda run_experiments odczytuje dane z pliku konfiguracyjnego config.yaml i odpowiednio uruchamia wybrane algorytmy i testy.
    """
    def run_experiments(self):

        # Pobieramy ustawienia z pliku:
        self.inputParser.read_config_file()

        # Generujemy zbiory cech i wyników klasyfikacji. Losowe lub wczytywane z pliku csv:
        if self.inputParser.settings['use_real_data']:
            self.inputParser.prepare_tabular_data()
            print("Przetwarzanie danych rzeczywistych - 6 klas, 9 cech znaczących")

        else:
            self.inputParser.prepare_random_data(self.RANDOM_STATE)
            print("Przetwarzanie danych generowanych - 3 klasy, 2 cechy") # tylko 2 cechy, aby wyświetlanie na wykresie miało sens.

        # Pobieramy przygotowane zbiory danych:
        data_x, data_y = self.inputParser.get_prepared_data()

        # Dzielimy na zbiór uczący i testowy (na kolejny etap to się zamieni na walidację krzyżową):
        x_train, x_test, y_train, y_test = train_test_split(
            data_x, data_y,
            random_state = self.RANDOM_STATE
        )

        splits = self.SPLITS_NUM
        repeats = self.REPEATS
        self.rskf = RepeatedStratifiedKFold(
            n_splits=splits,
            n_repeats=repeats,
            random_state=self.RANDOM_STATE)

        # Uruchamiamy pokolej wybrane w pliku config.yaml algorytmy i liczymy metryki:

        # Support Vector Machines:
        if self.inputParser.settings['enabled_algorithsm']['support_vector_machines']:

            self.cross_valid(SupportVectorMachines(), "SVM", data_x, data_y)

        # One vs one:
        if self.inputParser.settings['enabled_algorithsm']['one_vs_one']:

            self.cross_valid(OneVsOneClassifier(GaussianNB()), "1vs1", data_x, data_y)

        # One vs rest:
        if self.inputParser.settings['enabled_algorithsm']['one_vs_rest']:

            self.cross_valid(OneVsRestClassifier(GaussianNB()), "1vsR", data_x, data_y)

        # One vs one:
        if self.inputParser.settings['enabled_algorithsm']['random_forest']:

            self.cross_valid(RandomForestClassifier(random_state = self.RANDOM_STATE), "RF", data_x, data_y)

        self.outputData.t_student(self.inputParser.settings['alpha'])

    """
    Metoda cross_valid jest metodą przeprowadzającą walidację krzyżową dla danego algorytmu.

    Parametry:
    - algorithm - obiekt algorytmu,
    - algorithm_name - nazwa algorytmu,
    - data_x - wektor cech,
    - data_y - poprawne wyniki klasyfikacji.
    """
    def cross_valid(self, algorithm, algorithm_name: str, data_x, list_y):

        prec_scores = []
        recc_scores = []
        data_y = np.array(list_y)

        # Walidacja krzyżowa:
        for train_index, test_index in self.rskf.split(data_x, data_y):

            x_train, x_test = data_x[train_index], data_x[test_index]
            y_train, y_test = data_y[train_index], data_y[test_index]
            algcopy = deepcopy(algorithm)
            algcopy.fit(x_train, y_train)
            y_predicted = algcopy.predict(x_test)

            prec_scores.append(precision_score(y_test, y_predicted, average = 'weighted', zero_division = 0))
            recc_scores.append(recall_score(y_test, y_predicted, average = 'weighted', zero_division = 0))

        # Wyliczenie statystyk:
        mean_prec = np.mean(prec_scores)
        std_prec = np.std(prec_scores)
        mean_rec = np.mean(recc_scores)
        std_rec = np.std(recc_scores)

        self.outputData.algs.append(ArgScores(algorithm_name, mean_prec, mean_rec, std_prec, std_rec, prec_scores + recc_scores))
        print(self.outputData.algs[-1])

        file_name = "../Data/" + algorithm_name

        # Wyliczenie linii na wykres SVMu:
        lines = []
        if algorithm_name == "SVM":

            x0_1 = np.amin(x_test[:, 0])
            x0_2 = np.amax(x_test[:, 0])

            x1_1 = algcopy.get_hyperplane_value(x0_1, 0)
            x1_2 = algcopy.get_hyperplane_value(x0_2, 0)

            x2_1 = algcopy.get_hyperplane_value(x0_1, 1)
            x2_2 = algcopy.get_hyperplane_value(x0_2, 1)

            x3_1 = algcopy.get_hyperplane_value(x0_1, 2)
            x3_2 = algcopy.get_hyperplane_value(x0_2, 2)

            lines = [x0_1, x0_2, x1_1, x1_2, x2_1, x2_2, x3_1, x3_2]

        # Zapis danych i wykresów:
        if not self.inputParser.settings['use_real_data']:
            self.outputData.save_one_fold_plot(file_name, x_test, y_test, y_predicted, algorithm_name, lines)
        self.outputData.save_folds_plot(file_name + "folds", prec_scores, recc_scores, mean_prec, mean_rec, algorithm_name)

        self.outputData.save_results("../Data/results.csv")
