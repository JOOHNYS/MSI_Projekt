from input_parser import InputParser
from output_data import OutputData
from support_vector_machines import SupportVectorMachines
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt

class MachineLearning:

    RANDOM_STATE = 2137

    inputParser = InputParser()
    outputData = OutputData()

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

        # Uruchamiamy pokolej wybrane w pliku config.yaml algorytmy i liczymy metryki:

        # Support Vector Machines:
        if self.inputParser.settings['enabled_algorithsm']['support_vector_machines']:

            algorithm = SupportVectorMachines(random_state = self.RANDOM_STATE)
            algorithm.fit(x_train, y_train)
            y_predicted = algorithm.predict(x_test)
            print(y_test)
            print(y_predicted)

            # debugowy wykres:
            if not self.inputParser.settings['use_real_data']:
                fig1 = plt.figure('Figure 1')
                plt.scatter(x_test[:, 0], x_test[:, 1], c= y_test)
                plt.axline((-1*algorithm.hyperplanes[0][1]/algorithm.hyperplanes[0][0], 0), (0, algorithm.hyperplanes[0][1]))
                plt.axline((-1*algorithm.hyperplanes[1][1]/algorithm.hyperplanes[1][0], 0), (0, algorithm.hyperplanes[1][1]))
                plt.axline((-1*algorithm.hyperplanes[2][1]/algorithm.hyperplanes[2][0], 0), (0, algorithm.hyperplanes[2][1]))
                plt.xlim(-4, 4)
                fig2 = plt.figure('Figure 2')
                plt.scatter(x_test[:, 0], x_test[:, 1], c= y_predicted)
                plt.show()

            self.do_metrics("SVM", y_test, y_predicted)

        # One vs one:
        if self.inputParser.settings['enabled_algorithsm']['one_vs_one']:

            algorithm = OneVsOneClassifier(GaussianNB())
            algorithm.fit(x_train, y_train)
            y_predicted = algorithm.predict(x_test)

            self.do_metrics("1vs1", y_test, y_predicted)

        # One vs rest:
        if self.inputParser.settings['enabled_algorithsm']['one_vs_rest']:

            algorithm = OneVsRestClassifier(GaussianNB())
            algorithm.fit(x_train, y_train)
            y_predicted = algorithm.predict(x_test)

            self.do_metrics("1vsR", y_test, y_predicted)

        # One vs one:
        if self.inputParser.settings['enabled_algorithsm']['random_forest']:

            algorithm = RandomForestClassifier(random_state = self.RANDOM_STATE)
            algorithm.fit(x_train, list(y_train))
            y_predicted = algorithm.predict(x_test)

            self.do_metrics("RF", y_test, y_predicted)


        # Wyśletlamy wykresy, zapisujemy do plików itp:
        # TODO next etap.

    def do_metrics(self, algName: str, y_test: list, y_pred: list):

        prec_score = precision_score(y_test, y_pred, average = 'weighted', zero_division = 0)
        recc_score = recall_score(y_test, y_pred, average = 'weighted', zero_division = 0)
        print("Metryki algorytmu " + algName + ":\tPrecision: %.3f" % prec_score + "\tRecall: %.03f" % recc_score)
