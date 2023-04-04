from input_parser import InputParser
from output_data import OutputData
from support_vector_machines import SupportVectorMachines
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.naive_bayes import GaussianNB

class MachineLearning:

    inputParser = InputParser()
    outputData = OutputData()

    def run_experiments(self):
        
        # Pobieramy ustawienia z pliku:
        self.inputParser.read_config_file()

        # Generujemy zbiory cech i wyników klasyfikacji. Losowe lub wczytywane z pliku csv:
        if self.inputParser.settings['use_real_data']:
            self.inputParser.prepare_tabular_data()

        else:
            self.inputParser.prepare_random_data()

        # Pobieramy przygotowane zbiory danych:
        data_x, data_y = self.inputParser.get_prepared_data()

        # Dzielimy na zbiór uczący i testowy (na kolejny etap to się zamieni na walidację krzyżową):
        x_train, x_test, y_train, y_test = train_test_split(
            data_x, data_y,
            random_state = 2137
        )

        # Uruchamiamy pokolei wybrane w pliku config.yaml algorytmy i liczymy metryki:

        # Support Vector Machines:
        if self.inputParser.settings['enabled_algorithsm']['support_vector_machines']:

            algorithm = SupportVectorMachines() # Jeszcze nie działa, ale nto astepny etap juz chyba
            algorithm.fit(x_train, y_train)
            y_predicted = algorithm.predict(x_test)
            score = precision_score(y_test, y_predicted)
            print("Metryka algorytmu SVM: " + str(score))

        # One vs one:
        if self.inputParser.settings['enabled_algorithsm']['one_vs_one']:

            algorithm = OneVsOneClassifier(GaussianNB()) # Jeszcze nie działa - przemyślenie tego estymatora.
            algorithm.fit(x_train, y_train)
            y_predicted = algorithm.predict(x_test)
            score = precision_score(y_test, y_predicted)
            print("Metryka algorytmu 1 vs 1: " + str(score))

        # One vs rest:
        if self.inputParser.settings['enabled_algorithsm']['one_vs_rest']:

            algorithm = OneVsRestClassifier(GaussianNB()) # Jeszcze nie działa -  przemyślenie tego estymatora.
            algorithm.fit(x_train, y_train)
            y_predicted = algorithm.predict(x_test)
            score = precision_score(y_test, y_predicted)
            print("Metryka algorytmu 1 vs Rest: " + str(score))

        # One vs one:
        if self.inputParser.settings['enabled_algorithsm']['random_forest']:

            algorithm = RandomForestClassifier()
            algorithm.fit(x_train, y_train)
            y_predicted = algorithm.predict(x_test)
            score = precision_score(y_test, y_predicted)
            print("Metryka algorytmu RF: " + str(score))


        # Wyśletlamy wykresy, zapisujemy do plików itp:
