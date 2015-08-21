import os.path
from collections import defaultdict


class FileNotFoundError(Exception):
    pass


class HMMPOSTagger:
    def __init__(self, conll2000_path):
        self.observations = []
        self.observations_no = {}
        self.labels = []
        self.labels_no = {}

        self.transition = {}
        self.transition_probabilities = defaultdict(dict)
        self.emission = {}
        self.emission_probabilities = defaultdict(dict)

        self.alpha = 0.95

        if not os.path.isfile(conll2000_path):
            raise FileNotFoundError

        self.corpus_path = conll2000_path

        self.label_type = 'brill'
        self.start_label = '<<<start'
        self.end_label = 'end>>>'

    def prepare_corpus(self):
        with open(self.corpus_path) as corpus:
            self.__start_of_prepare()
            previous_label = self.start_label

            for line in corpus:
                try:
                    word, brill_label, swj_label = line.strip().split(' ')
                except:
                    self.__update_data(self.end_label, previous_label, self.end_label, self.end_label)
                    self.__update_data(self.start_label, self.end_label, self.start_label, self.start_label)
                    previous_label = self.start_label

                    continue

                previous_label = self.__update_data(word, previous_label, brill_label, swj_label)

            self.__end_of_prepare()

    def train(self):
        self.prepare_corpus()

        # P(wi | ti) = count(wi, ti) / count(ti)
        for word in self.observations_no:
            for label in self.labels_no:
                if label in self.emission and word in self.emission[label] and label in self.labels_no:
                    self.emission_probabilities[word][label] = self.alpha * (
                    self.emission[label][word] / float(self.labels_no[label])) + (1 - self.alpha) * (
                    1 / len(self.labels_no))
                else:
                    self.emission_probabilities[word][label] = (1 - self.alpha) * (1 / len(self.labels_no))

        # P(ti | t{i-1}) = count(t{i-1}, ti) / count(t{i-1})
        for label0 in self.labels_no:
            for label1 in self.labels_no:
                if label0 in self.transition and label1 in self.transition[label0]:
                    self.transition_probabilities[label1][label0] = self.alpha * (
                    self.transition[label0][label1] / float(self.labels_no[label0])) + \
                                                                    (1 - self.alpha) * (
                                                                    self.labels_no[label1] / float(len(self.labels_no)))
                else:
                    self.transition_probabilities[label1][label0] = (1 - self.alpha) * (
                    self.labels_no[label1] / float(len(self.labels_no)))

    def tagger(self, test_data):
        """ viterbi algorithm """
        test_labels, test_observations = self.__prepare_test_corpus(test_data)

        V = [{}]
        path = {}

        for label in self.labels_no:
            V[0][label] = self.emission_probabilities[test_observations[0]][label]
            path[label] = [label]

        for t in xrange(1, len(test_observations)):
            V.append({})
            new_path = {}

            for label in self.labels_no:
                best_prob = 0.0
                best_label = None

                for label0 in self.labels_no:
                    try:
                        prob = V[t - 1][label0] * \
                               self.transition_probabilities[label][label0] * \
                               self.emission_probabilities[test_observations[t]][label]
                    except:
                        prob = V[t - 1][label0] * self.transition_probabilities[label][label0]

                    if prob >= best_prob:
                        best_prob = prob
                        best_label = label0

                V[t][label] = best_prob
                new_path[label] = path[best_label] + [label]

            path = new_path

        n = 0
        if len(test_observations) != 1:
            n = t

        (best_prob, best_label) = max((V[n][y], y) for y in self.labels_no)
        best_labels = path[best_label]

        error = 0.0
        all_test = len(test_observations)
        for test_label, hmm_table in zip(test_labels, best_labels):
            if test_label <> hmm_table:
                error += 1

        return best_prob, best_labels, (all_test - error) / all_test

    def setLabelType(self, type):
        self.label_type = type

    def __prepare_test_corpus(self, corpus_path):
        if not os.path.isfile(corpus_path):
            raise FileNotFoundError

        labels = []
        observations = []

        with open(corpus_path) as corpus:
            observations.append(self.start_label)
            labels.append(self.start_label)

            for line in corpus:
                try:
                    word, brill_label, swj_label = line.strip().split(' ')
                except:
                    observations.append(self.end_label)
                    labels.append(self.end_label)

                    observations.append(self.start_label)
                    labels.append(self.start_label)

                    continue

                observations.append(word)

                if self.label_type == 'brill':
                    labels.append(brill_label)
                else:
                    labels.append(swj_label)

            observations.pop()
            labels.pop()

        return labels, observations

    def __update_data(self, word, previous_label, brill_label, swj_label):
        self.observations.append(word)

        if self.label_type == 'brill':
            label = brill_label
            self.labels.append(brill_label)
        else:
            label = swj_label
            self.labels.append(swj_label)

        try:
            self.labels_no[label] += 1
        except:
            self.labels_no.setdefault(label, 1)

        try:
            self.observations_no[word] += 1
        except:
            self.observations_no.setdefault(word, 1)

        try:
            self.transition[previous_label][label] += 1
        except:
            self.transition.setdefault(previous_label, {label: 1})
            self.transition[previous_label].setdefault(label, 1)

        try:
            self.emission[label][word] += 1
        except:
            self.emission.setdefault(label, {word: 1})
            self.emission[label].setdefault(word, 1)

        return label

    def __start_of_prepare(self):
        self.observations.append(self.start_label)
        self.labels.append(self.start_label)

        try:
            self.labels_no[self.start_label] += 1
        except:
            self.labels_no.setdefault(self.start_label, 1)

    def __end_of_prepare(self):
        self.observations.pop()
        self.labels.pop()
        self.labels_no[self.start_label] -= 1

        try:
            self.transition[self.end_label][self.start_label] -= 1
        except:
            pass
