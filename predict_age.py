"""
    This is an algorithm to predict age from users' reading preferences
    based on book crossing dataset.
    Copyright (C) 2017  Leye Wang (wangleye@gmail.com)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model, metrics, ensemble, svm


def load_training_data(file_name):
    training_y = []
    training_X = []
    with open(file_name) as trainingInput:
        for line in trainingInput:
            words = line.split(' ')
            y = int(words[0])
            training_y.append(y)
            x = [float(i) for i in words[1:]]
            training_X.append(x)
    return np.asarray(training_y), np.asarray(training_X)


def cross_validation_sklearner(learner, X, Y, K=5):
    print("X shape: {}".format(X.shape))
    print("Y shape: {}".format(Y.shape))
    Y_1D = Y.reshape(-1)
    Y_pred = cross_val_predict(learner, X, Y_1D, cv=K, verbose=1)
    Y_pred = Y_pred.reshape(-1, 1)
    print("Y_pred shape: {}".format(Y_pred.shape))
    print("Prediction accuracy: {}".format(np.sqrt(metrics.accuracy_score(Y, Y_pred))))


def cross_validation_multiple_learners(learners, X, Y, K=5):
    for learner in learners:
        print("======= {} =======".format(str(learner)))
        cross_validation_sklearner(learner, X, Y)


if __name__ == "__main__":
    training_y, training_X = load_training_data("training_data/feature_avg_filtered.txt")
    learner_svm = svm.SVC()
    learner_rf = ensemble.RandomForestClassifier(n_estimators=50)
    learner_sdg = linear_model.SGDClassifier()
    cross_validation_multiple_learners((learner_svm, learner_rf, learner_sdg), training_X, training_y)
