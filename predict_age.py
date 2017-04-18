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
from sklearn import linear_model, metrics, ensemble, svm, tree


def load_training_data(file_name):
    """
    load training data from file_name
    """
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


def cross_validation_sklearner(learner, learner_name, X, Y, K=5):
    """
    cross validation for one learner
    """
    Y_1D = Y.reshape(-1)
    Y_pred = cross_val_predict(learner, X, Y_1D, cv=K, verbose=1, method='predict_proba')
    np.savetxt("predict_data/{}.txt".format(learner_name), Y_pred)

    Y_pred_label = cross_val_predict(learner, X, Y_1D, cv=K, method='predict').reshape(-1, 1)
    print("Prediction accuracy: {}".format(metrics.accuracy_score(Y, Y_pred_label)))


def cross_validation_multi_learners(learners, learner_names, X, Y, K=5):
    """
    cross validation for multiple learners
    """
    for idx, learner in enumerate(learners):
        print("======= {} =======".format(str(learner_names[idx])))
        cross_validation_sklearner(learner, learner_names[idx], X, Y, K)


if __name__ == "__main__":
    book_y, book_X = load_training_data("training_data/feature_avg_filtered.txt")
    learner_lr = linear_model.LogisticRegression()
    learner_svm = svm.SVC(probability=True)
    learner_rf = ensemble.RandomForestClassifier(n_estimators=50)
    learner_dt = tree.DecisionTreeClassifier()
    learner_gbc = ensemble.GradientBoostingClassifier(n_estimators=50)
    learner_ada = ensemble.AdaBoostClassifier(n_estimators=50)
    # cross_validation_multi_learners((learner_lr, learner_svm, learner_rf, learner_dt, learner_gbc),
                                    # ("lr", "svm", "rf", "dt", "gbc"), book_X, book_y)
    cross_validation_multi_learners((learner_ada,),
                                    ("ada",), book_X, book_y)
