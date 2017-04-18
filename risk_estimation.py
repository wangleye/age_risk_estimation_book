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
from sklearn import linear_model, svm, ensemble
from sklearn.model_selection import cross_val_predict
from predict_age import load_training_data, metrics


def load_predict_data(learner_name):
    """
    load prediction results
    """
    risk_data = np.loadtxt("predict_data/{}_risk.txt".format(learner_name))
    y = risk_data[:, 0]
    X = risk_data[:, 1:]
    return X, y


def risk_estimation(learner_name):
    """
    estimate risk first and then predict accuracy on the estimated risk level
    """
    lr = linear_model.LogisticRegression()
    risk_X, risk_y = load_predict_data(learner_name)
    y_pred_prob = cross_val_predict(lr, risk_X, risk_y, cv=5, method='predict_proba')[:, 1]

    return y_pred_prob


def prediction_accuracy(y_pred_prob, book_X, book_y, learner):
    """
    prediction accuracy on differnet risk level users
    """
    l_masks = []
    l_masks.append(np.where(y_pred_prob <= 0.25)[0])
    l_masks.append(np.where((y_pred_prob > 0.25) & (y_pred_prob <= 0.5))[0])
    l_masks.append(np.where((y_pred_prob > 0.5) & (y_pred_prob <= 0.75))[0])
    l_masks.append(np.where((y_pred_prob > 0.75) & (y_pred_prob <= 0.9))[0])
    l_masks.append(np.where(y_pred_prob > 0.9)[0])

    # prediction on estimated risk levels
    accuracy = np.zeros((5, ))
    for msk_idx, msk in enumerate(l_masks):
        if len(msk) == 0:
            accuracy[msk_idx] = 0
            continue
        train_msk = np.ones(len(y_pred_prob), dtype=bool)
        train_msk[msk] = False
        learner.fit(book_X[train_msk], book_y[train_msk])
        pred_book_y = learner.predict(book_X[msk]).reshape(-1, 1)
        accuracy[msk_idx] = metrics.accuracy_score(book_y[msk], pred_book_y)
    print(accuracy)
    return accuracy


if __name__ == "__main__":
    train_book_y, train_book_X = load_training_data("training_data/feature_avg_filtered.txt")
    learner_lr = linear_model.LogisticRegression()
    learner_svm = svm.SVC(probability=True)
    learner_rf = ensemble.RandomForestClassifier(n_estimators=50)
    learner_gbc = ensemble.GradientBoostingClassifier(n_estimators=50)
    learner_ada = ensemble.AdaBoostClassifier(n_estimators=50)
    learners = [learner_lr, learner_rf, learner_gbc, learner_ada, learner_svm]
    learner_names = ['lr', 'rf', 'gbc', 'ada', 'svm']

    y_prob_all = np.zeros((len(learner_names), len(train_book_y)))
    for ln_idx, lname in enumerate(learner_names):
        y_prob_all[ln_idx] = risk_estimation(lname)
    y_prob_mean = np.mean(y_prob_all, axis=0)

    print("======results on mean user risk======")
    for each_learner in learners:
        prediction_accuracy(y_prob_mean, train_book_X, train_book_y, each_learner)

    print("======results on single user risk======")
    for i in range(5):
        prediction_accuracy(y_prob_all[i], train_book_X, train_book_y, learners[i])
