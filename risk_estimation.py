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
from sklearn import linear_model, svm, ensemble, metrics
from sklearn.model_selection import KFold
from sklearn.calibration import CalibratedClassifierCV
from predict_age import load_training_data
import pandas as pd


def load_predict_data(learner_name):
    """
    load prediction results
    """
    risk_data = np.loadtxt("predict_data/{}_risk.txt".format(learner_name))
    y = risk_data[:, 0]
    X = risk_data[:, 1:]
    return X, y


def load_noise_data(learner_name, noise_ratio):
    """
    load noise data
    """
    risk_data = np.loadtxt("predict_data/noise/{}_noise_{}.txt".format(learner_name, noise_ratio))
    y = (risk_data[:, -1] == risk_data[:, -2])
    X = risk_data[:, 0:-2]
    return X, y


def risk_estimation(risk_X, risk_y):
    """
    estimate risk
    the risk learner is selected by the brier score loss / f1 score
    """
    risk_learners = [  # candidate risk learners
        linear_model.LogisticRegression(),
        ensemble.RandomForestClassifier(n_estimators=10),
        #ensemble.AdaBoostClassifier(n_estimators=10),
        #ensemble.GradientBoostingClassifier(n_estimators=10)]
        svm.SVC(),
        svm.LinearSVC()]
    
    risk_learner_names = ['lr', 'rf', 'kde_SVC', 'linear_svc']

    #best_brier_score_loss = 100
    best_y_pred_cali_prob = None
    best_f1_score = 0
    for i, risk_learner in enumerate(risk_learners):
        cali_learner = CalibratedClassifierCV(risk_learner, cv=3, method='isotonic')
        k_fold = KFold(3)
        y_pred_cali_prob = np.zeros((len(risk_y), ))
        for (train, test) in k_fold.split(risk_X, risk_y):
            cali_learner.fit(risk_X[train], risk_y[train])
            y_pred_cali_prob[test] = cali_learner.predict_proba(risk_X[test])[:, 1]
        current_f1_score = metrics.f1_score(risk_y, (y_pred_cali_prob>0.5).astype(int))
        print(risk_learner_names[i], "f1_score", current_f1_score)
        if current_f1_score > best_f1_score:
            current_f1_score = best_f1_score
            best_y_pred_cali_prob = y_pred_cali_prob
        #current_brier_score_loss = metrics.brier_score_loss(risk_y, y_pred_cali_prob)
        #if current_brier_score_loss < best_brier_score_loss:
            #best_brier_score_loss = current_brier_score_loss
            #best_y_pred_cali_prob = y_pred_cali_prob

    return best_y_pred_cali_prob


def prediction_accuracy(y_pred_prob, book_X, book_y, learner):
    """
    prediction accuracy on differnet risk level users
    """
    risk_labels = np.zeros((len(book_y), 1))

    l_masks = []
    l_masks.append(np.where(y_pred_prob <= 0.25)[0])
    l_masks.append(np.where((y_pred_prob > 0.25) & (y_pred_prob <= 0.5))[0])
    l_masks.append(np.where((y_pred_prob > 0.5) & (y_pred_prob <= 0.75))[0])
    l_masks.append(np.where((y_pred_prob > 0.75) & (y_pred_prob <= 0.9))[0])
    l_masks.append(np.where(y_pred_prob > 0.9)[0])

    for i, mask in enumerate(l_masks):
        risk_labels[mask] = i+1

    np.savetxt("risk_labels.txt", risk_labels, fmt='%i')  # save as integer

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


def evaluate_accuracy_on_each_risk_level():
    """
    Run the experiment to evluate the acutal prediction accuracy on each risk level of users
    """
    train_book_y, train_book_X = load_training_data("training_data/feature_avg_filtered.txt")
    learners = [
        linear_model.LogisticRegression(),
        ensemble.RandomForestClassifier(n_estimators=10),
        ensemble.GradientBoostingClassifier(n_estimators=10),
        ensemble.AdaBoostClassifier(n_estimators=10),
        svm.SVC(probability=True)]
    learner_names = ['lr', 'rf', 'gbc', 'ada', 'svm']

    y_prob_all = np.zeros((len(learner_names), len(train_book_y)))
    for ln_idx, lname in enumerate(learner_names):
        bx_risk_X, bx_risk_y = load_predict_data(lname)
        y_prob_all[ln_idx] = risk_estimation(bx_risk_X, bx_risk_y)
    y_prob_mean = np.mean(y_prob_all, axis=0)

    for each_learner in learners:
        prediction_accuracy(y_prob_mean, train_book_X, train_book_y, each_learner)


def evaluate_accuracy_on_each_risk_level_noise(noise_ratio):
    """
    Run the expierment to evaluate the actual prediction accuracy on each risk level of users
    based on the noisy dataset
    """
    train_book_y, train_book_X = load_training_data("training_data/feature_avg_filtered.txt")
    learners = [
        linear_model.LogisticRegression(),
        ensemble.RandomForestClassifier(n_estimators=10),
        ensemble.GradientBoostingClassifier(n_estimators=10),
        ensemble.AdaBoostClassifier(n_estimators=10),
        svm.SVC(probability=True)]
    learner_names = ['lr', 'rf', 'gbc', 'ada', 'svm']

    y_prob_all = np.zeros((len(learner_names), len(train_book_y)))
    for ln_idx, lname in enumerate(learner_names):
        noisy_risk_X, noisy_risk_y = load_noise_data(lname, noise_ratio)
        y_prob_all[ln_idx] = risk_estimation(noisy_risk_X, noisy_risk_y)
    y_prob_mean = np.mean(y_prob_all, axis=0)

    for each_learner in learners:
        prediction_accuracy(y_prob_mean, train_book_X, train_book_y, each_learner)


if __name__ == '__main__':
    # evaluate_accuracy_on_each_risk_level_noise(noise_ratio=10)
    learners = ['ada','svm','gbc','lr','rf']
    df = pd.DataFrame()
    for learner in learners:
        print("======", learner, "=======")
        X, y = load_predict_data(learner)
        df[learner+'_risk'] = risk_estimation(X, y)
    print(df.sample(5))

    df.to_pickle('./predict_data/risk_estimation_results.pkl')
