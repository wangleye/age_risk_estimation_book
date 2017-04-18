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
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from predict_age import load_training_data

def load_test_data(learner_name):
    risk_data = np.loadtxt("predict_data/{}_risk.txt".format(learner_name))
    Y = risk_data[0, :]
    X = risk_data[1:, :]
    return X, Y

def risk_estimation(learner, learner_name):
    X, Y = load_test_data(learner_name)
    Y_pred_prob = cross_val_predict(learner, X, Y, cv=5, method='predict_proba')[:, 1]
    level_1_mask = np.argwhere(Y_pred_prob <= 0.25)
    level_2_mask = np.argwhere(Y > 0.25 and Y <= 0.5)
    level_3_mask = np.argwhere(Y > 0.5 and Y <= 0.75)
    level_4_mask = np.argwhere(Y > 0.75 and Y <= 0.9)
    level_5_mask = np.argwhere(Y > 0.9)

if __name__ == "__main__":
    book_y, book_X = load_training_data("training_data/feature_avg_filtered.txt")