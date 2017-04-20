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
from random import randint

DATA_FILE_PATH = 'predict_data/noise'

def add_noise(percent, original_file, noisy_file):
    """
    Add noise to the oiriginal prediction data file
    """
    output_lines = []
    with open("{}/{}".format(DATA_FILE_PATH, original_file)) as input_file:
        for line in input_file:
            words = line.split('\t')
            age_label = int(words[-1])
            random_int = randint(1, 100)
            if random_int <= percent:
                while True:
                    random_label = randint(1, 5)
                    if random_label != age_label:
                        break
                age_label = random_label
            new_line = ' '.join(words[0:-1]) + ' ' + str(age_label) + '\n'
            output_lines.append(new_line)

    with open("{}/{}".format(DATA_FILE_PATH, noisy_file), 'w') as output_file:
        output_file.writelines(output_lines)


if __name__ == '__main__':
    for learner_name in ['lr', 'rf', 'ada', 'gbc', 'svm']:
        for noise_ratio in [2, 5, 10]:
            add_noise(noise_ratio, "{}_original.txt".format(learner_name),
                      "{}_noise_{}.txt".format(learner_name, noise_ratio))
