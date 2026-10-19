import csv
import numpy as np


def load_data(filename):
    age = []
    hg = []
    cls = []

    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip the header

        for row in reader:
            age.append(float(row[0]))
            hg.append(float(row[1]))
            cls.append(int(row[2]))

    return age, hg, cls
