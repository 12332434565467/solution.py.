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
def perceptron(age, hg, cls):
    np.random.seed(42)

    age = np.asarray(age, dtype=float)
    hg = np.asarray(hg, dtype=float)
    labels = np.where(np.asarray(cls) == 1, 1, -1)

    # Normalise features internally for numerical stability.
    # Convert the learned coefficients back to the original scale.
    age_mean, age_std = age.mean(), age.std()
    hg_mean, hg_std = hg.mean(), hg.std()

    if age_std == 0:
        age_std = 1.0
    if hg_std == 0:
        hg_std = 1.0

    x_age = (age - age_mean) / age_std
    x_hg = (hg - hg_mean) / hg_std

    X = np.column_stack((np.ones(len(age)), x_age, x_hg))
    weights = np.random.uniform(-0.01, 0.01, size=3)

    max_epochs = 10000

    for epoch in range(max_epochs):
        mistakes = 0

        for i in np.random.permutation(len(labels)):
            if labels[i] * np.dot(weights, X[i]) <= 0:
                weights += labels[i] * X[i]
                mistakes += 1

        if mistakes == 0:
            # Convert coefficients back to original feature units.
            w0 = weights[0] - (
                weights[1] * age_mean / age_std
                + weights[2] * hg_mean / hg_std
            )
            w_age = weights[1] / age_std
            w_hg = weights[2] / hg_std

            return float(w0), float(w_age), float(w_hg)

    raise RuntimeError(
        "The perceptron did not converge. Check whether the data are linearly separable."
    )


if __name__ == "__main__":
    age, hg, cls = load_data("stroke_separable.csv")
    w0, w_age, w_hg = perceptron(age, hg, cls)

    print("Weights:", w0, w_age, w_hg)

    scores = w0 + w_age * np.asarray(age) + w_hg * np.asarray(hg)
    predictions = (scores > 0).astype(int)

    correct = np.all(predictions == np.asarray(cls))
    inverted = np.all((1 - predictions) == np.asarray(cls))

    print("Correct separation:", correct or inverted)
