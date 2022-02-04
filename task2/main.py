import csv
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


cols_vectors = list(range(0, 80))
cols_labels = ["id", *cols_vectors]

imagesCount = [3, 5, 10, 15]

for num in imagesCount:
    # 3 images per class
    # prepare data for y_train
    training_csv_src = str(num) + '_training_imgs.csv'

    y_train = pd.read_csv(training_csv_src,
                          names=["id", "class"])

    y_train = y_train.sort_values(by=['id'], ascending=True)

    print(y_train)

    # prepare data for x_train
    x_train = pd.read_csv(
        'EdgeHistogram.csv', names=cols_labels, skiprows=1, delimiter=';')
    mask = x_train['id'].isin(y_train['id'])
    x_train = x_train.loc[mask]

    # final data prepared for x_train
    x_train = x_train.drop('id', axis=1)
    x_training_csv_src = "x_" + str(num) + "_train.csv"
    x_train.to_csv(x_training_csv_src, index=False)

    # final data prepared for y_train
    y_train = y_train['class']
    y_training_csv_src = "y_" + str(num) + "_train.csv"
    y_train.to_csv(y_training_csv_src, index=False)

    # prepare x_test data
    x_test = pd.read_csv(
        'EdgeHistogram.csv', names=cols_labels, skiprows=1, delimiter=';')
    x_test = x_test.drop('id', axis=1)
    x_test_csv_src = "x_" + str(num) + "_test.csv"
    x_test.to_csv(x_test_csv_src, index=False)

    # prepare y_test data
    y_test = pd.read_csv('Images.csv', names=[
        "id", "class"], skiprows=1, delimiter=';')
    y_test = y_test['class']
    y_test_csv_src = "y_" + str(num) + "_test.csv"
    y_test.to_csv(y_test_csv_src, index=False)

    model = LogisticRegression(max_iter=5000)
    # input data in model and

    # print(x_train)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)*100
    print("-----------------------------")
    print("number of the images: " + str(num))
    print("Accuracy: ")
    print(accuracy)
