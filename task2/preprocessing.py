import csv
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


images = pd.read_csv('Images.csv', names=[
                     "id", "class"], delimiter=';', skiprows=1)


cols_vectors = list(range(0, 80))
cols_labels = ["id", *cols_vectors]

edge_histogram = pd.read_csv(
    'EdgeHistogram.csv', names=cols_labels, delimiter=';', skiprows=1)

merged = images.merge(edge_histogram, on='id')
merged.to_csv("merged.csv", index=False)

merged.sort_values(by=['class'], ascending=True).to_csv(
    "merged.csv", index=False)

class_names = merged.drop_duplicates(subset=['class'])

class_names = class_names['class']


imgs_3_file = open("3_training_imgs.csv", "w")
w3 = csv.writer(imgs_3_file)

imgs_5_file = open("5_training_imgs.csv", "w")
w5 = csv.writer(imgs_5_file)

imgs_10_file = open("10_training_imgs.csv", "w")
w10 = csv.writer(imgs_10_file)

imgs_15_file = open("15_training_imgs.csv", "w")
w15 = csv.writer(imgs_15_file)


# print(class_names)
for class_name in class_names:
    images_csv = csv.reader(open('Images.csv'),  delimiter=';')
    i = 0
    for row in images_csv:
        if len(row) == 2:
            if row[1] == class_name:
                if i < 3:
                    w3.writerow(row)
                if i < 5:
                    w5.writerow(row)
                if i < 10:
                    w10.writerow(row)
                if i < 15:
                    w15.writerow(row)
                i += 1
