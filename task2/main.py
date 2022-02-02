import csv
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# write_file3 = open("3_training_imgs.csv", "w")
# writer3 = csv.writer(write_file3)

# write_file5 = open("5_training_imgs.csv", "w")
# writer5 = csv.writer(write_file5)

# write_file10 = open("10_training_imgs.csv", "w")
# writer10 = csv.writer(write_file10)

# write_file15 = open("15_training_imgs.csv", "w")
# writer15 = csv.writer(write_file15)


# images = pd.read_csv('Images.csv', names=[
#                      "id", "class"], delimiter=';', skiprows=1)


col_names_edge_histogram = ["id"]
for i in range(80):
    col_names_edge_histogram.append(str(i))

# edge_histogram = pd.read_csv(
#     'EdgeHistogram.csv', names=col_names_edge_histogram, delimiter=';', skiprows=1)

# merged = images.merge(edge_histogram, on='id')
# merged.to_csv("output.csv", index=False)

# merged.sort_values(by=['class'], ascending=True).to_csv(
#     "output.csv", index=False)

# unique_classname = merged.drop_duplicates(subset=['class'])

# unique_classname = unique_classname['class']


# for class_name in unique_classname:
#     imagesCSV = csv.reader(open('Images.csv'), delimiter=';')
#     next(imagesCSV)
#     i = 0
#     for row in imagesCSV:
#         if row[1] == class_name:
#             if i < 3:
#                 writer3.writerow(row)
#             if i < 5:
#                 writer5.writerow(row)
#             if i < 10:
#                 writer10.writerow(row)
#             if i < 15:
#                 writer15.writerow(row)
#             i += 1

imagesCount = [3, 5, 10, 15]

for num in imagesCount:
    # 3 images per class
    # prepare data for y_train
    training_csv_src = str(num) + '_training_imgs.csv'
    y_train = pd.read_csv(training_csv_src,
                          names=["id", "class"])

    y_train = y_train.sort_values(by=['id'], ascending=True)

    # prepare data for x_train
    x_train = pd.read_csv(
        'EdgeHistogram.csv', names=col_names_edge_histogram, skiprows=1, delimiter=';')
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
        'EdgeHistogram.csv', names=col_names_edge_histogram, skiprows=1, delimiter=';')
    x_test = x_test.drop('id', axis=1)
    x_test_csv_src = "x_" + str(num) + "_test.csv"
    x_test.to_csv(x_test_csv_src, index=False)

    # prepare y_test data
    y_test = pd.read_csv('Images.csv', names=[
        "id", "class"], skiprows=1, delimiter=';')
    y_test = y_test['class']
    y_test_csv_src = "y_" + str(num) + "_test.csv"
    y_test.to_csv(y_test_csv_src, index=False)

    model = svm.SVC(kernel='linear', max_iter=300)
    # input data in model and

    # print(x_train)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)*100
    print("-----------------------------")
    print("number of the images: " + str(num))
    print("Accuracy: ")
    print(accuracy)
