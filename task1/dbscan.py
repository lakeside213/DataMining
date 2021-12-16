# https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf
import cv2
import os
import numpy as np
import enum
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec


dirname = os.path.dirname(__file__)


class Label(enum.Enum):
    UNCLASSIFIED = -1.0
    NOISE = 0.0


def main(path, eps, min_pts):

    # The function imread loads an image from the specified file and returns it
    image = cv2.imread(os.path.join(dirname, path))
    # load the image and convert it from BGR to RGB so that
    # we can dispaly it with matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3))

    labels = np.empty([len(pixels)])
    labels.fill(-1)

    core_points = []

    print(len(pixels))
    dbscan(pixels, eps, min_pts, labels, core_points)

    print(labels)
    print("Plotting Results")
    core_points = np.uint8(core_points)
    labels = np.uint8(labels)
    labels = labels.flatten()
    segmented_image = core_points[labels.flatten()]

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    for i in range(len(pixels)):
        ax.scatter(pixels[i][0], pixels[i][1],
                   pixels[i][2], color=(segmented_image[i] / 255))

    segmented_image = segmented_image.reshape(image.shape)

    fig.add_subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.show()


def dbscan(pixels, eps, minPts, labels, core_points):
    # visited_points = []
    queque = []
    cluster_id = 0
    for i, pixel in enumerate(pixels):
        print(i)
        random_point = i
        if labels[random_point] == -1:
            queque = get_neighbours(pixels, random_point, eps)

            # print(queque)
            if len(queque) < minPts:
                labels[random_point] = 0
                continue
            else:
                cluster_id += 1
                print("processing cluster", cluster_id)
                core_points.append(pixels[random_point])

                change_point_cluster_ids(queque, labels, cluster_id)
                queque.remove(random_point)

                while len(queque) != 0:
                    currentP = queque[0]
                    result = get_neighbours(pixels, currentP, eps)

                    if len(result) >= minPts:
                        index = 1
                        for item in result:
                            resultP = result[index]

                            if labels[resultP] == -1 or labels[resultP] == 0:
                                if labels[resultP] == -1:
                                    queque.append(resultP)
                                labels[resultP] = cluster_id
                        index += 1

                    queque.pop(0)


def get_neighbours(pixels, point, eps):
    neighbours = []
    # Scan all points in the database
    for index in range(len(pixels)):
        distance = np.linalg.norm(pixels[index]-pixels[point])
        if distance <= eps:  # Compute distance and check epsilon */
            neighbours.append(index)  # Add to result
    return neighbours


def change_point_cluster_ids(neighbours, labels, cluster_id):
    for n in neighbours:
        labels[n] = cluster_id


# def gen_unique_random_number(pixels, visited_points):
#     random_num = random.randint(0, len(pixels) - 1)
#     while random_num in visited_points:
#         random_num = random.randint(0, len(pixels) - 1)
#     visited_points.append(random_num)
#     return random_num


if __name__ == '__main__':
    main()
