
import dbscan
import os

dirname = os.path.dirname(__file__)


def main():
    print("Pick an image from the list ")
    images = os.listdir(
        'C:/Users/OlemilekanKoku/Desktop/Workspace/DataMining/task1/images')
    for image in images:
        print(image)

    path = input()
    print("Choose the algorithm")
    print("1 for dbscan and 2 for k means")

    selection = int(input())

    if selection == 1:
        print("Enter an EPS value ")
        # The distance that specifies the neighborhoods.
        # Two points are considered to be neighbors if the distance between them are less than or equal to eps.
        eps = int(input())
        print("Enter the min points")
        # Minimum number of data points to define a cluster.
        min_pts = int(input())
        dbscan.main(path, eps, min_pts)
    else:
        print("Enter a value for k:")
        # k_value = int(input())
        # k_means.main(path, k_value)


if __name__ == '__main__':
    main()
