# https://stanford.edu/~cpiech/cs221/handouts/kmeans.html

def main(path, k):
    centers = []
    iterations = 0
    image = cv2.imread(os.path.join(dirname, path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))
