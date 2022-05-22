"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os
import re

from helper_classes import WeakClassifier, VJ_Classifier

OUTPUT_DIR="output"


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   (tuple): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """

    images_files = [f for f in os.listdir(folder) if f.endswith(".png")]

    imgs = [np.array(cv2.imread(os.path.join(folder, f), 0)) for f in images_files] # 0 grayscale

    imgs_resize = np.stack((cv2.resize(f, size).flatten().astype(float) for f in imgs))

    labels = np.array([int((re.search(r'subject(.*?)\.', f).group(1))) for f in images_files])

    # print (labels)

    return (imgs_resize, labels)


def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    n = y.size
    M = int(n*p)
    rand = np.random.permutation(n)

    Xtrain = X[rand[:M], :]
    ytrain = y[rand[:M]]
    Xtest = X[rand[M:], :]
    ytest = y[rand[M:]]

    return Xtrain, ytrain, Xtest, ytest


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """

    return np.sum(x, 0)/x.shape[0]



def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """

    mean_face = get_mean_face(X)

    phy = X - mean_face

    eig_vals, eig_vecs = np.linalg.eig(phy.dot(phy.T))

    idx = (-eig_vals).argsort()[:k]

    eigenvalues = np.take(eig_vals, idx)

    # print (eig_vecs.shape)
    # print ()

    eigenvectors = phy.T.dot(eig_vecs[idx, :].T)
    #
    # print (eigenvectors.shape)
    # print ()
    # print (eigenvalues)

    return eigenvectors, eigenvalues


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""

        for j in range(self.num_iterations):
            self.weights = self.weights/np.sum(self.weights)
            # uniform_weights = np.ones((self.Xtrain.shape[0],)) / self.Xtrain.shape[0]
            # wk_clf = WeakClassifier(self.Xtrain, self.ytrain, uniform_weights)
            wk_clf = WeakClassifier(self.Xtrain, self.ytrain, self.weights)
            wk_clf.train()
            wk_results = [wk_clf.predict(x) for x in self.Xtrain]
            weights_sum = np.sum(self.weights[wk_results != self.ytrain])
            alpha = 0.5*np.log((1-weights_sum)/weights_sum)

            if weights_sum > self.eps:
                self.weights = self.weights*np.exp(-self.ytrain*alpha*wk_results)
                self.weakClassifiers.append(wk_clf)
                self.alphas.append(alpha)
            else:
                break


    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """

        y_pred = self.predict(self.Xtrain)

        incorrect = int(np.sum(abs(self.ytrain-y_pred))/2)
        correct = y_pred.size - incorrect

        return correct, incorrect


    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        summ = np.array([0] * X.shape[0])
        for i in range(len(self.weakClassifiers)):
            wk_pred = np.array([self.weakClassifiers[i].predict(x) for x in X])
            summ = summ + self.alphas[i]*wk_pred

        summ[summ[:] > 0] = 1
        summ[summ[:] < 0] = -1

        return summ

class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        row = self.position[0]
        col = self.position[1]
        h = self.size[0]
        w = self.size[1]

        img = np.zeros(shape, np.uint8)
        img[row:row+int(h/2), col:col+w] = 255
        img[row+int(h/2):row+h, col:col+w] = 126

        return img

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        row = self.position[0]
        col = self.position[1]
        h = self.size[0]
        w = self.size[1]

        img = np.zeros(shape, np.uint8)
        img[row:row + h, col:col + int(w / 2)] = 255
        img[row:row + h, col + int(w / 2):col + w] = 126

        return img

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        row = self.position[0]
        col = self.position[1]
        h = self.size[0]
        w = self.size[1]

        img = np.zeros(shape, np.uint8)
        img[row:row + int(h / 3), col:col + w] = 255
        img[row + int(h / 3):row + int(2*h / 3), col:col + w] = 126
        img[row + int(2*h / 3):row + h, col:col + w] = 255

        return img

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        row = self.position[0]
        col = self.position[1]
        h = self.size[0]
        w = self.size[1]

        img = np.zeros(shape, np.uint8)
        img[row:row + h, col:col + int(w / 3)] = 255
        img[row:row + h, col + int(w / 3):col + int(2*w / 3)] = 126
        img[row:row + h, col + int(2*w / 3):col + w] = 255

        return img

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        row = self.position[0]
        col = self.position[1]
        h = self.size[0]
        w = self.size[1]

        img = np.zeros(shape, np.uint8)
        img[row:row+int(h/2), col:col+int(w/2)] = 126
        img[row:row+int(h/2), col+int(w/2):col+w] = 255
        img[row+int(h/2):row+h, col:col+int(w/2)] = 255
        img[row+int(h/2):row+h, col+int(w/2):col+w] = 126

        return img

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """
        row = self.position[0]
        col = self.position[1]

        h = self.size[0]
        w = self.size[1]
#        ii = ii.astype('int64')
        
#        print (ii[:10, :10])
        
        if row == 0:
            score = 0
            return score
        
        if col == 0:
            score = 0
            return score

            
        
        # if self.feat_type == (2, 1):
        #     A = ii[row + h // 2, col + w] - ii[row + h // 2, col] - ii[row, col + w] + ii[row, col]
        #     B = ii[row + h, col + w] - ii[row + h, col] - ii[row + h // 2, col + w] + ii[row + h // 2, col]
        #     score = A - B

        if self.feat_type == (2, 1):
            A = ii[row + h // 2 - 1, col + w - 1] - ii[row + h // 2 - 1, col - 1] - ii[row - 1, col + w - 1] + ii[row - 1, col - 1]
            B = ii[row + h - 1, col + w - 1] - ii[row + h - 1, col - 1] - ii[row + h // 2 - 1, col + w - 1] + ii[row + h // 2 - 1, col - 1]
            if A < 0 or B < 0:
                print (A, B)
                
                raise NotImplementedError
            score = A - B

        if self.feat_type == (1, 2):
            A = ii[row + h - 1, col + w // 2 - 1] - ii[row + h - 1, col - 1] - ii[row - 1, col + w // 2 - 1] + ii[row - 1, col - 1]
            B = ii[row + h - 1, col + w - 1] - ii[row + h - 1, col + w // 2 - 1] - ii[row - 1, col + w - 1] + ii[row - 1, col + w // 2 - 1]
            if A < 0 or B < 0:
                print (A, B)
                raise NotImplementedError
            score = A - B

        if self.feat_type == (3, 1):
            A = ii[row + h // 3 - 1, col + w - 1] - ii[row + h // 3 - 1, col - 1] - ii[row - 1, col + w - 1] + ii[row - 1, col - 1]
            B = ii[row + h * 2 // 3 - 1, col + w - 1] - ii[row + h * 2 // 3 - 1, col - 1] - ii[row + h // 3 - 1, col + w - 1] + ii[row + h // 3 - 1, col - 1]
            C = ii[row + h - 1, col + w - 1] - ii[row + h - 1, col - 1] - ii[row + h * 2 // 3 - 1, col + w - 1] + ii[row + h * 2 // 3 - 1, col - 1]
            if A < 0 or B < 0 or C < 0:
                print (A, B, C)
                raise NotImplementedError
            score = A - B + C

        if self.feat_type == (1, 3):
            A = ii[row + h - 1, col + w // 3 - 1] - ii[row + h - 1, col - 1] - ii[row - 1, col + w // 3 - 1] + ii[row - 1, col - 1]
            B = ii[row + h - 1, col + w * 2 // 3 - 1] - ii[row + h - 1, col + w // 3 - 1] - ii[row - 1, col + w * 2 // 3 - 1] + ii[row - 1, col + w // 3 - 1]
            C = ii[row + h - 1, col + w - 1] - ii[row + h - 1, col + w * 2 // 3 - 1] - ii[row - 1, col + w - 1] + ii[row - 1, col + w * 2 // 3 - 1]
            if A < 0 or B < 0 or C < 0:
                print (A, B, C)
                raise NotImplementedError            
            score = A - B + C

        if self.feat_type == (2, 2):
            A = ii[row + h // 2 - 1, col + w // 2 - 1] - ii[row + h // 2 - 1, col - 1] - ii[row - 1, col + w // 2 - 1] + ii[row - 1, col - 1]
            B = ii[row + h // 2 - 1, col + w - 1] - ii[row + h // 2 - 1, col + w // 2 - 1] - ii[row - 1, col + w - 1] + ii[row - 1, col + w // 2 - 1]
            C = ii[row + h - 1, col + w // 2 - 1] - ii[row + h - 1, col - 1] - ii[row + h // 2 - 1, col + w // 2 - 1] + ii[row + h // 2 - 1, col - 1]
            D = ii[row + h - 1, col + w - 1] - ii[row + h - 1, col + w // 2 - 1] - ii[row + h // 2 - 1, col + w - 1] + ii[row + h // 2 - 1, col + w // 2 - 1]
            if A < 0 or B < 0 or C < 0 or D < 0:
                print (A, B, C, D)
                raise NotImplementedError
            score = - A + B + C - D

        return score


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """
    img_out = []

    for img in images:
        integral_img = np.zeros_like(img, dtype='int64')
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                integral_img[i, j] = np.sum(img[:i+1, :j+1])
        img_out.append(integral_img)

    return img_out

class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei-1, sizej-1]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):

        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.

        # print (len(self.integralImages), len(self.haarFeatures))
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print(" -- compute all scores --")
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        print(" -- select classifiers --")

        for i in range(num_classifiers):
            weights = weights / np.sum(weights)
            h = VJ_Classifier(scores, self.labels, weights)
            h.train()
            et = h.error
            self.classifiers.append(h)
            beta = et/(1-et)
            #
            # for i in range(len(scores)):
            #     m = int(h.predict(scores[i]) == self.labels[i] )
            #     weights[i] = weights[i]*beta**m

            # print (len(scores))
            m = np.array([beta**int(h.predict(scores[i]) == self.labels[i]) for i in range(len(scores))])
            # m = np.array([beta ** (int(h.predict(scores[i]) == self.labels[i])*2) for i in range(len(scores))])
            weights = np.multiply(weights, m)

            alpha = np.log(1.0/beta)
            self.alphas.append(alpha)

            # print(et, alpha, beta)


    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))

        # Populate the score location for each classifier 'clf' in
        # self.classifiers.
        
        for clf in self.classifiers:
            idx = clf.feature
            feat = self.haarFeatures[idx]
            for i, im in enumerate(ii):
                score = feat.evaluate(im)
                scores[i,idx] = score

        # Obtain the Haar feature id from clf.feature

        # Use this id to select the respective feature object from
        # self.haarFeatures

        # Add the score value to score[x, feature id] calling the feature's
        # evaluate function. 'x' is each image in 'ii'

        result = []

        # Append the results for each row in 'scores'. This value is obtained
        # using the equation for the strong classifier H(x).

        alpha_sum = np.sum(self.alphas)
        for x in scores:
            # TODO
            hx = 0
            for i in range(len(self.classifiers)):
                pred_label = self.classifiers[i].predict(x)
                # hx += (pred_label+1)/2*self.alphas[i]
                hx += pred_label * self.alphas[i]
            if hx >= 0.5*alpha_sum:
                result.append(1)
            else:
                result.append(-1)

        return result

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        wind_size = 24
        # print (image.shape)
        nrow, ncol = image_grey.shape

        images = []
        for i in range(nrow-wind_size):
            for j in range(ncol-wind_size):
                images.append(image_grey[i:i+wind_size, j:j+wind_size])

        result = self.predict(images)
        result = np.reshape(result, (nrow-wind_size, ncol-wind_size))

        xcor = []
        ycor = []
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if result[i, j] == 1:
                    # print (i,j)
                    xcor.append(j)
                    ycor.append(i)

        x = int(np.mean(xcor))
        y = int(np.mean(ycor))
        # print (x, y)

        # cv2.rectangle(image, (j, i), (j+wind_size, i+wind_size), (255, 0, 0))
        cv2.rectangle(image, (x, y), (x + wind_size, y + wind_size), (255, 0, 0))

        cv2.imwrite(os.path.join(OUTPUT_DIR, filename+".png"), image)

