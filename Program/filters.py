import math
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


class filters:
    def __init__(self, img):
        self.in_image = img
        self.gray = cv.cvtColor(self.in_image, cv.COLOR_BGR2GRAY)
        self.img = self.in_image
        self.result = None

    def add_noise(self, noise_type="s&p"):
        
        if noise_type == "gauss":
            row, col, ch = self.img.shape
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = self.img + gauss
            self.result = noisy
            return noisy
        elif noise_type == "s&p":
            row, col, ch = self.img.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(self.img)
            num_salt = np.ceil(amount * self.img.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.img.shape]
            out[tuple(coords)] = 255
            num_pepper = np.ceil(amount * self.img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.img.shape]
            out[tuple(coords)] = 0
            self.result = out
            return out
     
        elif noise_type == "poisson":
            vals = len(np.unique(self.img))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(self.img * vals) / float(vals)
            self.result = noisy
            return noisy

    # Point Transformation
    def bc_adj(self, alpha=1.0, beta=0):
        # brightness and contrast adjustment
        new_img = np.zeros(self.img.shape, self.img.dtype)
        new_img[:, :, :] = np.clip(alpha * self.img[:, :, :] + beta, 0, 255)
        self.result = new_img
        return new_img

    def show_histogram(self):
        # show histogram of an img
        blank = np.zeros(self.img.shape[:2], dtype='uint8')
        mask = cv.circle(blank, (self.img.shape[1] // 2, self.img.shape[0] // 2), 100, 255, -1)
        plt.figure()
        plt.title('Colour Histogram')
        plt.xlabel('Bins')
        plt.ylabel('# of pixels')
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv.calcHist([self.img], [i], mask, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.show()
        cv.waitKey(0)

    def hist_equal(self):
        ycrcb = cv.cvtColor(self.img, cv.COLOR_BGR2YCR_CB)
        channels = cv.split(ycrcb)
        cv.equalizeHist(channels[0], channels[0])
        cv.merge(channels, ycrcb)
        image = cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR)
        self.result = image
        return image

    def low_pass_filter(self):
        # low pass filter to blur the image
        image = cv.blur(self.img, (3, 3))
        self.result = image
        return image

    def high_pass_filter(self):
        # high pass filter to highlight edges
        kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
        image = cv.filter2D(self.img, -1, kernel)
        self.result = image
        return image

    def median(self):
        image = cv.medianBlur(self.img, 3)
        self.result = image
        return image

    def avg(self):
        # same as low pass filter
        image = self.low_pass_filter()
        self.result = image
        return image

    def laplace(self):
        image = np.uint8(np.absolute(cv.Laplacian(self.gray, cv.CV_64F)))
        self.result = image
        return image

    def gauss(self):
        image = cv.GaussianBlur(self.img, (3, 3), 0)
        self.result = image
        return image

    def log(self):
        gauss = cv.cvtColor(self.gauss(), cv.COLOR_BGR2GRAY)
        image = np.uint8(np.absolute(cv.Laplacian(gauss, cv.CV_64F)))
        self.result = image
        return image

    def sobel_v(self):  # x
        image = cv.convertScaleAbs(cv.Sobel(self.img, cv.CV_16S, 1, 0, ksize=3))
        self.result = image
        return image

    def sobel_h(self):  # y
        image = cv.convertScaleAbs(cv.Sobel(self.img, cv.CV_16S, 0, 1, ksize=3))
        self.result = image
        return image

    def sobel(self):
        image = cv.addWeighted(self.sobel_v(), 0.5, self.sobel_h(), 0.5, 0)
        self.result = image
        return image

    def prewitt_v(self):  # x
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        image = cv.filter2D(self.img, -1, kernelx)
        self.result = image
        return image

    def prewitt_h(self):  # y
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        image = cv.filter2D(self.img, -1, kernely)
        self.result = image
        return image

    def prewitt(self):
        image = self.prewitt_v() + self.prewitt_h()
        self.result = image
        return image

    def canny(self):
        image = cv.Canny(self.gray, 150, 175)
        self.result = image
        return image

    def zero_crossing(self):
        # same as log
        image = self.log()
        self.result = image
        return image

    def skeleton(self):
        threshold, image = cv.threshold(self.gray, 150, 255, cv.THRESH_BINARY)
        skel = np.zeros(image.shape, np.uint8)
        element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        while True:
            opening = cv.morphologyEx(image, cv.MORPH_OPEN, element)
            temp = cv.subtract(image, opening)
            eroded = cv.erode(image, element)
            skel = cv.bitwise_or(skel, temp)
            image = eroded.copy()
            if cv.countNonZero(image) == 0:
                break
        self.result = skel
        return skel

    # Hough Transformation
    def line_detect(self):
        self.result = cv.medianBlur(self.img, 5)
        dst = cv.Canny(self.img, 50, 200, None, 3)
        lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
                cv.line(self.result, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
        return self.result

    def circle_detect(self):
        self.result = cv.medianBlur(self.img, 5)
        gray = cv.medianBlur(self.gray, 5)
        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8, param1=100, param2=30, minRadius=0, maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv.circle(self.result, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(self.result, center, radius, (255, 0, 255), 3)
        return self.result

    # Morphological Operations
    def erosion(self):
        element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        image = cv.erode(self.img, element)
        self.result = image
        return image

    def dilatation(self):
        element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        image = cv.dilate(self.img, element)
        self.result = image
        return image

    def opening(self):
        element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        image = cv.morphologyEx(self.img, cv.MORPH_OPEN, element)
        self.result = image
        return image

    def closing(self):
        element = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        image = cv.morphologyEx(self.img, cv.MORPH_CLOSE, element)
        self.result = image
        return image
