import cv2, time
import numpy as np
import matplotlib.pyplot as plt

class SobelPaddedComparator:


    def __init__(self, image_path):
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None: 
            raise ValueError("Image not found")
        self.h, self.w = self.img.shape

    def manual_implementation(self):
        """
            _               _                   _                _
            |                 |                 |                  |
            | 1.0   0.0  -1.0 |                 |  1.0   2.0   1.0 |
        Gx = | 2.0   0.0  -2.0 |    and     Gy = |  0.0   0.0   0.0 |
            | 1.0   0.0  -1.0 |                 | -1.0  -2.0  -1.0 |
            |_               _|                 |_                _|
        """

        Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        
        padded_img = cv2.copyMakeBorder(
            self.img, 1, 1, 1, 1, 
            cv2.BORDER_CONSTANT, value=0
        )
        
        output = np.zeros((self.h, self.w))
        
        print("Crunching manual pixels with padding... 🐢")
        
        for i in range(self.h):
            for j in range(self.w):
                roi = padded_img[i : i + 3, j : j + 3]
                
                gx_val = np.sum(Gx * roi)
                gy_val = np.sum(Gy * roi)
                
                output[i, j] = np.sqrt(gx_val**2 + gy_val**2)
                
        return cv2.convertScaleAbs(output)

    def opencv_implementation(self):
        gx = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
        gy = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
        
        mag = cv2.magnitude(gx, gy)
        return cv2.convertScaleAbs(mag)

    def run_analysis(self):

        start = time.time()
        manual = self.manual_implementation()
        manual_time = time.time() - start

        start = time.time()
        cv_ref = self.opencv_implementation()
        cv_time = time.time() - start
        
        print(f"Manual Time: {manual_time:.4f}s")
        print(f"OpenCV Time: {cv_time:.4f}s")

        diff = cv2.absdiff(manual, cv_ref)
        
        hist = cv2.calcHist([diff], [0], None, [256], [0, 256])
        
        plt.figure(figsize=(12, 3))
        
        plt.subplot(1, 3, 1)
        plt.imshow(manual, cmap='gray')
        plt.title("Manual (Zero Padded)")
        
        plt.subplot(1, 3, 2)
        plt.imshow(diff, cmap='gray')
        plt.title(f"Difference image")
        
        plt.subplot(1, 3, 3)
        plt.plot(hist)
        plt.title("Histogram of Differences")
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    task = SobelPaddedComparator('size_50x50.jpg')
    task.run_analysis()