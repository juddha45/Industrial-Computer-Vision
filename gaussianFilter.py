import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

class GaussianFilterTask:


    def __init__(self, image_path, Kernel_size=5, sigma=1.0):
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None: 
            raise ValueError("Image not found!")
        self.Kernel_size = Kernel_size
        self.sigma = sigma
        self.pad = Kernel_size // 2

    def manual_gaussian(self):
        g_1d = cv2.getGaussianKernel(self.Kernel_size, self.sigma) 
        kernel_2d = g_1d @ g_1d.T 
        
        padded = cv2.copyMakeBorder(
            self.img, self.pad, self.pad, self.pad, self.pad, 
            cv2.BORDER_CONSTANT, value=0
        )
        
        h, w = self.img.shape
        output = np.zeros((h, w), dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                roi = padded[i : i + self.Kernel_size, j : j + self.Kernel_size]
                weighted_sum = np.sum(roi * kernel_2d)
                output[i, j] = int(round(weighted_sum))
                
        return output

    def run_comparison(self):
        # 1. Manual
        start = time.time()
        res_manual = self.manual_gaussian()
        manual_time = time.time() - start

        # 2. OpenCV
        start = time.time()
        res_cv = cv2.GaussianBlur(self.img, (self.Kernel_size, self.Kernel_size), self.sigma, borderType=cv2.BORDER_CONSTANT)
        cv_time = time.time() - start

        # 3. Compare 
        diff_img = cv2.absdiff(res_manual, res_cv)
        hist = cv2.calcHist([diff_img], [0], None, [256], [0, 256])
        max_error = np.max(diff_img)
        print(f"\nMax Pixel Error: {max_error}")

        print(f"\n--- Gaussian Filter ({self.Kernel_size}x{self.Kernel_size}) Results ---")
        print(f"Manual Time: {manual_time:.4f}s")
        print(f"OpenCV Time: {cv_time:.4f}s")

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(res_manual, cmap='gray')
        plt.title("Manual Gaussian")
        plt.subplot(1, 3, 2)
        plt.imshow(diff_img, cmap='gray')
        plt.title("Difference Image")
        plt.subplot(1, 3, 3)
        plt.plot(hist); plt.title("Diff Histogram")
        plt.show()

if __name__ == "__main__":
    
    task = GaussianFilterTask('size_50x50.jpg', Kernel_size=3)
    task.run_comparison()