import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, image_path, kernel_size=3):
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None:
            raise ValueError("Image not found!")
        self.k_size = kernel_size
        self.pad = kernel_size // 2

    def crop_image(self, start_y, start_x, height, width):
        end_y = start_y+height
        end_x = start_x+width
        self.img = self.img[start_y:end_y, start_x:end_x]

    def manual_mean_filter(self):
        padded = cv2.copyMakeBorder(
            self.img, self.pad, self.pad, self.pad, self.pad, 
            cv2.BORDER_CONSTANT, value=0
        )
        
        h, w = self.img.shape
        output = np.zeros((h, w), dtype=np.uint8)
        normalizer = self.k_size * self.k_size

        for i in range(h):
            for j in range(w):
                roi = padded[i : i + self.k_size, j : j + self.k_size]
                pixel_sum = np.sum(roi)
                output[i, j] = int(round(pixel_sum / normalizer))
                
        return output

    def run_comparison(self):
        # --- 1. Manual Method ---
        start_time = time.time()
        res_manual = self.manual_mean_filter()
        end_time = time.time()
        manual_duration = end_time - start_time
        print(f"Manual Time: {manual_duration:.4f} sec")

        # --- 2. OpenCV Method ---
        start_time = time.time()
        res_cv = cv2.blur(self.img, (self.k_size, self.k_size), borderType=cv2.BORDER_CONSTANT)
        end_time = time.time()
        cv_duration = end_time - start_time
        print(f"OpenCV Time: {cv_duration:.4f} sec")
        
        diff_img = cv2.absdiff(res_manual, res_cv)
        hist = cv2.calcHist([diff_img], [0], None, [256], [0, 256])
        max_error = np.max(diff_img)
        print(f"\nMax Pixel Error: {max_error}")

        plt.figure(figsize=(10, 4)) 
        plt.subplot(1, 3, 1)
        plt.imshow(res_manual, cmap='gray')
        plt.title("Manual Mean")
        plt.subplot(1, 3, 2)
        plt.imshow(diff_img, cmap='gray', vmin=0, vmax=255)
        plt.title("Difference Image")
        plt.subplot(1, 3, 3)
        plt.plot(hist)
        plt.title("Diff Histogram")
        plt.show()

if __name__ == "__main__":
    processor = ImageProcessor('size_50x50.jpg', kernel_size=3)
    border_offset = processor.pad
    processor.crop_image(border_offset, border_offset, 50, 50) 
    processor.run_comparison()