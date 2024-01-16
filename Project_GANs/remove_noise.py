import glob
import shutil
import cv2


def apply_gaussian_blur_filter(image, sigma_par, index):
    blurred_lena = cv2.GaussianBlur(src=image, ksize=(3, 3), sigmaX=sigma_par, sigmaY=sigma_par,
                                    borderType=cv2.BORDER_REPLICATE)
    cv2.imwrite(f".\\denoised_images\\{index}.jpg", blurred_lena)


# for i, img in enumerate(glob.glob(".\\images (7)\\*.jpg")):
#     img = cv2.imread(img)
#     apply_gaussian_blur_filter(img, 1, i)


shutil.make_archive("images", "zip", "denoised_images")

