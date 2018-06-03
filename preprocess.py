import os
import glob
import cv2
import numpy as np

dir_list = os.listdir("C:\\Users\\95327\\Desktop\\image")
for d in dir_list:
    print("begin new dir...")
    image_list = []
    image_dir = os.path.join("C:\\Users\\95327\\Desktop\\image", d, "*.bmp")
    image_list.extend(glob.glob(image_dir))
    if not image_list:
        print("no image found.")
    else:
        size = len(image_list)
        count = 0
        for f in image_list:
            name = d
            image = cv2.imread(f)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            length = min(image.shape[0], image.shape[1]) // 2
            image = cv2.resize(image, (length, length))
            cv2.imwrite(name + "_" + str(count) + ".jpg", image)
            image_rot = image.copy()
            for i in range(3):
                image_rot = np.rot90(image_rot)
                cv2.imwrite(name + "_" + str(count + size * (i + 1)) + ".jpg", image_rot)
            axis = [[1, -1], [-1, 1]]
            for i in range(2):
                image_rev = image[::axis[i][0], ::axis[i][1]]
                cv2.imwrite(name + "_" + str(count + size * (i + 4)) + ".jpg", image_rev)
            image_t = image.T
            cv2.imwrite(name + "_" + str(count + size * 6) + ".jpg", image_t)
            image_t = image_t[::-1, ::-1]
            cv2.imwrite(name + "_" + str(count + size * 7) + ".jpg", image_t)
            count = count + 1
            if (count % 10 == 0):
                print("{} images has been processed...".format(count))