import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from resizeimage import resizeimage
import cv2
import sys

# Size of the new images
resized_size = 64

# Color or greyscale
color = True

# Number of images to extract
nb_max_images = 100000

# The original dataset directory filepath
# It needs to contain 2 subfolders (train and test)
original_dataset_dir = sys.argv[1]

# The style is passed as a second argument, as a .txt
# file with the name of an image in each line
style = sys.argv[2].replace(".txt", "")
dataset_dir = style + "_dataset/"
os.makedirs(dataset_dir, exist_ok = True)

# Extract the names of the .jpg files from the .txt file
f = open(style + ".txt", "r")
images_name = f.readlines()
for i in range(0, len(images_name) - 1):
    images_name[i] = images_name[i].rstrip()

# Number of total images
nb_total_images = len(images_name)
nb_total_images = min(nb_total_images, nb_max_images)

# Iterate through all images in the original dataset and
# resize and greyscale the ones that appear in the .txt file
index = 1
for dir in os.listdir(original_dataset_dir):
    if os.path.isdir(original_dataset_dir + dir):
        for file in os.listdir(original_dataset_dir + dir):
            if file in images_name:
                print("Image n°" + str(index) + "/" + str(nb_total_images))

                # Resize
                try:
                    image = Image.open(original_dataset_dir + dir + "/" + file)
                except:
                    # An error occured on some images because they were too large
                    # and PIL refused to open them , fearing a DOS attack
                    # So we just ignore these images
                    continue

                image_resized = resizeimage.resize_cover(image, [resized_size, resized_size])
                image_resized.save(dataset_dir + file, image.format)

                if not color:
                    # Greyscale
                    image = cv2.imread(dataset_dir + file)
                    image_greyscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(dataset_dir + file, image_greyscaled)

                if index >= nb_max_images:
                    break
                else:
                    index += 1
