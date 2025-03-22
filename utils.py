import os
import glob
import shutil

def class_image_base_name(img_base_path, class_names):
    for class_name in class_names:
        os.makedirs(os.path.join(img_base_path, class_name), exist_ok=True)
        img_list = glob.glob(os.path.join(img_base_path, "*.jpg"))
        for img_path in img_list:
            if class_name in img_path:
                shutil.move(img_path, os.path.join(img_base_path, class_name, os.path.basename(img_path)))

if __name__ == "__main__":
    img_base_path = '/home/code/experiment/modal/small_batch/test'
    class_names = ['dog', 'cat']
    class_image_base_name(img_base_path, class_names)