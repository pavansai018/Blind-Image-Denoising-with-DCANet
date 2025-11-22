import os
import shutil

orig_path = r'/home/pavan/Downloads/SUTD/Project/DCANet/dataset/coco2017/train2017'

val_img_path = r'/home/pavan/Downloads/SUTD/Project/DCANet/dataset/train/dirty'
out_path = r'/home/pavan/Downloads/SUTD/Project/DCANet/dataset/train/clean/'
val_images = os.listdir(val_img_path)

for image in val_images:
    shutil.copy(src=os.path.join(orig_path, image), dst=os.path.join(out_path, image))