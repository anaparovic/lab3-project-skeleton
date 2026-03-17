import os
import urllib.request
import zipfile
import shutil

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(project_root, "dataset")
zip_path = os.path.join(dataset_dir, "tiny-imagenet-200.zip")
extract_path = os.path.join(dataset_dir, "tiny-imagenet-200")

os.makedirs(dataset_dir, exist_ok=True)

url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

if not os.path.exists(zip_path):
    print("Downloading dataset...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download finished.")
else:
    print("Zip already exists.")

if not os.path.exists(extract_path):
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)
    print("Extraction finished.")
else:
    print("Dataset already extracted.")

val_dir = os.path.join(extract_path, "val")
val_images_dir = os.path.join(val_dir, "images")
val_annotations_file = os.path.join(val_dir, "val_annotations.txt")

if os.path.exists(val_images_dir):
    print("Reorganizing validation images...")

    with open(val_annotations_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            img_name = parts[0]
            class_name = parts[1]

            class_dir = os.path.join(val_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            src = os.path.join(val_images_dir, img_name)
            dst = os.path.join(class_dir, img_name)

            if os.path.exists(src):
                shutil.move(src, dst)

    shutil.rmtree(val_images_dir)
    print("Validation images reorganized.")
else:
    print("Validation folder already reorganized.")

print("Dataset is ready.")