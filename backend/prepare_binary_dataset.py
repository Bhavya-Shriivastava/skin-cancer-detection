import os
import shutil

INPUT_DIR = "dataset"
OUTPUT_DIR = "dataset_binary"

# Cancer classes
cancer_classes = ["mel", "bcc", "akiec"]

# Non-cancer classes
non_cancer_classes = ["nv", "bkl", "df", "vasc"]

for split in ["train", "val"]:
    for category in ["cancer", "non_cancer"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, category), exist_ok=True)

    for cls in cancer_classes:
        src = os.path.join(INPUT_DIR, split, cls)
        dst = os.path.join(OUTPUT_DIR, split, "cancer")

        for img in os.listdir(src):
            shutil.copy(os.path.join(src, img), os.path.join(dst, img))

    for cls in non_cancer_classes:
        src = os.path.join(INPUT_DIR, split, cls)
        dst = os.path.join(OUTPUT_DIR, split, "non_cancer")

        for img in os.listdir(src):
            shutil.copy(os.path.join(src, img), os.path.join(dst, img))

print("Binary dataset ready!")