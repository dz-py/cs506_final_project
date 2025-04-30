import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

input_dir = os.path.join(os.getcwd(), "food-101", "images_processed")
output_dir = os.path.join(os.getcwd(), "food-101", "dataset_split")

def split_dataset(input_dir, output_dir, train_ratio=0.8, seed=42):
    categories = [cat for cat in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, cat))]
    data = []
    for cat in categories:
        for img in os.listdir(os.path.join(input_dir, cat)):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                data.append({'category': cat, 'img': img})
    df = pd.DataFrame(data)
    train_df, val_df = train_test_split(df, test_size=1-train_ratio, stratify=df['category'], random_state=seed)

    for split, split_df in [('train', train_df), ('val', val_df)]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for _, row in split_df.iterrows():
            src = os.path.join(input_dir, row['category'], row['img'])
            dst_dir = os.path.join(split_dir, row['category'])
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, row['img'])
            if not os.path.exists(dst):
                shutil.copy(src, dst)

if __name__ == "__main__":
    split_dataset(input_dir, output_dir)
