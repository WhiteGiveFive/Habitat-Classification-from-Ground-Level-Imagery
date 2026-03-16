import os
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Define the function to split and move images
def stratified_train_test_split(index_path, image_folder, test_folder, n_splits=4, random_state=42):
    """
    Split the training and test sets and move the test image files to a new folder.
    :param index_path: the loaded index file which contains the successfully loaded images info
    :param image_folder: the directory which contains all the images
    :param test_folder: the directory to store all the test images
    :param n_splits:
    :param random_state:
    :return:
    """
    # Load the csv file into a Pandas DataFrame
    df = pd.read_csv(index_path)

    # Check if the necessary columns exist in the DataFrame.
    # We use the generated file from the function load_images_from_folder in dataset.py, loaded index file.
    # So we first need to run the dataloader.py once to generate the loaded data.
    if not {'file_names', 'plot_labels', 'plot_idx'}.issubset(df.columns):
        raise ValueError("Index file must contain 'file_names', 'plot_labels', and 'plot_idx' columns.")

    # Split data using StratifiedGroupKFold
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(skf.split(df, df['plot_labels'], groups=df['plot_idx']))

    # Create training and test DataFrames
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # Move test images to the specified test folder
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    for _, row in test_df.iterrows():
        src_path = os.path.join(image_folder, row['file_names'])  # Source image path
        dst_path = os.path.join(test_folder, row['file_names'])   # Destination path

        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"Warning: File {src_path} does not exist.")

    # Summary statistics
    print("Split Summary:")
    print(f"Number of training images: {len(train_df)}")
    print(f"Number of test images: {len(test_df)}")

if __name__ == '__main__':
    loaded_index_path = os.path.join('./CS_Xplots_2019_2023', 'loaded_CS_Xplots_2019_23_NEW02OCT24.csv')
    all_image_folder = './CS_Xplots_2019_2023'
    testset_folder = './CS_Xplots_2019_2023_test'

    stratified_train_test_split(loaded_index_path, all_image_folder, testset_folder)