import os
import shutil
from sklearn.model_selection import train_test_split

def create_split_folders(base_dir, subfolders):
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_dir, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)

def copy_files(img_files, mask_files, dest_img_dir, dest_mask_dir):
    for img_file, mask_file in zip(img_files, mask_files):
        # Copy images to the imgs folder
        shutil.copy(img_file, os.path.join(dest_img_dir, os.path.basename(img_file)))
        # Copy masks to the masks folder
        shutil.copy(mask_file, os.path.join(dest_mask_dir, os.path.basename(mask_file)))

def split_data(img_dir, mask_dir, split_ratio):
    # Get all image files (sorted alphabetically)
    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.nii.gz'))])
    
    # Get all mask files (sorted alphabetically)
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.nii.gz'))])

    # Ensure same number of img and mask files
    assert len(img_files) == len(mask_files), f"Number of images and masks do not match! Images: {len(img_files)}, Masks: {len(mask_files)}"

    print(f"Total images: {len(img_files)}, Total masks: {len(mask_files)}")

    # Adjust mask filenames by removing the 'labels-' prefix to match image filenames
    mask_files_dict = {}
    for mask_file in mask_files:
        base_mask_name = os.path.basename(mask_file).replace("labels-", "")
        mask_files_dict[base_mask_name] = mask_file

    # Create the list of masks that match image files
    mask_files_matched = [mask_files_dict[os.path.basename(img_file)] for img_file in img_files if os.path.basename(img_file) in mask_files_dict]

    # Check for mismatched files
    if len(mask_files_matched) != len(img_files):
        print(f"Error: Some images do not have corresponding masks.")
        return

    # Split the data
    total_files = len(img_files)
    total_ratio = sum(split_ratio)
    trainA_size = int((split_ratio[0] / total_ratio) * total_files)
    test_size = int((split_ratio[1] / total_ratio) * total_files)
    validation_size = total_files - trainA_size - test_size

    # First split: TrainA and the rest (test + validation)
    trainA_split, temp_split = train_test_split(img_files, train_size=trainA_size, random_state=42)

    # Second split: Split temp into test and validation
    test_split, validation_split = train_test_split(temp_split, test_size=validation_size / (test_size + validation_size), random_state=42)

    # Create the target directories under 'datasets/'
    create_split_folders('datasets/trainA_500', ['imgs', 'masks'])
    create_split_folders('datasets/GT_data_vessels_testing_100', ['imgs', 'masks'])
    create_split_folders('datasets/GT_data_vessels_stopp_crit', ['imgs', 'masks'])

    # Copy the files to their respective destinations
    copy_files(trainA_split, [mask_files_dict[os.path.basename(img)] for img in trainA_split], 'datasets/trainA_500/imgs', 'datasets/trainA_500/masks')
    copy_files(test_split, [mask_files_dict[os.path.basename(img)] for img in test_split], 'datasets/GT_data_vessels_testing_100/imgs', 'datasets/GT_data_vessels_testing_100/masks')
    copy_files(validation_split, [mask_files_dict[os.path.basename(img)] for img in validation_split], 'datasets/GT_data_vessels_stopp_crit/imgs', 'datasets/GT_data_vessels_stopp_crit/masks')

    # Check that the sum of splits equals the initial number of images
    assert len(trainA_split) + len(test_split) + len(validation_split) == total_files, "Split sizes do not match total data size."

    print(f"Data split completed. Train: {len(trainA_split)}, Test: {len(test_split)}, Validation: {len(validation_split)}")

if __name__ == '__main__':
    img_dir = 'datasets/all_data/imgs'
    mask_dir = 'datasets/all_data/masks'

    # Define the split ratio: 500:100:10
    split_ratio = [500, 100, 10]

    split_data(img_dir, mask_dir, split_ratio)
