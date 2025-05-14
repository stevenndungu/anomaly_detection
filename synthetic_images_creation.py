
#%%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
random.seed(42)
import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def process_and_superimpose_images(image1_path, image2_path, angle=90):
    """
    Read two images, rotate the first one by specified angle, and superimpose them
    
    Parameters:
    image1_path (str): Path to first image
    image2_path (str): Path to second image
    angle (float): Rotation angle in degrees
    
    Returns:
    tuple: (Original image array, rotated image array, superimposed image array)
    """
    # Read images
    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')
    
    # Store original image array
    original_arr = np.array(img1)
    
    # Rotate first image
    img1_rotated = img1.rotate(angle, expand=True)
    rotated_arr = np.array(img1_rotated)
    
    # Convert second image to array
    arr2 = np.array(img2)
    
    # Resize second image if dimensions don't match after rotation
    if rotated_arr.shape != arr2.shape:
        img2_resized = img2.resize((rotated_arr.shape[1], rotated_arr.shape[0]))
        arr2 = np.array(img2_resized)
    
    # Take maximum values between the arrays
    superimposed = np.maximum(rotated_arr, arr2)
    
    return original_arr, rotated_arr, superimposed

def plot_comparison_with_rotation(img1_path, img2_path, angle=90):
    """
    Plot original, rotated, and superimposed images
    
    Parameters:
    img1_path (str): Path to first image
    img2_path (str): Path to second image
    angle (float): Rotation angle in degrees
    """
    # Process images
    original, rotated, superimposed = process_and_superimpose_images(
        img1_path, img2_path, angle=angle
    )

    print(f"Image 1 shape: {original.shape}, Image 2 shape: {rotated.shape}, Superimposed shape: {superimposed.shape}")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Plot images
    axes[0].imshow(original[:,:,1])
    axes[0].set_title('Original Image 1')
    axes[0].axis('off')
    
    axes[1].imshow(rotated[:,:,1])
    axes[1].set_title(f'Image 1 (Rotated {angle}°)')
    axes[1].axis('off')
    
    axes[2].imshow(np.array(Image.open(img2_path).convert('RGB'))[:,:,1])
    axes[2].set_title('Image 2')
    axes[2].axis('off')
    
    axes[3].imshow(superimposed[:,:,1])
    axes[3].set_title('Superimposed Image')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()

def delete_image_os(image_path):
    try:
        os.remove(image_path)
        print(f"Successfully deleted {image_path}")
    except FileNotFoundError:
        print(f"Error: {image_path} not found")
    except PermissionError:
        print(f"Error: Permission denied to delete {image_path}")
#%%
#FRI
# image1 = [59,5,0]
# image2 = [36,7,10]
nn = 10
image_nums = list(range(len(glob.glob('../FRGADB_cleaned_with_synthetic_valid_data/FRI/train/*'))))
image1_fri = random.sample(image_nums, nn)

diff = set(image_nums) - set(image1_fri)

image2_fri = random.sample(diff, nn)
#%%
for i in range(len(image1_fri)):
   print(i)
   image1_path = f'../FRGADB_cleaned_with_synthetic_valid_data/FRI/train/FRI_{image1_fri[i]}.jpg'  
   image2_path = f'../FRGADB_cleaned_with_synthetic_valid_data/FRI/train/FRI_{image2_fri[i]}.jpg'
   
   plot_comparison_with_rotation(image1_path, image2_path, angle=90) 
   _, _, superimposed = process_and_superimpose_images(image1_path, image2_path, angle=90)
   Image.fromarray(superimposed).save(f'../FRGADB_cleaned_with_synthetic_valid_data/RRG/test/a_FRI_{image1_fri[i]}_{image2_fri[i]}_rot_90.jpg')

   plot_comparison_with_rotation(image1_path, image2_path, angle=180) 
   _, _, superimposed = process_and_superimpose_images(image1_path, image2_path, angle=180)
   Image.fromarray(superimposed).save(f'../FRGADB_cleaned_with_synthetic_valid_data/RRG/test/a_FRI_{image1_fri[i]}_{image2_fri[i]}_rot_180.jpg')

   plot_comparison_with_rotation(image1_path, image2_path, angle=270) 
   _, _, superimposed = process_and_superimpose_images(image1_path, image2_path, angle=270)
   Image.fromarray(superimposed).save(f'../FRGADB_cleaned_with_synthetic_valid_data/RRG/test/a_FRI_{image1_fri[i]}_{image2_fri[i]}_rot_270.jpg')
   

   # delete_image_os(image1_path)
   # delete_image_os(image2_path)

#FRII
image1 = [2,14,103]
image2 = [144,257,114]

#%%
image1 = [103]
image2 = [114]
image1 = [2,14,103]

image_nums = list(range(len(glob.glob('../FRGADB_cleaned_with_synthetic_valid_data/FRII/train/*'))))
image1_frii = random.sample(image_nums, nn)

diff = set(image_nums) - set(image1_frii)

image2_frii = random.sample(diff, nn)
#image2 = [144,257,114]

#%%
for i in range(len(image1_frii)):
   print(i)
   image1_path = f'../FRGADB_cleaned_with_synthetic_valid_data/FRII/train/FRII_{image1_frii[i]}.jpg'
   image2_path = f'../FRGADB_cleaned_with_synthetic_valid_data/FRII/train/FRII_{image2_frii[i]}.jpg'

   plot_comparison_with_rotation(image1_path, image2_path, angle=90) 
   _, _, superimposed = process_and_superimpose_images(image1_path, image2_path, angle=90)
   Image.fromarray(superimposed).save(f'../FRGADB_cleaned_with_synthetic_valid_data/RRG/test/a_FRII_{image1_frii[i]}_{image2_frii[i]}_rot_90.jpg')

   plot_comparison_with_rotation(image1_path, image2_path, angle=180) 
   _, _, superimposed = process_and_superimpose_images(image1_path, image2_path, angle=180)
   Image.fromarray(superimposed).save(f'../FRGADB_cleaned_with_synthetic_valid_data/RRG/test/a_FRII_{image1_frii[i]}_{image2_frii[i]}_rot_180.jpg')

   plot_comparison_with_rotation(image2_path, image1_path, angle=270) 
   _, _, superimposed = process_and_superimpose_images(image1_path, image2_path, angle=270)
   Image.fromarray(superimposed).save(f'../FRGADB_cleaned_with_synthetic_valid_data/RRG/test/a_FRII_{image1_frii[i]}_{image2_frii[i]}_rot_270.jpg')

   delete_image_os(image1_path)               
   delete_image_os(image2_path)

# %%
# FRI & FRII
friis = image1_frii + image2_frii
image_nums = set(list(range(len(glob.glob('../FRGADB_cleaned_with_synthetic_valid_data/FRII/train/*'))))) - set(friis)
image1_frii = random.sample(image_nums, nn)

fris = image1_fri + image2_fri
image_nums = set(list(range(len(glob.glob('../FRGADB_cleaned_with_synthetic_valid_data/FRI/train/*'))))) - set(fris)
image2_fri = random.sample(image_nums, nn)
#image2 = [144,257,114]
                                                                
#%%
for i in range(len(image1_frii)):
   print(i)
   image1_path = f'../FRGADB_cleaned_with_synthetic_valid_data/FRII/train/FRII_{image1_frii[i]}.jpg'
   image2_path = f'../FRGADB_cleaned_with_synthetic_valid_data/FRI/train/FRI_{image2_fri[i]}.jpg'

   plot_comparison_with_rotation(image1_path, image2_path, angle=90) 
   _, _, superimposed = process_and_superimpose_images(image1_path, image2_path, angle=90)
   Image.fromarray(superimposed).save(f'../FRGADB_cleaned_with_synthetic_valid_data/RRG/test/a_FRI_FRII_{image2_fri[i]}_{image1_frii[i]}_rot_90.jpg')


   plot_comparison_with_rotation(image1_path, image2_path, angle=180) 
   _, _, superimposed = process_and_superimpose_images(image1_path, image2_path, angle=180)
   Image.fromarray(superimposed).save(f'../FRGADB_cleaned_with_synthetic_valid_data/RRG/test/a_FRI_FRII_{image2_fri[i]}_{image1_frii[i]}_rot_180.jpg')

   
   plot_comparison_with_rotation(image1_path, image2_path, angle=270) 
   _, _, superimposed = process_and_superimpose_images(image1_path, image2_path, angle=270)
   Image.fromarray(superimposed).save(f'../FRGADB_cleaned_with_synthetic_valid_data/RRG/test/a_FRI_FRII_{image2_fri[i]}_{image1_frii[i]}_rot_270.jpg')
 
#    delete_image_os(image1_path)
#    delete_image_os(image2_path)



#%%


# %%
import pandas as pd
from utils import *
path = f'COSFIREdescriptor_1.mat'
dic_labels = {'RRG': 2, 'XRG': 3, 'FRI': 0, 'FRII': 1}
num_filters = 100

def process_data(path ,dic_labels, random_state=42):
   """
   Process train and test data into train, test and validation sets
   """
   # Read data
   df_train, df_test = get_data(path, dic_labels)
   
   # Sample validation data from train
   frii_valid = df_train[df_train['label_code'] == 1].sample(n=53, random_state=random_state)
   fri_valid = df_train[df_train['label_code'] == 0].sample(n=22, random_state=random_state)
   
   # Get remaining train data
   selected_indices = list(frii_valid.index) + list(fri_valid.index)
   train_data = df_train.drop(index=selected_indices).reset_index(drop=True)
   
   # Process test data by label
   test_splits = {}
   for label, code in {'FRII': 1, 'FRI': 0, 'XRG': 3, 'RRG': 2}.items():
       test_splits[label] = df_test[df_test['label_code'] == code].reset_index(drop=True)
   
   # Create final datasets
   test_data = pd.concat([
       test_splits['FRII'], 
       test_splits['FRI'],
       test_splits['XRG'],
       test_splits['RRG'].tail(13)
   ])
   
   valid_data = pd.concat([
       test_splits['RRG'].head(36),
       frii_valid,
       fri_valid
   ]).reset_index(drop=True)
   
   return train_data, test_data, valid_data

train_data, test_data, valid_data = process_data(path = f'COSFIREdescriptor_16.mat', 
                                              dic_labels = dic_labels,
                                              random_state=42)
# %%
train_data.shape,test_data.shape, valid_data.shape
print('Train',train_data.label_code.value_counts())
print('Test',test_data.label_code.value_counts())
print('Valid',valid_data.label_code.value_counts())
# %%
