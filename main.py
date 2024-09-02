import cv2
import numpy as np
import os
from cellpose import models, io
import pandas as pd
import glob
import sys
import argparse
from pathlib import Path


# Define the paths to the input images and the output directories.
output_path = 'data/newdata/datas/output_segments'
output_merge = 'data/newdata/datas/output_merge'
image_folder = 'data/newdata/datas/purple_img'

parser = argparse.ArgumentParser(description='Perform cell analysis on a set of images and save the results to an Excel file.')
parser.add_argument('--image_folder', type=str, default=image_folder, help='The folder containing the input images.')
parser.add_argument('--output_path', type=str, default=output_path, help='The folder to save the segmented images.')
parser.add_argument('--output_merge', type=str, default=output_merge, help='The folder to save the merged images.')


def determine_class(file_name):
    """
    Determines the class based on the given file name.
    Parameters:
    file_name (str): The name of the file.
    Returns:
    str: The determined class based on the file name.
    """
     
    if "name" in file_name and "name+" in file_name and "group" in file_name:
        return "name_name+_group1"
    if "name" in file_name and "name-" in file_name and "group" in file_name:
        return "name_name-_group2"
    if "name" in file_name and "name+" in file_name and "group" in file_name:
        return "name_name+_group3"
    if "name" in file_name and "name+" in file_name and "group" in file_name:
        return "name_name+_group4"
    


def save_segmented_image(original_image, masks, file_name):
    """
    Save the segmented image based on the original image and masks.
    Parameters:
    original_image (numpy.ndarray): The original image.
    masks (numpy.ndarray): The masks representing the segmented regions.
    file_name (str): The name of the file to save the segmented image.
    Returns:
    None
    """


    segmented_image = np.zeros((original_image.shape[0], original_image.shape[1], 3), dtype=np.uint8)


    for mask_id in np.unique(masks):
        if mask_id == 0:
            continue
        mask = masks == mask_id
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        segmented_image[mask] = color

    save_path = os.path.join(output_path, f'segmented_{file_name}')
    cv2.imwrite(save_path, segmented_image)
    save_merged_image(original_image, segmented_image, file_name)
    print(f'Saved segmented image: {save_path}')
def save_merged_image(original_image, segmented_image, file_name):
    """
    Save the merged image of the original image and the segmented image.
    Parameters:
    original_image (numpy.ndarray): The original image.
    segmented_image (numpy.ndarray): The segmented image.
    file_name (str): The name of the file to be saved.
    Returns:
    None
    """


    if len(original_image.shape) == 2:  
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    combined_image = cv2.addWeighted(original_image, 0.6, segmented_image, 0.4, 0)
    
    save_path = os.path.join(output_merge, f'merged_{file_name}')
    cv2.imwrite(save_path, combined_image)
    print(f'Saved merged image: {save_path}')

def cellpose_segmentation(img,file,org_img, diameter,model_type='cyto', channels=[0, 0], save_path=output_path):
    def cellpose_segmentation(img, file, org_img, diameter, model_type='cyto', channels=[0, 0], save_path=output_path):
        """
        Perform cell segmentation using the Cellpose algorithm.
        Args:
            img: The input image to be segmented.
            file: The file path of the input image.
            org_img: The original image before any modifications.
            diameter: The estimated diameter of the cells in pixels.
            model_type: The type of model to use for segmentation. Default is 'cyto'.
            channels: The channels to use for segmentation. Default is [0, 0].
            save_path: The path to save the segmented masks. Default is output_path.
        Returns:
            masks: The segmented masks representing the cells.
        """


    model = models.Cellpose(gpu=True, model_type=model_type)




    masks, flows, styles, diams = model.eval(img, diameter=diameter, channels=channels)


    if save_path is not None:
        io.masks_flows_to_seg(img, masks, flows, save_path)
   
    return masks


def calc_area(mask):
    return np.sum(mask)
def calc_brightness(img, mask):
    cell_pixels = img[mask]
    mean_brightness = np.mean(cell_pixels)
    return mean_brightness

def calc_irregularity(mask):
    """
    Calculates the irregularity of a given mask.

    Parameters:
    - mask: numpy.ndarray
        The input mask representing the object of interest.

    Returns:
    - irregularity: float
        The calculated irregularity of the object.
    - cx: int
        The x-coordinate of the centroid of the object.
    - cy: int
        The y-coordinate of the centroid of the object.
    """


    mask_uint8 = mask.astype(np.uint8) * 255


    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0
    contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return 0
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    angles = [0, 72, 144, 216, 288]
    distances = []
    for angle in angles:
        radian = np.deg2rad(angle)
        x = int(cx + 1000 * np.cos(radian))
        y = int(cy + 1000 * np.sin(radian))
        line = cv2.line(np.zeros_like(mask_uint8), (cx, cy), (x, y), 255, 1)
        intersection = cv2.bitwise_and(line, mask_uint8)
        points = np.column_stack(np.where(intersection == 255))
        if points.size > 0:
            distance = np.linalg.norm(points[0] - np.array([cx, cy]))
            distances.append(distance)

    if len(distances) > 0:
        max_distance = max(distances)
        normalized_distances = [d / max_distance for d in distances]
        irregularity = np.std(normalized_distances)

        return irregularity ,cx ,cy

    return 0
   
    


def calc_circularity(mask):
    """
    Calculates the circularity of a given mask.
    Parameters:
    - mask: numpy.ndarray
        The binary mask representing the object.
    Returns:
    - circularity: float
        The circularity value of the object.
    """

    mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0
    return circularity


def main(args):
    """
    Perform cell analysis on a set of images and save the results to an Excel file.
    This function reads a set of images from a specified folder, performs cell segmentation using the Cellpose algorithm,
    calculates various cell properties such as area, circularity, brightness, and irregularity, and saves the results
    to an Excel file.
    Returns:
        None
    Raises:
        None
    """
    
    results = []
    for file in glob.glob(os.path.join(args.image_folder, '*.jpg')):
        org_img = cv2.imread(file)
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        mask_small = cellpose_segmentation(img, file=file, org_img=org_img, diameter=20)
        mask_large =cellpose_segmentation(img,file=file,org_img=org_img,diameter=40) # 50 is fine for the large cells
        combined_mask = np.maximum(mask_small, mask_large)
        save_segmented_image(org_img, combined_mask, os.path.basename(file))



        

        

        class_label = determine_class(os.path.basename(file))

        cell_counter = 1
        for mask_id in np.unique(combined_mask):
            if mask_id == 0:
                continue
            mask = combined_mask == mask_id
            
            area = calc_area(mask)
            
            circularity = calc_circularity(mask)

            mean_brightness = calc_brightness(img, mask)
            
  
            irregularity, cx, cy = calc_irregularity(mask)
            
            results.append({
                'File Name': os.path.basename(file),
                'Class': class_label,  
                'Cell Number': cell_counter,
                'Mean Brightness': mean_brightness,
                'Irregularity': irregularity,
                'Area': area,
                'Circularity': circularity,
                'X coordinate': cx,
                'Y coordinate': cy
            })

            cell_counter += 1
    
    df = pd.DataFrame(results)
    output_excel_path = os.path.join(output_path, 'pcell_analysis_results_with_class.xlsx')
    df.to_excel(output_excel_path, index=False)
    print(f'Results saved to Excel file: {output_excel_path}')


    
if __name__ == '__main__':

    args = parser.parse_args()
    main(args)

    

    
