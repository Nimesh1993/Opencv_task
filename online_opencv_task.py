import cv2
import os
import numpy as np

def process_images(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    total_mask_pixels = 0
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        print(file_path)
        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Skipping non-image file: {file_name}")
            continue
        
        # Create a binary mask where all 3 channels > 200
        binary_mask = np.all(image > 200, axis=2).astype(np.uint8) * 255

        combined = cv2.bitwise_and(image, image, mask=binary_mask)
        
        # Count max (255) pixels in the mask
        max_pixel_count = np.sum(combined == 255)
        total_mask_pixels += max_pixel_count

        # Save the mask as a lossless PNG
        mask_output_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_mask.png")
        cv2.imwrite(mask_output_path, binary_mask)
        print(f"Processed {file_name}, mask saved to {output_dir}, max pixels: {max_pixel_count}")

    # Log the total count of mask pixels across all images
    print(f"Total max pixels across all images: {total_mask_pixels}")

#process_images("C:\Online-test" , "C:\online-test-output")
