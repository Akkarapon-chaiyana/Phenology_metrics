import os
import subprocess
import argparse

def convert_to_cog(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # List all files in the input directory
    file_list = os.listdir(input_directory)

    for item in file_list:
        # Only process .tif files
        if item.endswith('.tif'):
            filename = item.split('.')[0]

            # Ensure the conversion is completed before moving on
            print(f"Converting {filename} to COG format...")
            subprocess.run(['gdal_translate', os.path.join(input_directory, f'{filename}.tif'),
                            os.path.join(output_directory, f'{filename}.tif'), '-of', 'COG'], check=True)
            print(f"Process completed for {filename}.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert .tif images to Cloud Optimized GeoTIFF (COG) format.')
    
    # Use positional arguments for input and output directories
    parser.add_argument('input_directory', type=str, help='Path to the input directory containing .tif files.')
    parser.add_argument('output_directory', type=str, help='Path to the output directory where COG files will be saved.')

    # Parse arguments
    args = parser.parse_args()

    # Call the conversion function with the provided input and output directories
    convert_to_cog(args.input_directory, args.output_directory)

if __name__ == '__main__':
    main()
