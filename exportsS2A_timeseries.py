import argparse
import ee
import google.auth
from google.api_core import retry
import requests
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from osgeo import gdal
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import subprocess
import datetime
import os

# Initialize Earth Engine
ee.Initialize(project='tony-1122')

patch_size = 512
NUM_THREADS = 8  # Adjust based on system capability
yearly = 2023

@retry.Retry(deadline=1 * 60)
def get_patch(image: ee.Image, lonlat: tuple[float, float], patch_size: int, scale: int):
    """Download a patch of the given image using a buffered region."""
    try:
        point = lonlat
        url = image.getDownloadURL({
            "region": point.buffer(scale * patch_size / 2, 1).bounds(1),
            "dimensions": [patch_size, patch_size],
            "format": "NPY",
        })
        response = requests.get(url)
        response.raise_for_status()

        patch = np.load(io.BytesIO(response.content), allow_pickle=True)

        # Validate shape before returning
        if patch.ndim == 2:  # Expected (H, W) for single-band NDVI
            patch = np.expand_dims(patch, axis=-1)  # Convert to (H, W, 1)
        elif patch.ndim != 3:
            raise ValueError(f"Unexpected patch shape {patch.shape}")

        return patch, point.buffer(scale * patch_size / 2, 1).bounds(1)

    except Exception as e:
        print(f"Error downloading patch: {e}")
        return None, None  # Return None if patch fails

def write_output(raster, out_file, patch_size, coords):
    """Write raster data to a GeoTIFF file."""
    if raster is None:
        print(f"Skipping {out_file} due to missing data.")
        return

    xmin, xmax, ymin, ymax = coords[0][0], coords[1][0], coords[0][1], coords[2][1]
    driver = gdal.GetDriverByName("GTiff")
    temp_file = out_file.replace(".tif", "_temp.tif")

    out_raster = driver.Create(temp_file, patch_size, patch_size, 1, gdal.GDT_Float32)
    out_raster.SetProjection("EPSG:4326")
    out_raster.SetGeoTransform((xmin, (xmax - xmin) / patch_size, 0, ymax, 0, -(ymax - ymin) / patch_size))

    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(raster.squeeze())  # Ensure it's 2D

    out_raster = None  # Close dataset
    gdal_translate_to_cog(temp_file, out_file)

def gdal_translate_to_cog(temp_file, out_file):
    """Convert a TIFF file to a Cloud Optimized GeoTIFF (COG)."""
    try:
        subprocess.run([
            "gdal_translate", temp_file, out_file,
            "-co", "TILED=YES",
            "-co", "COPY_SRC_OVERVIEWS=YES",
            "-co", "COMPRESS=DEFLATE"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {temp_file} to COG: {e}")
    finally:
        subprocess.run(["rm", temp_file])  # Remove temp file

def get_data(i, location):
    """Retrieve and download time-series images per tile."""
    my_samples = ee.FeatureCollection(location).randomColumn("random").sort("random")
    sample = ee.Feature(my_samples.toList(my_samples.size()).get(i))
    
    geometry = sample.geometry()
    lonlat = geometry
    
    CLEAR_THRESHOLD = 0.80
    QA_BAND = 'cs_cdf'
    
    start_date = ee.Date.fromYMD(yearly, 1, 1)
    end_date = ee.Date.fromYMD(yearly, 12, 31)

    def mask_data(img):
        return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD))

    def cal_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)

    s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(geometry).filterDate(start_date, end_date).sort("system:time_start")
    csPlus = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED").filterBounds(geometry).filterDate(start_date, end_date)
    s2_collection = s2.linkCollection(csPlus, [QA_BAND]).map(mask_data).map(cal_ndvi)

    Indices = ['NDVI']
    images = s2_collection.select(Indices).toList(s2_collection.size())

    # Create subfolder for each tile
    tile_folder = f"/content/Phenological_project/Images/s2/tile_{str(i).zfill(3)}"
    os.makedirs(tile_folder, exist_ok=True)

    for j in range(images.size().getInfo()):
        img = ee.Image(images.get(j))
        date_str = datetime.datetime.utcfromtimestamp(img.date().millis().getInfo() / 1000).strftime("%Y%m%d")

        img_path_before = os.path.join(tile_folder, f"{date_str}.tif")

        patch, bounds = get_patch(img, lonlat, patch_size, 10)

        if patch is None:
            print(f"Skipping {img_path_before} due to missing data.")
            continue

        image_patch = structured_to_unstructured(patch) if patch.ndim == 3 else patch
        coords = np.array(bounds.getInfo().get("coordinates"))[0]

        write_output(image_patch, img_path_before, patch_size, coords)
        print(f"Saved tile {i} date {date_str}: {image_patch.shape}")

def process_data(i, location):
    """Thread-safe function to process each tile."""
    try:
        get_data(i, location)
    except Exception as e:
        print(f"Error processing tile {i}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download time-series satellite data for tiles.")
    parser.add_argument("location", type=str, help="Earth Engine FeatureCollection location path")
    parser.add_argument("year", type=int, help="Year for data processing")
    args = parser.parse_args()

    location = args.location  
    global yearly
    yearly = args.year

    my_samples = ee.FeatureCollection(location)
    total_samples = my_samples.size().getInfo()  # Get total number of tiles

    print(f"Processing {total_samples} tiles for the year {yearly}")

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(process_data, i, location) for i in range(total_samples)]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Exception in thread: {e}")

if __name__ == '__main__':
    main()
