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

# Initialize Earth Engine with the default project and authentication
ee.Initialize(project = 'tony-1122')

patch_size = 512
NUM_THREADS = 8  # You can adjust this based on your system capabilities

yearly = 2023

# Retry mechanism for rate-limited requests (HTTP 429)
@retry.Retry(deadline=1 * 60)  # Retry for up to 1 minute
def get_patch(image: ee.Image, lonlat: tuple[float, float], patch_size: int, scale: int) -> np.ndarray:
    point = lonlat
    url = image.getDownloadURL({
        "region": point.buffer(scale * patch_size / 2, 1).bounds(1),
        "dimensions": [patch_size, patch_size],
        "format": "NPY",
    })
    response = requests.get(url)
    if response.status_code == 429:
        raise requests.exceptions.RequestException("Too many requests, rate limited.")
    response.raise_for_status()
    return np.load(io.BytesIO(response.content), allow_pickle=True), point.buffer(scale * patch_size / 2, 1).bounds(1)

def writeOutput(raster, out_file, patch_size, coords):
    xmin, xmax, ymin, ymax = coords[0][0], coords[1][0], coords[0][1], coords[2][1]
    coords = [xmin, ymin, xmax, ymax]
    
    driver = gdal.GetDriverByName("GTiff")
    l = raster.shape[2]
    temp_file = out_file.replace(".tif", "_temp.tif")  # Create a temporary file to write non-COG

    out_raster = driver.Create(temp_file, patch_size, patch_size, l, gdal.GDT_Float32)
    out_raster.SetProjection("EPSG:4326")
    out_raster.SetGeoTransform((xmin, (xmax - xmin) / patch_size, 0, ymax, 0, -(ymax - ymin) / patch_size))
    
    for i in range(0, l):
        out_band = out_raster.GetRasterBand(i+1)
        out_band.WriteArray(raster[:, :, i])

    out_raster = None  # Close the dataset to write it to disk

    # Convert to COG using gdal_translate
    gdal_translate_to_cog(temp_file, out_file)
    # print(f"Converted {out_file} to Cloud Optimized GeoTIFF.")

def writeOutputSingle(raster, out_file, patch_size, coords):
    xmin, xmax, ymin, ymax = coords[0][0], coords[1][0], coords[0][1], coords[2][1]
    coords = [xmin, ymin, xmax, ymax]
    
    driver = gdal.GetDriverByName("GTiff")
    temp_file = out_file.replace(".tif", "_temp.tif")  # Create a temporary file to write non-COG

    out_raster = driver.Create(temp_file, patch_size, patch_size, 1, gdal.GDT_Int16)
    out_raster.SetProjection("EPSG:4326")
    out_raster.SetGeoTransform((xmin, (xmax - xmin) / patch_size, 0, ymax, 0, -(ymax - ymin) / patch_size))
    
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(raster)
    out_raster = None  # Close the dataset to write it to disk

    # Convert to COG using gdal_translate
    gdal_translate_to_cog(temp_file, out_file)
    # print(f"Converted {out_file} to Cloud Optimized GeoTIFF.")

def gdal_translate_to_cog(temp_file, out_file):
    try:
        subprocess.run([
            "gdal_translate", temp_file, out_file,
            "-co", "TILED=YES",
            "-co", "COPY_SRC_OVERVIEWS=YES",
            "-co", "COMPRESS=DEFLATE"
        ], check=True)
        #print(f"Successfully converted {temp_file} to COG: {out_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {temp_file} to COG: {e}")
    finally:
        # Optionally, remove the temp file after conversion
        subprocess.run(["rm", temp_file])

def addMyClass(feat):
    year = yearly
    month = 1
    d = ee.Date.fromYMD(year, month, 1)
    return feat.set("class", 1).set("system:time_start", d).set("year", year).set("month", month)

def getData(i, location): 
    mySamples = ee.FeatureCollection(location).randomColumn("random").sort("random").map(addMyClass)
    samples = mySamples.toList(mySamples.size())  # Adjusting to handle all samples
    sample = ee.Feature(samples.get(i))
        
    img_path_before = f"/content/Phenological_project/Images/s2/{str(i).zfill(5)}.tif"
    img_path_label = f"/content/Phenological_project/Images/label/{str(i).zfill(5)}.tif"

    geometry = sample.geometry()
    lonlat = geometry

    CLEAR_THRESHOLD = 0.80
    QA_BAND = 'cs_cdf'
        
    beforeStartDate = ee.Date.fromYMD(yearly, 1, 1)
    beforeEndDate = ee.Date.fromYMD(yearly, 11, 1)

    def maskData(img):
        return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD))

    # Write a function that computes NDVI for an image and adds it as a band
    def cal_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)

    s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterBounds(geometry).filterDate(beforeStartDate, beforeEndDate).sort("CLOUDY_PIXEL_PERCENTAGE")
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED').filterBounds(geometry).filterDate(beforeStartDate, beforeEndDate)
    s2collection = s2.linkCollection(csPlus, [QA_BAND]).map(maskData).map(cal_ndvi)
    
    # BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    # s2Before = s2collection.select(BANDS).median()

    Indices = ['NDVI']
    s2Before = s2collection.select(Indices).median()
       
    patch, bounds = get_patch(s2Before, lonlat, patch_size, 10)
    imagePatch = structured_to_unstructured(patch) 
    coords = np.array(bounds.getInfo().get("coordinates"))[0]
    writeOutput(imagePatch, img_path_before, patch_size, coords)
    print(f"s2 images: {i}", imagePatch.shape)

    mangroves = ee.Image("projects/tony-1122/assets/mangrove_th_project/mangrovesindo_2023")
    patch, bounds = get_patch(mangroves, lonlat, patch_size, 10)
    imagePatch = structured_to_unstructured(patch).squeeze()
    coords = np.array(bounds.getInfo().get("coordinates"))[0]
    # writeOutputSingle(imagePatch, img_path_label, patch_size, coords)
    # print(f"label: {i}", imagePatch.shape)

def process_data(i, location):
    try:
        getData(i, location)
    except Exception as e:
        print(f"Error processing index {i}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Process satellite data for a specified location and year.")
    parser.add_argument("location", type=str, help="Earth Engine FeatureCollection location path")
    parser.add_argument("year", type=int, help="Year for data processing")
    args = parser.parse_args()

    location = args.location  # Location provided as an argument
    global yearly  # Update the global yearly variable
    yearly = args.year

    # Get the total number of features in the collection
    mySamples = ee.FeatureCollection(location)
    total_samples = mySamples.size().getInfo()  # Get total number of samples

    print(f"Processing {total_samples} samples from the collection: {location} for the year {yearly}")
    
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(process_data, i, location) for i in range(total_samples)]
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Exception in thread: {e}")

if __name__ == '__main__':
    main()

