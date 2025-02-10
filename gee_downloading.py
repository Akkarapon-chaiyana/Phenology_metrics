import ee
import rasterio
import datetime
import numpy as np
import copy
from typing import List, Tuple
from google.auth import default
from google.api_core import retry, exceptions
import requests
import os
import io
import google.auth
from numpy.lib.recfunctions import structured_to_unstructured
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation
#import cloud_removal
import hickle as hkl
from skimage.transform import resize
import math
from utils import calculate_and_save_best_images
from slope import calcSlope

# Initialize the Earth Engine module.
ee.Initialize(project = 'tony-1122')

def ee_init() -> None:
    """Authenticate and initialize Earth Engine with the default credentials."""
    credentials, project = default(
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/earthengine",
        ]
    )
    ee.Initialize(
        credentials.with_quota_project(None),
        project=project,
        opt_url="https://earthengine-highvolume.googleapis.com",
    )

@retry.Retry(deadline=10 * 60)  # seconds
def get_patch(image, roi, patch_size,scale,crs) -> np.ndarray:
    """Fetches a patch of pixels from Earth Engine.
    
    Args:
        image: Image to get the patch from.
        roi: Region of interest as an ee.Geometry.
        patch_size: Size in pixels of the surrounding square patch.
    
    Raises:
        requests.exceptions.RequestException
    
    Returns: 
        The requested patch of pixels as a NumPy array with shape (width, height, bands).
    """
    url = image.getDownloadURL(
        {
            "region": roi,
            "dimensions": [patch_size, patch_size],
            "crs":crs,
            #"scale":scale,
            #"crs_transform":[10,0,799980,0,-10,1600020],
            "format": "NPY",
        }
    )

    response = requests.get(url)
    if response.status_code == 429:
        raise exceptions.TooManyRequests(response.text)

    response.raise_for_status()
    return np.load(io.BytesIO(response.content), allow_pickle=True)

def to_int16(array: np.array) -> np.array:
    '''Converts a float32 array to uint16'''
    assert np.min(array) >= 0, np.min(array)
    assert np.max(array) <= 1, np.max(array)

    array = np.clip(array, 0, 1)
    array = np.trunc(array * 65535)
    assert np.min(array >= 0)
    assert np.max(array <= 65535)

    return array.astype(np.uint16)

def extract_dates(image_collection):
    # Extract dates from image collection
    dates = image_collection.aggregate_array('system:time_start').getInfo()
    dates = [datetime.datetime.utcfromtimestamp(ms / 1000).strftime('%Y-%m-%d') for ms in dates]
    filenames = image_collection.aggregate_array('system:index').getInfo()
    return dates, filenames

def make_bbox(initial_bbx: list, expansion: int = 10) -> list:
    """Makes a (min_x, min_y, max_x, max_y) bounding box that
       is expanded by a given number of pixels.

       Parameters:
            initial_bbx (list): [min_x, min_y, max_x, max_y]
            expansion (int): Number of pixels to expand by

       Returns:
            bbx (list): Expanded [min_x, min_y, max_x, max_y]
    """
    multiplier = 1 / 360  # Sentinel-2 pixel size in decimal degrees (adjust if necessary)
    bbx = copy.deepcopy(initial_bbx)
    bbx[0] -= expansion * multiplier
    bbx[1] -= expansion * multiplier
    bbx[2] += expansion * multiplier
    bbx[3] += expansion * multiplier
    return bbx

def _check_for_alt_img(probs, dates, date):
    # Checks to see if there is an image within win days that has
    # Less local cloud cover. If so, remove the higher CC image.
    # This is done to avoid having to remove (interpolate)
    # All of an image for a given tile, as this can cause artifacts
    begin = [-60, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 341]
    end = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 341, 410]

    begins = (end - date)
    begins[begins < 0] = 999
    month_start = begin[np.argmin(begins)]
    month_end = end[np.argmin(begins)]
    lower = month_start
    upper = month_end
    upper = np.maximum(date + 28, upper)
    lower = np.minimum(date - 28, lower)

    candidate_idx = np.argwhere(
        np.logical_and(np.logical_and(dates >= lower, dates <= upper),
                       dates != date))
    candidate_probs = probs[candidate_idx]
    if len(candidate_probs) == 0:
        return False

    idx = np.argwhere(dates == date).flatten()
    begin_prob = probs[idx]
    if np.min(candidate_probs) < (begin_prob - 0.20):
        return True
    else:
        return False


def downsample_to_40m(bands_20m, nSteps):
    """
    Downsample 20m resolution bands to 40m.
    
    Parameters:
    bands_20m (numpy.ndarray): Input array of shape (steps, height, width, n_bands)
    nSteps (int): Number of time steps
    
    Returns:
    numpy.ndarray: Downsampled array of shape (steps, height//2, width//2, n_bands)
    """
    steps, height, width, n_bands = bands_20m.shape
    bands_40m = np.zeros((steps, height // 2, width // 2, n_bands), dtype=bands_20m.dtype)
    
    for step in range(steps):
        for band in range(n_bands):
            bands_40m[step, :, :, band] = resize(bands_20m[step, :, :, band], (height // 2, width // 2), order=0, preserve_range=True)
    
    return bands_40m

def adjust_dimensions(s220img, s240img):
    """
    Adjust dimensions of downsampled 40m bands to match 20m bands.
    
    Parameters:
    s220img (numpy.ndarray): Original 20m bands of shape (steps, height, width, n_bands)
    s240img (numpy.ndarray): Downsampled 40m bands of shape (steps, height//2, width//2, n_bands)
    
    Returns:
    numpy.ndarray: Adjusted 40m bands of shape (steps, height, width, n_bands)
    """
    img_40 = s240img.repeat(2, axis=1).repeat(2, axis=2)

    if (s220img.shape[1] > img_40.shape[1]) or (s220img.shape[2] > img_40.shape[2]):
        img_40 = resize(img_40, (s220img.shape[0], s220img.shape[1], s220img.shape[2], img_40.shape[-1]), order=0)

    if img_40.shape[1] > s220img.shape[1]:
        to_remove = (img_40.shape[1] - s220img.shape[1])
        if to_remove == 2:
            img_40 = img_40[:, 1:-1, ...]
        if to_remove == 1:
            img_40 = img_40.repeat(2, axis=1).repeat(2, axis=2)
            img_40 = img_40[:, 1:-1, ...]
            img_40 = np.reshape(img_40, (img_40.shape[0], img_40.shape[1] // 2, 2, img_40.shape[2] // 2, 2, img_40.shape[-1]))
            img_40 = np.mean(img_40, axis=(2, 4))

    if img_40.shape[2] > s220img.shape[2]:
        to_remove = (img_40.shape[2] - s220img.shape[2])
        if to_remove == 2:
            img_40 = img_40[:, :, 1:-1, ...]
        if to_remove == 1:
            img_40 = img_40.repeat(2, axis=1).repeat(2, axis=2)
            img_40 = img_40[:, :, 1:-1, ...]
            img_40 = np.reshape(img_40, (img_40.shape[0], img_40.shape[1] // 2, 2, img_40.shape[2] // 2, 2, img_40.shape[-1]))
            img_40 = np.mean(img_40, axis=(2, 4))

    return img_40
    
    
def write_to_tif(crs,arr,point):
    fileName = "/home/ate-laptop/temp/s210c.tif"

    west, east = point[0], point[2]
    north, south = point[3], point[1]

    transform = rasterio.transform.from_bounds(west=west,
                                               south=south,
                                               east=east,
                                               north=north,
                                               width=arr.shape[1],
                                               height=arr.shape[0])
    arr = to_int16(arr)
    
    new_dataset = rasterio.open(fileName,
                                'w',
                                driver='GTiff',
                                height=arr.shape[0],
                                width=arr.shape[1],
                                count=arr.shape[-1],
                                dtype="uint16",
                                compress='zstd',
                                crs=crs,
                                transform=transform)
    arr = np.rollaxis(arr, 2)
    arr = np.flip(arr, axis = 0)
    new_dataset.write(arr)
    new_dataset.close()


def download_sentinel_2_new(fnames,cloud_bbx, dates, year,crs, maxclouds=0.4):
    """ Downloads the L2A sentinel layer with 10 and 20 meter bands

        Parameters:
         bbox (list): output of calc_bbox
         clean_steps (list): list of steps to filter download request
         epsg (float): EPSG associated with bbox
         time (tuple): YY-MM-DD - YY-MM-DD bounds for downloading

        Returns:
         img (arr):
         img_request (obj):
    """
    start_date, end_date = dates
    QA_BAND = 'cs_cdf'

    roi = ee.Geometry.Point([cloud_bbx[0], cloud_bbx[1]]).transform(crs,0.001).buffer(320*10).bounds(0.0)
    roi = roi.transform(crs,0.001)

        
    # Load Sentinel-2 image collection
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filterBounds(roi) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', maxclouds * 100))\
        .filter(ee.Filter.inList("system:index",ee.List(fnames.tolist())))
        
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    s2 = s2.map(lambda img: img.addBands(csPlus.filter(ee.Filter.equals('system:index', img.get('system:index'))).first()))
    
    s210 = s2.select(["B2", "B3", "B4", "B8"])
    s220 = s2.select(["B5", "B6", "B7", "B8A", "B11", "B12"])
    s2cloud = s2.select(['cs_cdf'])
    
    nSteps  = int(s2.size().getInfo())
    
    patchsize10 = 640
    
    # need to split with more than 10 we hit limit
    
    # Define the bands for each request
    bands_part1 = ["B2", "B3"]
    bands_part2 = ["B4", "B8"]

    # Select the bands for each part
    s210_part1 = s2.select(bands_part1)
    s210_part2 = s2.select(bands_part2)


    # Get the patches for each part
    patch_part1 = get_patch(s210_part1.toBands(), roi, patchsize10,10,crs)
    patch_part2 = get_patch(s210_part2.toBands(), roi, patchsize10,10,crs)

    # Convert the structured arrays to unstructured arrays
    s210img_part1 = structured_to_unstructured(patch_part1)
    s210img_part2 = structured_to_unstructured(patch_part2)
    
    num_bands = 4
    new_shape = (nSteps, patchsize10, patchsize10, num_bands)
    result = np.empty(new_shape)


    # Number of bands in each part
    num_bands_part1 = 2
    num_bands_part2 = 2
    num_bands_total = num_bands_part1 + num_bands_part2
    
    # Loop through each time step and assign the corresponding bands
    for i in range(nSteps):
        for j in range(num_bands_part1):
            result[i, :, :, j] = s210img_part1[:, :, i * num_bands_part1 + j]
        for j in range(num_bands_part2):
            result[i, :, :, num_bands_part1 + j] = s210img_part2[:, :, i * num_bands_part2 + j]
    
    s210img = result / 10000

    
    patchsize20 = 320
    patch = get_patch(s220.toBands(), roi, patchsize20,20,crs)    
    s220img = structured_to_unstructured(patch)


    num_bands = 6
    new_shape = (nSteps, patchsize20, patchsize20, num_bands)
    result = np.empty(new_shape)

    # Loop through each time step and assign the corresponding bands
    for i in range(nSteps):
        for j in range(num_bands):
            result[i, :, :, j] = s220img[:, :, i * num_bands + j]
    s220img = result / 10000



    patch = get_patch(s2cloud.toBands(), roi, patchsize20,20,crs)   
    s2cloud = structured_to_unstructured(patch)

    s2cloud = s2cloud.reshape((patchsize20, patchsize20, 1, nSteps))
    s2cloud = s2cloud.transpose(3, 0, 1, 2)
    s2cloud = np.where(s2cloud > maxclouds, 0, 1)
    
    # Downsample to 40m
    s240img = downsample_to_40m(s220img, nSteps)

    # Adjust dimensions to match 20m bands
    s240img = adjust_dimensions(s220img, s240img)
	
    s220img = np.concatenate([s220img, s240img], axis=-1)

    # Convert 10m bands to np.float32, ensure correct dimensions
    if not isinstance(s210img.flat[0], np.floating):
        assert np.max(s210img) > 1
        s210img = np.float32(s210img) / 65535.
        assert np.max(s210img) <= 1
        assert s210img.dtype == np.float32


    # Convert 10m bands to np.float32, ensure correct dimensions
    if not isinstance(s220img.flat[0], np.floating):
        assert np.max(s220img) > 1
        s220img = np.float32(s220img) / 65535.
        assert np.max(s220img) <= 1
        assert s220img.dtype == np.float32

    
    s210img = np.clip(s210img, 0, 1)
    s220img = np.clip(s220img, 0, 1)

    return s210img, s220img, s2cloud, dates
    


def getCRS(cloud_bbx):
	
    roi = ee.Geometry.Point([cloud_bbx[0], cloud_bbx[1]]).buffer(320*10).bounds(0.0)
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(roi)
    img = ee.Image(s2.first())
    crs = img.select("B1").projection().crs().getInfo()
    
    return crs
	
def testGeom():
	# Load Sentinel-2 image collection
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filterBounds(roi) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))\
        

    s210 = s2.select(["B2", "B3", "B4"])
    

    patchsize10 = 640
    transform = ee.Image(s210.first()).select("B2").projection().transform().getInfo()
    print(transform)
    print(ee.Image(s210.first()).get("system:index").getInfo())
    
    # Get the patches for each part
    transform = [10,0,600000,0,-10,1500000]
    patch_part1 = get_patch(ee.Image(s210.first()), roi, patchsize10,10,crs,transform)
    
    
    s210img = structured_to_unstructured(patch_part1) / 10000
    
    

    print("result",s210img.shape)
    write_to_tif(crs,s210img,bbx)

def get_bbx(cloud_bbx, crs):
    """
    Calculates the bounding box coordinates transformed to the specified CRS.

    Parameters:
    cloud_bbx (list): Bounding box coordinates [minLng, minLat]
    crs (str): Coordinate reference system

    Returns:
    bbx (list): Bounding box coordinates [min_x, min_y, max_x, max_y]
    """
    
    # Define the region of interest (ROI) and transform its coordinates
    roi = ee.Geometry.Point([cloud_bbx[0], cloud_bbx[1]]).transform(crs, 0.001)
    roi = roi.buffer(3200).bounds().transform(crs, 0.001)
    
    # Get the bounding box coordinates
    roi_info = roi.getInfo()
    
    # Extract the coordinates and calculate the bounding box
    coords = np.array(roi_info['coordinates'][0])
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    
    # Format the coordinates into the desired bounding box format
    bbx = [min_x, min_y, max_x, max_y]
    
    return bbx




def identify_clouds_big_bbx(cloud_bbx, dates, year,crs, maxclouds=0.5):
    """
    Downloads and calculates cloud cover and shadow

    Parameters:
     cloud_bbx (list): Bounding box coordinates [minLng, minLat, maxLng, maxLat]
     dates (tuple): Start and end date for filtering images ('YYYY-MM-DD', 'YYYY-MM-DD')
     year (int): Year for date calculations
     maxclouds (float): Maximum cloud cover percentage

    Returns:
     cloudImage (np.ndarray): Cloud images
     cloud_percent (np.ndarray): Cloud percentage
     cloud_dates (np.ndarray): Dates of the cloud images
     local_clouds (np.ndarray): Local cloud data
    """
    start_date, end_date = dates
    QA_BAND = 'cs_cdf'
    
    #roi = ee.Geometry.Point([cloud_bbx[0], cloud_bbx[1]])
    #print("before", roi.buffer(640*10).bounds(0.0).getInfo()) 
    roi = ee.Geometry.Point([cloud_bbx[0], cloud_bbx[1]]).transform(crs,0.001)
    roi = roi.buffer(320*10).bounds().transform(crs,0.001)
    #roi = roi

    # Get the bounding box coordinates
    roi_info = roi.getInfo()

    # Extract the coordinates
    coords = np.array(roi_info['coordinates'][0])

    # Calculate the bounding box
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)

    # Format the coordinates into the desired bounding box format
    bbx = [min_x, min_y, max_x, max_y]
  

    # Load Sentinel-2 image collection
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filterBounds(roi) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', maxclouds * 100))

    # Extract dates of the images
    cloud_dates, filenames = extract_dates(s2)

    base_date = f'{year - 1}-12-31'  # Base date to calculate day of year
    cloud_dates = convert_to_day_of_year(cloud_dates, base_date)
    
    # Apply cloud mask
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    #s2 = s2.map(lambda img: img.addBands(csPlus.filter(ee.Filter.equals('system:index', img.get('system:index'))).first()))
    s2 = s2.linkCollection(csPlus, [QA_BAND])

    clouds = ee.Image(s2.select([QA_BAND]).toBands())
    patch = get_patch(clouds, roi, 160,160,crs)

    cloudImage = structured_to_unstructured(patch)

    # Filter out arrays with -inf or nan values
    valid_indices = ~np.isnan(cloudImage).any(axis=(0, 1)) & ~np.isinf(cloudImage).any(axis=(0, 1))
    

    cloud_dates = np.array(cloud_dates)[valid_indices]
    filenames = np.array(filenames)[valid_indices]

    cloudImage = cloudImage[:, :, valid_indices]
    cloudImage = np.transpose(cloudImage, (2, 0, 1))

    mid_idx = cloudImage.shape[1] // 2
    mid_idx_y = cloudImage.shape[2] // 2

    # Apply the clear threshold
    cloudImage = 1-cloudImage
    cloudImage[cloudImage < 0.4] = 0.01
    #cloudImage[cloudImage >= 0.6] = 0
    
    cloud_percent = np.nanmean(cloudImage, axis=(1, 2))

    local_clouds = np.copy(cloudImage[:, mid_idx - 15:mid_idx + 15, mid_idx_y - 15:mid_idx_y + 15])
    for i in range(cloudImage.shape[0]):
        clouds = local_clouds[i]
        local_clouds[i] = binary_dilation(clouds)

    local_clouds = np.nanmean(local_clouds, axis=(1, 2))

    cloudImage[np.isnan(cloudImage)] = 1

    cloud_steps = np.argwhere(cloud_percent > 0.5)
    cloudImage = np.delete(cloudImage, cloud_steps, 0)
    cloud_percent = np.delete(cloud_percent, cloud_steps)
    cloud_dates = np.delete(cloud_dates, cloud_steps)
    filenames = np.delete(filenames, cloud_steps)
    local_clouds = np.delete(local_clouds, cloud_steps)


    cloud_percent[cloud_percent > 0.4] = ((0.25 * cloud_percent[cloud_percent > 0.4] +
                                           0.75 * local_clouds[cloud_percent > 0.4]))

    cloud_steps = np.argwhere(cloud_percent > maxclouds)
    cloudImage = np.delete(cloudImage, cloud_steps, 0)
    cloud_percent = np.delete(cloud_percent, cloud_steps)
    cloud_dates = np.delete(cloud_dates, cloud_steps)
    filenames = np.delete(filenames, cloud_steps)
    local_clouds = np.delete(local_clouds, cloud_steps)

    to_remove = []
    for i, x, l in zip(cloud_dates, local_clouds, range(len(cloud_dates))):
        if x < 0.60 and _check_for_alt_img(local_clouds, cloud_dates, i):
            to_remove.append(l)

    if to_remove:
        cloud_dates = np.delete(cloud_dates, to_remove)
        filenames = np.delete(filenames, to_remove)
        cloudImage = np.delete(cloudImage, to_remove, 0)
        cloud_percent = np.delete(cloud_percent, to_remove)
        local_clouds = np.delete(local_clouds, to_remove)

    cloudImage = cloudImage.astype(np.float32)
    assert np.max(cloudImage) <= 1, np.max(cloudImage)
    assert cloudImage.dtype == np.float32

    return cloudImage, cloud_percent, np.array(cloud_dates), local_clouds, filenames, bbx

def generate_date_range(year):
    start_date = f'{year - 1}-11-15'
    end_date = f'{year + 1}-02-15'
    return start_date, end_date

def convert_to_day_of_year(dates, base_date):
    base_date = datetime.datetime.strptime(base_date, '%Y-%m-%d')
    day_of_year = [(datetime.datetime.strptime(date, '%Y-%m-%d') - base_date).days for date in dates]
    return np.array(day_of_year)


def toNatural(img):
  """Function to convert from dB to natural"""
  return ee.Image(10.0).pow(img.select(0).divide(10.0));

def toDB(img):
  """ Function to convert from natural to dB """
  return ee.Image(img).log10().multiply(10.0);

def addRatio(img):
  geom = img.geometry()
  vv = toNatural(img.select(['VV'])).rename(['VV']);
  vh = toNatural(img.select(['VH'])).rename(['VH']);
  ratio = vh.divide(vv).rename(['ratio']);
  return ee.Image(ee.Image.cat(vh,vv).copyProperties(img,['system:time_start'])).clip(geom).copyProperties(img);


def terrainCorrection(image):
    date = ee.Date(image.get('system:time_start'))
    imgGeom = image.geometry()
    srtm = ee.Image('USGS/SRTMGL1_003').clip(imgGeom)    # 30m srtm
    sigma0Pow = ee.Image.constant(10).pow(image.divide(10.0))

    #Article ( numbers relate to chapters)
    #2.1.1 Radar geometry
    theta_i = image.select('angle')
    phi_i = ee.Terrain.aspect(theta_i).reduceRegion(ee.Reducer.mean(), theta_i.get('system:footprint'), 1000).get('aspect')

    #2.1.2 Terrain geometry
    alpha_s = ee.Terrain.slope(srtm).select('slope')
    phi_s = ee.Terrain.aspect(srtm).select('aspect')

    # 2.1.3 Model geometry
    # reduce to 3 angle
    phi_r = ee.Image.constant(phi_i).subtract(phi_s)

    #convert all to radians
    phi_rRad = phi_r.multiply(math.pi / 180)
    alpha_sRad = alpha_s.multiply(math.pi / 180)
    theta_iRad = theta_i.multiply(math.pi / 180)
    ninetyRad = ee.Image.constant(90).multiply(math.pi / 180)

    # slope steepness in range (eq. 2)
    alpha_r = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()

    # slope steepness in azimuth (eq 3)
    alpha_az = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()

    # local incidence angle (eq. 4)
    theta_lia = (alpha_az.cos().multiply((theta_iRad.subtract(alpha_r)).cos())).acos()
    theta_liaDeg = theta_lia.multiply(180 / math.pi)

    # 2.2
    # Gamma_nought_flat
    gamma0 = sigma0Pow.divide(theta_iRad.cos())
    gamma0dB = ee.Image.constant(10).multiply(gamma0.log10())
    ratio_1 = gamma0dB.select('VV').subtract(gamma0dB.select('VH'))

    # Volumetric Model
    nominator = (ninetyRad.subtract(theta_iRad).add(alpha_r)).tan()
    denominator = (ninetyRad.subtract(theta_iRad)).tan()
    volModel = (nominator.divide(denominator)).abs()

    # apply model
    gamma0_Volume = gamma0.divide(volModel)
    gamma0_VolumeDB = ee.Image.constant(10).multiply(gamma0_Volume.log10())

    # we add a layover/shadow maskto the original implmentation
    # layover, where slope > radar viewing angle
    alpha_rDeg = alpha_r.multiply(180 / math.pi)
    layover = alpha_rDeg.lt(theta_i);

    # shadow where LIA > 90
    shadow = theta_liaDeg.lt(85)

    # calculate the ratio for RGB vis
    ratio = gamma0_VolumeDB.select('VV').subtract(gamma0_VolumeDB.select('VH'))

    output = gamma0_VolumeDB.addBands(ratio).addBands(alpha_r).addBands(phi_s).addBands(theta_iRad)\
			    .addBands(layover).addBands(shadow).addBands(gamma0dB).addBands(ratio_1)

    return output.select(['VH', 'VV'], ['VH', 'VV']).set("system:time_start",date).clip(imgGeom ).copyProperties(image)


def process_sentinel_1_tile(sentinel1: np.ndarray,
                            dates: np.ndarray) -> np.ndarray:
    """Converts a (?, X, Y, 2) Sentinel 1 array to a regular monthly grid

        Parameters:
         sentinel1 (np.array):
         dates (np.array):

        Returns:
         s1 (np.array)
    """

    s1, _ = calculate_and_save_best_images(sentinel1, dates)
    monthly = np.zeros((12, sentinel1.shape[1], sentinel1.shape[2], 2), dtype = np.float32)
    index = 0
    for start, end in zip(
            range(0, 24 + 2, 24 // 12),  #0, 72, 6
            range(24 // 12, 24 + 2, 24 // 12)):  # 6, 72, 6
        monthly[index] = np.median(s1[start:end], axis=0)
        index += 1

    return monthly

# Function to get the Julian day for the 15th of each month in a given year
def get_mid_month_julian_days(year):
    mid_month_days = []
    for month in range(1, 13):
        date = datetime.datetime(year, month, 15)
        julian_day = date.timetuple().tm_yday
        mid_month_days.append(julian_day)
    return mid_month_days

def download_sentinel_1_composite(cloud_bbx, year,crs):

    # Define the region of interest
    #initial_bbx = cloud_bbx #[cloud_bbx[0], cloud_bbx[1], cloud_bbx[0], cloud_bbx[1]]
    #cloud_bbx_expanded = make_bbox(initial_bbx, expansion=300 / 30)


    roi = ee.Geometry.Point([cloud_bbx[0], cloud_bbx[1]]).transform(crs,0.001)
    roi = roi.buffer(320*10).bounds().transform(crs,0.001)

    months = ee.List.sequence(1,12,3)
	
    start = ee.Date.fromYMD(year,1,1)
    end	 = ee.Date.fromYMD(year,12,31)
	
    s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(roi)\
												.filterDate(start,end)\
												.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
												.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
												.filter(ee.Filter.eq('instrumentMode', 'IW'))\
												.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
												
    s1 = s1.map(terrainCorrection).map(addRatio).select(["VV","VH"])
    
    # Function to create a median composite for a given month
    def get_monthly_median(month):
        start_month = ee.Date.fromYMD(year, month, 1)
        end_month = start_month.advance(3, 'month')
        monthly_composite = s1.filterDate(start_month, end_month).median()
        return monthly_composite.set('month', month)

    # Create monthly median composites
    monthly_composites = ee.ImageCollection(months.map(lambda m: get_monthly_median(ee.Number(m))).flatten())   
    
    patchsize = 160
    patch = get_patch(monthly_composites.toBands(), roi, patchsize,40,crs)
    s1img = structured_to_unstructured(patch)
    
    print("myshape",s1img.shape)
  

    num_bands = 2
    nSteps = 4
    new_shape = (nSteps, patchsize, patchsize, num_bands)
    result = np.empty(new_shape)

    # Loop through each time step and assign the corresponding bands
    for i in range(nSteps):
        for j in range(num_bands):
            result[i, :, :, j] = s1img[:, :, i * num_bands + j]
    
    s1img = result.clip(0,1) 
    print("myshape",s1img.shape)
    #s1img  = s1img.repeat(2, axis=1).repeat(4, axis=2)
    s1img  = s1img.repeat(12 // s1img.shape[0], axis=0)
    s1img  =  s1img.repeat(4, axis=1).repeat(4, axis=2)
    print("new shape",s1img.shape)
    #exit()
   
    
    dates = get_mid_month_julian_days(year)

    s1img = process_sentinel_1_tile(s1img, dates)

    return s1img, dates


def download_dem(cloud_bbx,crs):
    """ Downloads the DEM layer from Sentinel hub

        Parameters:
         bbox (list): output of calc_bbox
         epsg (float): EPSG associated with bbox

        Returns:
         dem_image (arr):
    """
    # Define the region of interest
    initial_bbx = [cloud_bbx[0], cloud_bbx[1], cloud_bbx[0], cloud_bbx[1]]
    cloud_bbx_expanded = make_bbox(initial_bbx, expansion=300 / 30)

    roi = ee.Geometry.Point([cloud_bbx[0], cloud_bbx[1]]).transform(crs,0.001)
    roi = roi.buffer(320*10).bounds().transform(crs,0.001)

    
    dem = ee.Image("CGIAR/SRTM90_V4")
	
    patchsize = 214
    patch = get_patch(dem, roi, patchsize,30,crs) 
    dem_image = structured_to_unstructured(patch).squeeze()


    # Convert the uint16 data to float32
    dem_image = dem_image - 12000
    dem_image = dem_image.astype(np.float32)
    width = dem_image.shape[0]
    height = dem_image.shape[1]

    # Apply median filter, calculate slope
    #dem_image = median_filter(dem_image, size=5)
    dem_image = calcSlope(dem_image.reshape((1, width, height)),
                          np.full((width, height), 10),
                          np.full((width, height), 10),
                          zScale=1,
                          minSlope=0.02)
    dem_image = dem_image.reshape((width, height, 1))

    dem_image = dem_image[1:width - 1, 1:height - 1, :]
    dem_image = dem_image.squeeze()

    dem_image = dem_image.repeat(3, axis=0).repeat(3, axis=1)

    return dem_image

