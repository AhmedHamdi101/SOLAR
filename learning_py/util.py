import itertools
import os
import pandas as pd
import numpy as np
import io
import re
import subprocess
from scipy.spatial.distance import jensenshannon
import statistics
from shapely.geometry import MultiPoint
import math
import pickle

def get_dataset_paths(hdfs_dir):
    dataset_paths = []
    try:
        cmd = ["hadoop", "fs", "-ls", hdfs_dir]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)

        lines = output.decode("utf-8").strip().split("\n")
        
        for line in lines[1:]:
            parts = line.split()
            if len(parts) > 0:
                full_path = parts[-1]
                dataset_paths.append(full_path)
    except subprocess.CalledProcessError as e:
        print(f"Error listing files in HDFS directory {hdfs_dir}: {e.output.decode('utf-8')}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

    return dataset_paths




def read_csv_from_hdfs(hdfs_path, header=0, names=None, dtype=object):
    try:
        process = subprocess.Popen(["hadoop", "fs", "-cat", hdfs_path],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data, err = process.communicate()  
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, "hadoop fs -cat", err)

        try:
            df = pd.read_csv(io.BytesIO(data), header=header, names=names, dtype=dtype)
        except UnicodeDecodeError:
            print("Encoding error, retrying with ISO-8859-1:", hdfs_path)
            df = pd.read_csv(io.BytesIO(data), encoding='ISO-8859-1', header=header, names=names, dtype=dtype)

        df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce')

        df.dropna(inplace=True)

        return df

    except subprocess.CalledProcessError as e:
        print(f"Error executing Hadoop command: {e.output.decode('utf-8')}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None





def bert_encoder(hdfs_dir, debug=True):
    poly_features = read_polygon_feature(hdfs_dir)

    minx, miny, maxx, maxy = poly_features['bounds']
    centroid_x = poly_features['centroid_x']
    centroid_y = poly_features['centroid_y']

    scaled_num_points = np.log1p(poly_features['num_points'])
    scaled_area       = np.log1p(poly_features['area'])

    scale_factor = 1e6
    scaled_centroid_x = centroid_x / scale_factor
    scaled_centroid_y = centroid_y / scale_factor
    scaled_minx       = minx       / scale_factor
    scaled_miny       = miny       / scale_factor
    scaled_maxx       = maxx       / scale_factor
    scaled_maxy       = maxy       / scale_factor

    compactness = poly_features['compactness']

    final_vector = np.array([
        scaled_num_points,
        scaled_area,
        scaled_centroid_x,
        scaled_centroid_y,
        scaled_minx,
        scaled_miny,
        scaled_maxx,
        scaled_maxy,
        compactness
    ], dtype=np.float32)

    #if debug:
    #    print(f"Dataset: {hdfs_dir}")
    #    print("Final scaled vector:", final_vector)

    return final_vector





def compute_jsd_distance(hist1, hist2):

    hist1_normalized = hist1 / np.sum(hist1)
    hist2_normalized = hist2 / np.sum(hist2)


    hist1_flat = hist1_normalized.flatten()
    hist2_flat = hist2_normalized.flatten()

    jsd = jensenshannon(hist1_flat, hist2_flat, base=2)

    return jsd


        

def compute_histogram(hdfs_dir, div):
    dir_path = f"stat/histogram/{div}/"
    filename = f"{dir_path}{hdfs_dir.replace('/', '@')}.pkl"
    
    if os.path.exists(filename):
        #print(f"Histogram already exists: {filename}")
        return

    os.makedirs(dir_path, mode=0o777, exist_ok=True)

    miny, maxy, minx, maxx = (-20040000.0, 20040000.0, -20040000.0, 20040000.0)

    x_bins = np.linspace(minx, maxx, div + 1)
    y_bins = np.linspace(miny, maxy, div + 1)

    dataset_csv = read_csv_from_hdfs(hdfs_dir, header=0, names=['x', 'y'], dtype={'x': float, 'y': float})
    histogram, _, _ = np.histogram2d(dataset_csv['x'], dataset_csv['y'], bins=[x_bins, y_bins])

    with open(filename, 'wb') as f:
        pickle.dump(histogram, f)




def compute_polygon_feature(hdfs_dir):
    dir_path = "stat/polygon_feature/"
    filename = f"{dir_path}{hdfs_dir.replace('/', '@')}.pkl"

    if os.path.exists(filename):
        print(f"Polygon feature already exists: {filename}")
        return

    os.makedirs(dir_path, mode=0o777, exist_ok=True)
    
    dataset_csv = read_csv_from_hdfs(hdfs_dir, header=0, names=['x', 'y'], dtype={'x': float, 'y': float})
    points = dataset_csv[['x', 'y']].values
    
    multipoint = MultiPoint(points)
    polygon = multipoint.convex_hull

    num_points = len(points)
    area = polygon.area
    centroid = polygon.centroid
    centroid_x, centroid_y = centroid.x, centroid.y
    bounds = polygon.bounds  
    perimeter = polygon.length

    compactness = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0

    features = {
        'num_points': num_points,
        'area': area,
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'bounds': bounds,
        'compactness': compactness
    }


    with open(filename, 'wb') as f:
        pickle.dump(features, f)

    print(f"Polygon feature computed and saved: {filename}")
    return features
    

def compute_all_polygon_features(datasets):
    print("\n--- calling compute_all_polygon_features in util.py")
    for dataset in datasets:
        compute_polygon_feature(dataset)


def read_polygon_feature(hdfs_dir):
    filename = f"stat/polygon_feature/{hdfs_dir.replace('/', '@')}.pkl"
    with open(filename, 'rb') as f:
        features = pickle.load(f)
    return features


def print_statistics(input_list):

    if not input_list:
        print("The list is empty.")
        return
    print("        ---- the following is all values from the list")
    print(input_list)

    data = np.array(input_list)
    data = data / 1000
 
    print(f"        ---- Max value: {max(data)}")
    print(f"        ---- Min value: {min(data)}")
    print(f"        ---- Average value: {sum(data) / len(data)}")
    print(f"        ---- Median value: {statistics.median(data)}")


    percentiles = [10, 25, 50, 75, 90]  
    for p in percentiles:
        print(f"        ---- {p}th Percentile: {np.percentile(data, p)}")
        

