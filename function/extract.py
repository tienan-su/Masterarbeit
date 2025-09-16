#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 23:32:27 2025

@author: tienansu
"""

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
from sklearn.preprocessing import MinMaxScaler


def extract_pressure_for_germany(nc_file, resolution=0.05):
    germany_bbox = {
        'min_lat': 47.3,
        'max_lat': 55.1,
        'min_lon': 5.9,
        'max_lon': 15.0
    }

    lats = np.arange(germany_bbox['min_lat'], germany_bbox['max_lat'], resolution)
    lons = np.arange(germany_bbox['min_lon'], germany_bbox['max_lon'], resolution)
    target_lats, target_lons = np.meshgrid(lats, lons)
    target_lats, target_lons = target_lats.flatten(), target_lons.flatten()

    with Dataset(nc_file, 'r') as nc:
        
        rlat = nc.variables['latitude'][:]   # shape (M,)
        rlon = nc.variables['longitude'][:]  # shape (N,)

        rlon_mesh, rlat_mesh = np.meshgrid(rlon, rlat)
        points = np.column_stack((rlat_mesh.ravel(), rlon_mesh.ravel()))

        tree = cKDTree(points)

        # Find nearest neighbor indices
        target_points = np.column_stack((target_lats, target_lons)) 
        distances, indices = tree.query(target_points)
        
        pressure = nc.variables['sp'][:]  # shape (T, M, N)
        extracted_pressure = pressure[:, indices // rlon.size, indices % rlon.size]

        if np.all(extracted_pressure == extracted_pressure[0, 0]):
            raise ValueError("Extracted pressure data is identical for all points. Check extraction process.")

        return extracted_pressure, lats, lons
    

def extract_wind_speed_for_germany(nc_file, resolution=0.05):
    germany_bbox = {
        'min_lat': 47.3,
        'max_lat': 55.1,
        'min_lon': 5.9,
        'max_lon': 15.0
    }

    lats = np.arange(germany_bbox['min_lat'], germany_bbox['max_lat'], resolution)
    lons = np.arange(germany_bbox['min_lon'], germany_bbox['max_lon'], resolution)
    target_lats, target_lons = np.meshgrid(lats, lons)
    target_lats, target_lons = target_lats.flatten(), target_lons.flatten()

    with Dataset(nc_file, 'r') as nc:
        rlat = nc.variables['latitude'][:]   
        rlon = nc.variables['longitude'][:] 

        rlon_mesh, rlat_mesh = np.meshgrid(rlon, rlat)
        points = np.column_stack((rlat_mesh.ravel(), rlon_mesh.ravel()))

        tree = cKDTree(points)

        target_points = np.column_stack((target_lats, target_lons)) 
        distances, indices = tree.query(target_points)

        wind_speed = nc.variables['ws'][:]
        extracted_wind_speed = wind_speed[:, indices // rlon.size, indices % rlon.size]

        if np.all(extracted_wind_speed == extracted_wind_speed[0, 0]):
            raise ValueError("Extracted wind speed data is identical for all points. Check extraction process.")

        return extracted_wind_speed, lats, lons

def extract_temperature_for_germany(nc_file, resolution=0.05):
    germany_bbox = {
        'min_lat': 47.3,
        'max_lat': 55.1,
        'min_lon': 5.9,
        'max_lon': 15.0
    }

    lats = np.arange(germany_bbox['min_lat'], germany_bbox['max_lat'], resolution)
    lons = np.arange(germany_bbox['min_lon'], germany_bbox['max_lon'], resolution)
    target_lats, target_lons = np.meshgrid(lats, lons)
    target_lats, target_lons = target_lats.flatten(), target_lons.flatten()

    with Dataset(nc_file, 'r') as nc:
        rlat = nc.variables['latitude'][:]   
        rlon = nc.variables['longitude'][:] 

        rlon_mesh, rlat_mesh = np.meshgrid(rlon, rlat)
        points = np.column_stack((rlat_mesh.ravel(), rlon_mesh.ravel()))

        tree = cKDTree(points)

        target_points = np.column_stack((target_lats, target_lons)) 
        distances, indices = tree.query(target_points)

        temperature = nc.variables['t2m'][:]
        extracted_temperature = temperature[:, indices // rlon.size, indices % rlon.size]

        if np.all(extracted_temperature == extracted_temperature[0, 0]):
            raise ValueError("Extracted temperature data is identical for all points. Check extraction process.")

        return extracted_temperature, lats, lons



def load_real_wind_csv(csv_file_path):
    df = pd.read_csv(csv_file_path, low_memory=False)
    df.dropna(axis=0, how='any', inplace=True)
    wind_data = df.to_numpy()
    x_y = wind_data[:, 1:3].astype(float)
    mask = np.any(wind_data[:, 5:] != 0, axis=1)
    filtered_x_y = x_y[mask]
    return filtered_x_y

def interpolate_wind_speed(wind_speeds, grid_lats, grid_lons, target_points):
    grid_lats = np.sort(grid_lats)
    grid_lons = np.sort(grid_lons)
    interpolated_wind_speeds = np.zeros((wind_speeds.shape[0], target_points.shape[0]))

    for t in range(wind_speeds.shape[0]):
        # Reshape wind_speeds[t] to a 2D array
        wind_speed_t = wind_speeds[t].reshape(len(grid_lats), len(grid_lons))
        
        interpolator = RegularGridInterpolator((grid_lats, grid_lons), wind_speed_t, method='linear', bounds_error=False, fill_value=np.nan)
        interpolated = interpolator(target_points)
        nan_mask = np.isnan(interpolated)
        if np.any(nan_mask):
            nearest_interpolator = RegularGridInterpolator((grid_lats, grid_lons), wind_speed_t, method='nearest', bounds_error=False, fill_value=np.nan)
            interpolated[nan_mask] = nearest_interpolator(target_points[nan_mask])
        interpolated_wind_speeds[t] = interpolated

    nan_count = np.isnan(interpolated_wind_speeds).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values remain after interpolation.")
    
    return interpolated_wind_speeds

def interpolate_pressure(data, grid_lats, grid_lons, target_points):
    grid_lats = np.sort(grid_lats)
    grid_lons = np.sort(grid_lons)
    interpolated_data = np.zeros((data.shape[0], target_points.shape[0]))

    for t in range(data.shape[0]):
        # Reshape data[t] to a 2D array
        data_t = data[t].reshape(len(grid_lats), len(grid_lons))
        
        interpolator = RegularGridInterpolator((grid_lats, grid_lons), data_t, method='linear', bounds_error=False, fill_value=np.nan)
        interpolated = interpolator(target_points)
        nan_mask = np.isnan(interpolated)
        if np.any(nan_mask):
            nearest_interpolator = RegularGridInterpolator((grid_lats, grid_lons), data_t, method='nearest', bounds_error=False, fill_value=np.nan)
            interpolated[nan_mask] = nearest_interpolator(target_points[nan_mask])
        interpolated_data[t] = interpolated

    nan_count = np.isnan(interpolated_data).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values remain after interpolation.")
    
    return interpolated_data

def interpolate_temperature(temp, grid_lats, grid_lons, target_points):
    grid_lats = np.sort(grid_lats)
    grid_lons = np.sort(grid_lons)
    interpolated_temperature = np.zeros((temp.shape[0], target_points.shape[0]))

    for t in range(temp.shape[0]):
        # Reshape data[t] to a 2D array
        temperature_t = temp[t].reshape(len(grid_lats), len(grid_lons))
        
        interpolator = RegularGridInterpolator((grid_lats, grid_lons), temperature_t, method='linear', bounds_error=False, fill_value=np.nan)
        interpolated = interpolator(target_points)
        nan_mask = np.isnan(interpolated)
        if np.any(nan_mask):
            nearest_interpolator = RegularGridInterpolator((grid_lats, grid_lons), temperature_t, method='nearest', bounds_error=False, fill_value=np.nan)
            interpolated[nan_mask] = nearest_interpolator(target_points[nan_mask])
        interpolated_temperature[t] = interpolated

    nan_count = np.isnan(interpolated_temperature).sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values remain after interpolation.")
    
    return interpolated_temperature

def scale_interpolated_data(data, feature_range=(-1, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    num_time_steps, num_points = data.shape
    scaled_data = np.zeros_like(data)

    for t in range(num_time_steps):
        scaled_data[t, :] = scaler.fit_transform(data[t, :].reshape(-1, 1)).flatten()
    
    return scaled_data

def combine_data(normalized_x_y, scaled_unix_time_array, filtered_wind_power):
    """
    Combine normalized_x_y, scaled_unix_time_array, and flattened filtered_wind_power into a single array.

    Parameters:
    - normalized_x_y (np.array): The normalized x and y coordinates (shape: (232, 2)).
    - scaled_unix_time_array (np.array): The scaled Unix timestamps (shape: (8783, 1)).
    - filtered_wind_power (np.array): The filtered wind power data (shape: (232, 8783)).

    Returns:
    - np.array: Combined array with shape (232*8783, 3).
    """
    num_rows = normalized_x_y.shape[0]  # Number of rows (232)
    num_columns = scaled_unix_time_array.shape[0]  # Number of columns (8783)

    # Repeat normalized_x_y for each timestamp in scaled_unix_time_array
    repeated_x_y = np.repeat(normalized_x_y, num_columns, axis=0)

    # Repeat scaled_unix_time_array for each row in normalized_x_y
    repeated_unix_time = np.tile(scaled_unix_time_array, (num_rows, 1))

    # Flatten the filtered_wind_power while preserving the sequence
    flattened_wind_power = filtered_wind_power.flatten()

    # Combine all three arrays into one
    combined_array = np.column_stack((repeated_x_y, repeated_unix_time.flatten(), flattened_wind_power))

    return combined_array

def repeat_target_points(scaled_target_points, num_time_steps):
    repeated_target_points = np.repeat(scaled_target_points, num_time_steps, axis=0)
    return repeated_target_points

def scale_target_points(target_points):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_target_points = scaler.fit_transform(target_points)
    return scaled_target_points
