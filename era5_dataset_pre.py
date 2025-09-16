#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 17:48:19 2025

@author: tienansu
"""

import xarray as xr
import pandas as pd
import numpy as np

era5_2020 = xr.open_dataset('/Users/tienansu/Documents/UFZ/era5_nc/Germany_2020.nc')
# print(era5_2020)

# Calculate wind speed
ws = np.sqrt(era5_2020['u10']**2 + era5_2020['v10']**2)
era5_2020['ws'] = xr.DataArray(ws)
# era5_2020.to_netcdf('/Users/tienansu/Documents/UFZ/ERA5/Germany_2020_with_ws.nc')

"""
csv = pd.read_csv('/Users/tienansu/Documents/UFZ/Results_2020_REMix_ReSTEP_hourly_REF.csv')
print(csv)

# Select and rename
csv_01 = csv[['POINT_Y', 'POINT_X', 'Capacity in kW']]
csv_01 = csv_01.rename(columns={
    'POINT_Y': 'latitude',
    'POINT_X': 'longitude',
    'Capacity in kW': 'capacity'
})

# Get the average of wind speed and surface pressure
ws_data = era5_2020['ws'].mean(dim="valid_time")  # or .isel(valid_time=0) for first time slice
sp_data = era5_2020['sp'].mean(dim="valid_time")

# Create empty lists to store results
ws_list = []
sp_list = []

# Loop over each row to interpolate ws and sp at each lat/lon
for _, row in csv_01.iterrows():
    lat = row['latitude']
    lon = row['longitude']

    # Interpolation
    ws_val = ws_data.interp(latitude=lat, longitude=lon).values.item()
    sp_val = sp_data.interp(latitude=lat, longitude=lon).values.item()

    ws_list.append(ws_val)
    sp_list.append(sp_val)

# Add the values to the DataFrame
csv_01['avg_ws'] = ws_list
csv_01['avg_sp'] = [sp / 1000 for sp in sp_list]


print(csv_01)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# %matplotlib inline

# Create the figure and axis using PlateCarree projection
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add geographic features
ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='lightblue')
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray')
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.LAKES.with_scale('50m'), alpha=0.5)

# Add gridlines with labels
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Plot average wind speed as colored dots
scatter = ax.scatter(
    csv_01['longitude'], csv_01['latitude'],
    c=csv_01['avg_ws'], cmap='plasma',
    s=30, edgecolors='black',
    transform=ccrs.PlateCarree()
)

# Add colorbar
cbar = plt.colorbar(scatter, orientation='vertical', shrink=0.6, pad=0.05)
cbar.set_label('Mean Wind Speed (m/s)')

# Add title and range
min_ws = csv_01['avg_ws'].min()
max_ws = csv_01['avg_ws'].max()
mean_ws = csv_01['avg_ws'].mean()

plt.title(f"Wind Speed Map\nMean Wind Speed Range: {min_ws:.2f} - {max_ws:.2f} m/s\nOverall Mean Wind Speed: {mean_ws:.2f} m/s", 
          fontsize=14, weight='bold')

plt.tight_layout()
plt.show()


# Plot surface pressure
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add geographic features
ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='lightblue')
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray')
ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':')
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.LAKES.with_scale('50m'), alpha=0.5)

# Add gridlines with labels
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Plot average pressure as colored dots
scatter = ax.scatter(
    csv_01['longitude'], csv_01['latitude'],
    c=csv_01['avg_sp'], cmap='plasma',
    s=30, edgecolors='black',
    transform=ccrs.PlateCarree()
)

# Add colorbar
cbar = plt.colorbar(scatter, orientation='vertical', shrink=0.6, pad=0.05)
cbar.set_label('Mean Pressure (kPa)')

# Add title and range
min_ws = csv_01['avg_sp'].min()
max_ws = csv_01['avg_sp'].max()
mean_ws = csv_01['avg_sp'].mean()

plt.title(f"Pressure Map\nMean Pressure Range: {min_ws:.2f} - {max_ws:.2f} kPa\nOverall Mean Pressure: {mean_ws:.2f} kPa", 
          fontsize=14, weight='bold')

plt.tight_layout()
plt.show()
"""


import os
os.chdir("/Users/tienansu/Documents/UFZ/era5_py")
print("Current working directory:", os.getcwd())

import extract
from extract import (
    extract_pressure_for_germany,
    extract_wind_speed_for_germany,
    extract_temperature_for_germany,
    load_real_wind_csv,
    interpolate_wind_speed,
    # loading_wind,
    interpolate_pressure,
    interpolate_temperature,
    scale_interpolated_data,
    combine_data,
    repeat_target_points,
    scale_target_points
    )

nc_file_path = '/Users/tienansu/Documents/UFZ/era5_nc/Germany_2020_with_ws.nc'
csv_file_path = '/Users/tienansu/Documents/UFZ/Results_2020_REMix_ReSTEP_hourly_REF.csv'

from netCDF4 import Dataset
nc = Dataset(nc_file_path, 'r')

print(nc.variables.keys())

# Pressure
pressure_data, grid_lats, grid_lons = extract_pressure_for_germany(nc_file_path)

print("Shape of extracted pressure:", pressure_data.shape)
print("First 5 values:\n", pressure_data[:, :5])  
print("First time step pressure map:\n", pressure_data[0])

# Load real wind CSV
target_points = load_real_wind_csv(csv_file_path)

# Interpolate pressure data
interpolated_pressure = interpolate_pressure(pressure_data, grid_lats, grid_lons, target_points)
interpolated_pressure = interpolated_pressure / 1000  # Now in kPa

print(f"Shape of interpolated pressure data: {interpolated_pressure.shape}")
print(f"Sample of interpolated pressure data (first 5 time steps, first 5 locations):")
print(interpolated_pressure[:5, :5])

print(f"Number of NaN values: {np.isnan(interpolated_pressure).sum()}")
print(f"Number of infinite values: {np.isinf(interpolated_pressure).sum()}")

# Analyze pressure data
location_variation = np.any(np.diff(interpolated_pressure, axis=1) != 0, axis=0)
time_variation = np.any(np.diff(interpolated_pressure, axis=0) != 0, axis=1)


print(f"Number of locations with variation: {location_variation.sum()} out of {interpolated_pressure.shape[1]}")
print(f"Number of time steps with variation: {time_variation.sum()} out of {interpolated_pressure.shape[0]}")

location_stats = {
    'min': np.min(interpolated_pressure, axis=0),
    'max': np.max(interpolated_pressure, axis=0),
    'mean': np.mean(interpolated_pressure, axis=0),
    'std': np.std(interpolated_pressure, axis=0)
}

print("\nPressure statistics across locations:")
for stat, values in location_stats.items():
    print(f"{stat.capitalize()}: min = {values.min():.2f}, max = {values.max():.2f}")

min_loc = np.unravel_index(np.argmin(location_stats['min']), location_stats['min'].shape)
max_loc = np.unravel_index(np.argmax(location_stats['max']), location_stats['max'].shape)

print(f"\nLocation with lowest minimum pressure: Lat {target_points[min_loc[0], 1]:.2f}, Lon {target_points[min_loc[0], 0]:.2f}")
print(f"Location with highest maximum pressure: Lat {target_points[max_loc[0], 1]:.2f}, Lon {target_points[max_loc[0], 0]:.2f}")

min_pressure_index = np.unravel_index(np.argmin(interpolated_pressure, axis=None), interpolated_pressure.shape)
max_pressure_index = np.unravel_index(np.argmax(interpolated_pressure, axis=None), interpolated_pressure.shape)

print(f"\nShape of interpolated pressure data: {interpolated_pressure.shape}")

print(f"\nMinimum pressure of {interpolated_pressure[min_pressure_index]:.2f} KPa occurred at:")
print(f"Time step: {min_pressure_index[0]}, Location: Lat {target_points[min_pressure_index[1], 1]:.2f}, Lon {target_points[min_pressure_index[1], 0]:.2f}")

print(f"\nMaximum pressure of {interpolated_pressure[max_pressure_index]:.2f} KPa occurred at:")
print(f"Time step: {max_pressure_index[0]}, Location: Lat {target_points[max_pressure_index[1], 1]:.2f}, Lon {target_points[max_pressure_index[1], 0]:.2f}")

avg_pressure = np.mean(interpolated_pressure, axis=0)
min_avg_loc = np.argmin(avg_pressure)
max_avg_loc = np.argmax(avg_pressure)

print(f"\nLocation with lowest average pressure ({avg_pressure[min_avg_loc]:.2f} KPa):")
print(f"Lat {target_points[min_avg_loc, 1]:.2f}, Lon {target_points[min_avg_loc, 0]:.2f}")

print(f"\nLocation with highest average pressure ({avg_pressure[max_avg_loc]:.2f} KPa):")
print(f"Lat {target_points[max_avg_loc, 1]:.2f}, Lon {target_points[max_avg_loc, 0]:.2f}")

std_pressure = np.std(interpolated_pressure, axis=0)
min_std_loc = np.argmin(std_pressure)
max_std_loc = np.argmax(std_pressure)

print(f"\nLocation with lowest pressure variability (std dev: {std_pressure[min_std_loc]:.2f} KPa):")
print(f"Lat {target_points[min_std_loc, 1]:.2f}, Lon {target_points[min_std_loc, 0]:.2f}")

print(f"\nLocation with highest pressure variability (std dev: {std_pressure[max_std_loc]:.2f} KPa):")
print(f"Lat {target_points[max_std_loc, 1]:.2f}, Lon {target_points[max_std_loc, 0]:.2f}")

# Wind Speed
wind_speeds, grid_lats, grid_lons = extract_wind_speed_for_germany(nc_file_path)

print(f"Shape of extracted wind speed: {wind_speeds.shape}")
print(f"Sample of extracted wind speed (first 5 time steps, first 5 locations):")
print(wind_speeds[:5, :5])

# Interpolate wind data
interpolated_wind_speeds = interpolate_wind_speed(wind_speeds, grid_lats, grid_lons, target_points)

print(f"Shape of interpolated wind speeds: {interpolated_wind_speeds.shape}")
print(f"Sample of interpolated wind speeds (first 5 time steps, first 5 locations):")
print(interpolated_wind_speeds[:5, :5])

print(f"Number of NaN values: {np.isnan(interpolated_wind_speeds).sum()}")
print(f"Number of infinite values: {np.isinf(interpolated_wind_speeds).sum()}")

location_variation = np.any(np.diff(interpolated_wind_speeds, axis=1) != 0, axis=0)
time_variation = np.any(np.diff(interpolated_wind_speeds, axis=0) != 0, axis=1)

print(f"Number of locations with variation: {location_variation.sum()} out of {interpolated_wind_speeds.shape[1]}")
print(f"Number of time steps with variation: {time_variation.sum()} out of {interpolated_wind_speeds.shape[0]}")

location_stats = {
    'min': np.min(interpolated_wind_speeds, axis=0),
    'max': np.max(interpolated_wind_speeds, axis=0),
    'mean': np.mean(interpolated_wind_speeds, axis=0),
    'std': np.std(interpolated_wind_speeds, axis=0)
}

print("\nWind speed statistics across locations:")
for stat, values in location_stats.items():
    print(f"{stat.capitalize()}: min = {values.min():.2f}, max = {values.max():.2f}")

min_loc = np.unravel_index(np.argmin(location_stats['min']), location_stats['min'].shape)
max_loc = np.unravel_index(np.argmax(location_stats['max']), location_stats['max'].shape)

print(f"\nLocation with lowest minimum wind speed: Lat {target_points[min_loc[0], 1]:.2f}, Lon {target_points[min_loc[0], 0]:.2f}")
print(f"Location with highest maximum wind speed: Lat {target_points[max_loc[0], 1]:.2f}, Lon {target_points[max_loc[0], 0]:.2f}")

min_wind_index = np.unravel_index(np.argmin(interpolated_wind_speeds, axis=None), interpolated_wind_speeds.shape)
max_wind_index = np.unravel_index(np.argmax(interpolated_wind_speeds, axis=None), interpolated_wind_speeds.shape)

print(f"\nShape of interpolated wind speeds: {interpolated_wind_speeds.shape}")

print(f"\nMinimum wind speed of {interpolated_wind_speeds[min_wind_index]:.2f} m/s occurred at:")
print(f"Time step: {min_wind_index[0]}, Location: Lat {target_points[min_wind_index[1], 1]:.2f}, Lon {target_points[min_wind_index[1], 0]:.2f}")

print(f"\nMaximum wind speed of {interpolated_wind_speeds[max_wind_index]:.2f} m/s occurred at:")
print(f"Time step: {max_wind_index[0]}, Location: Lat {target_points[max_wind_index[1], 1]:.2f}, Lon {target_points[max_wind_index[1], 0]:.2f}")

avg_wind_speeds = np.mean(interpolated_wind_speeds, axis=0)
min_avg_loc = np.argmin(avg_wind_speeds)
max_avg_loc = np.argmax(avg_wind_speeds)

print(f"\nLocation with lowest average wind speed ({avg_wind_speeds[min_avg_loc]:.2f} m/s):")
print(f"Lat {target_points[min_avg_loc, 1]:.2f}, Lon {target_points[min_avg_loc, 0]:.2f}")

print(f"\nLocation with highest average wind speed ({avg_wind_speeds[max_avg_loc]:.2f} m/s):")
print(f"Lat {target_points[max_avg_loc, 1]:.2f}, Lon {target_points[max_avg_loc, 0]:.2f}")

std_wind_speeds = np.std(interpolated_wind_speeds, axis=0)
min_std_loc = np.argmin(std_wind_speeds)
max_std_loc = np.argmax(std_wind_speeds)

print(f"\nLocation with lowest wind speed variability (std dev: {std_wind_speeds[min_std_loc]:.2f} m/s):")
print(f"Lat {target_points[min_std_loc, 1]:.2f}, Lon {target_points[min_std_loc, 0]:.2f}")

print(f"\nLocation with highest wind speed variability (std dev: {std_wind_speeds[max_std_loc]:.2f} m/s):")
print(f"Lat {target_points[max_std_loc, 1]:.2f}, Lon {target_points[max_std_loc, 0]:.2f}")

# Temperature 2 meter
temperature_data, grid_lats, grid_lons = extract_temperature_for_germany(nc_file_path)

print("Shape of extracted temperature:", temperature_data.shape)
print("First 5 values:\n", temperature_data[:, :5])  
print("First time step pressure map:\n", temperature_data[0])

# Interpolate temperature data
interpolated_temperature = interpolate_temperature(temperature_data, grid_lats, grid_lons, target_points)
interpolated_temperature = interpolated_temperature - 273.15 # now in °C

print(f"Shape of interpolated temperature: {interpolated_temperature.shape}")
print(f"Sample of interpolated temperature (first 5 time steps, first 5 locations):")
print(interpolated_temperature[:5, :5])

print(f"Number of NaN values: {np.isnan(interpolated_temperature).sum()}")
print(f"Number of infinite values: {np.isinf(interpolated_temperature).sum()}")


location_variation = np.any(np.diff(interpolated_temperature, axis=1) != 0, axis=0)
time_variation = np.any(np.diff(interpolated_temperature, axis=0) != 0, axis=1)

print(f"Number of locations with variation: {location_variation.sum()} out of {interpolated_temperature.shape[1]}")
print(f"Number of time steps with variation: {time_variation.sum()} out of {interpolated_temperature.shape[0]}")

location_stats = {
    'min': np.min(interpolated_temperature, axis=0),
    'max': np.max(interpolated_temperature, axis=0),
    'mean': np.mean(interpolated_temperature, axis=0),
    'std': np.std(interpolated_temperature, axis=0)
}

print("\nWind speed statistics across locations:")
for stat, values in location_stats.items():
    print(f"{stat.capitalize()}: min = {values.min():.2f}, max = {values.max():.2f}")

min_loc = np.unravel_index(np.argmin(location_stats['min']), location_stats['min'].shape)
max_loc = np.unravel_index(np.argmax(location_stats['max']), location_stats['max'].shape)

print(f"\nLocation with lowest minimum temperature: Lat {target_points[min_loc[0], 1]:.2f}, Lon {target_points[min_loc[0], 0]:.2f}")
print(f"Location with highest maximum temperature: Lat {target_points[max_loc[0], 1]:.2f}, Lon {target_points[max_loc[0], 0]:.2f}")

min_temperature_index = np.unravel_index(np.argmin(interpolated_temperature, axis=None), interpolated_temperature.shape)
max_temperature_index = np.unravel_index(np.argmax(interpolated_temperature, axis=None), interpolated_temperature.shape)

print(f"\nShape of interpolated temperatures: {interpolated_temperature.shape}")

print(f"\nMinimum temperature of {interpolated_temperature[min_temperature_index]:.2f} °C occurred at:")
print(f"Time step: {min_temperature_index[0]}, Location: Lat {target_points[min_temperature_index[1], 1]:.2f}, Lon {target_points[min_temperature_index[1], 0]:.2f}")

print(f"\nMaximum temperature speed of {interpolated_temperature[max_temperature_index]:.2f} °C occurred at:")
print(f"Time step: {max_temperature_index[0]}, Location: Lat {target_points[max_temperature_index[1], 1]:.2f}, Lon {target_points[max_temperature_index[1], 0]:.2f}")

avg_temperature = np.mean(interpolated_temperature, axis=0)
min_avg_loc = np.argmin(avg_temperature)
max_avg_loc = np.argmax(avg_temperature)

print(f"\nLocation with lowest average temperature ({avg_temperature[min_avg_loc]:.2f} °C):")
print(f"Lat {target_points[min_avg_loc, 1]:.2f}, Lon {target_points[min_avg_loc, 0]:.2f}")

print(f"\nLocation with highest average temperature ({avg_temperature[max_avg_loc]:.2f} °C):")
print(f"Lat {target_points[max_avg_loc, 1]:.2f}, Lon {target_points[max_avg_loc, 0]:.2f}")

std_temperature = np.std(interpolated_temperature, axis=0)
min_std_loc = np.argmin(std_temperature)
max_std_loc = np.argmax(std_temperature)

print(f"\nLocation with lowest temperature variability (std dev: {std_temperature[min_std_loc]:.2f} °C):")
print(f"Lat {target_points[min_std_loc, 1]:.2f}, Lon {target_points[min_std_loc, 0]:.2f}")

print(f"\nLocation with highest temperature variability (std dev: {std_temperature[max_std_loc]:.2f} °C):")
print(f"Lat {target_points[max_std_loc, 1]:.2f}, Lon {target_points[max_std_loc, 0]:.2f}")



