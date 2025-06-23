"""
A script to extract longitude, latitude, and elevation from a netCDF file's 'location' group, and save as an ENVI raster (LOC file).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import netCDF4
import numpy as np
from spectral.io import envi
from emit_utils.file_checks import envi_header

envi_typemap = {
    'uint8': 1,
    'int16': 2,
    'int32': 3,
    'float32': 4,
    'float64': 5,
    'complex64': 6,
    'complex128': 9,
    'uint16': 12,
    'uint32': 13,
    'int64': 14,
    'uint64': 15
}

def extract_and_save_loc(nc_ds, output_dir, interleave='BIP', overwrite=False):
    """
    Extract lon, lat, and elev from netCDF/'location' group and save as ENVI raster file.
    """
    # Check for the 'location' group
    if 'location' not in nc_ds.groups:
        raise AttributeError("No 'location' group found in netCDF file.")

    loc_group = nc_ds.groups['location']

    # Try common variable names in the group
    lon = None
    lat = None
    elev = None

    lon_names = ['longitude', 'lon', 'Longitude', 'LON']
    lat_names = ['latitude', 'lat', 'Latitude', 'LAT']
    elev_names = ['elevation', 'elev', 'Elevation', 'ELEV', 'z']

    for name in lon_names:
        if name in loc_group.variables:
            lon = np.array(loc_group.variables[name])
            break
    for name in lat_names:
        if name in loc_group.variables:
            lat = np.array(loc_group.variables[name])
            break
    for name in elev_names:
        if name in loc_group.variables:
            elev = np.array(loc_group.variables[name])
            break

    if lon is None or lat is None or elev is None:
        raise AttributeError("Could not find lon, lat, or elev in 'location' group of netCDF variables.")

    # Stack into (lines, samples, 3) [order: lon, lat, elev]
    loc = np.stack([lon, lat, elev], axis=-1)
    output_name = os.path.join(output_dir, 'LOC')

    if os.path.isfile(output_name) and not overwrite:
        raise AttributeError(f'File {output_name} already exists. Use --overwrite to replace.')

    dtype_name = str(loc.dtype)
    metadata = {
        'lines': loc.shape[0],
        'samples': loc.shape[1],
        'bands': 3,
        'interleave': interleave,
        'header offset': 0,
        'file type': 'ENVI Standard',
        'data type': envi_typemap.get(dtype_name, 4),  # default to float32 if not found
        'byte order': 0,
        'band names': ['longitude', 'latitude', 'elevation'],
        'description': '{Longitude, Latitude, Elevation extracted from netCDF location group}'
    }

    envi_ds = envi.create_image(envi_header(output_name), metadata, ext='', force=overwrite)
    mm = envi_ds.open_memmap(interleave='bip', writable=True)
    mm[...] = loc
    del mm, envi_ds
    print(f'Saved LOC file: {output_name} (with .hdr)')

def main(rawargs=None):
    parser = argparse.ArgumentParser(description="Extract longitude, latitude, and elevation from netCDF/'location' group and save as ENVI raster (LOC file).")
    parser.add_argument('input_netcdf', type=str, help='Input netCDF file.')
    parser.add_argument('output_dir', type=str, help='Directory to save the LOC ENVI file.')
    parser.add_argument('--interleave', type=str, default='BIP', choices=['BIL', 'BIP', 'BSQ'], help='ENVI interleave.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files.')
    args = parser.parse_args(rawargs)

    if not os.path.isdir(args.output_dir):
        raise AttributeError(f'Output directory {args.output_dir} does not exist - please create it and try again.')

    nc_ds = netCDF4.Dataset(args.input_netcdf, 'r', format='NETCDF4')
    extract_and_save_loc(nc_ds, args.output_dir, interleave=args.interleave, overwrite=args.overwrite)

if __name__ == "__main__":
    main()