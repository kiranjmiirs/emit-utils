"""
A simple script to extract and save GLT (Ground Look-up Table) from EMIT netCDFs as ENVI raster files.

Author: Adapted by Copilot, based on Philip G. Brodrick's original script
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

def extract_and_save_glt(nc_ds, output_dir, interleave='BIP', overwrite=False):
    """
    Extract GLT (glt_x, glt_y) from netCDF and save as ENVI raster file.
    """
    if 'location' not in nc_ds.groups:
        raise AttributeError('No "location" group with GLT data found in this netCDF file.')

    glt_x = np.array(nc_ds.groups['location']['glt_x'])
    glt_y = np.array(nc_ds.groups['location']['glt_y'])
    
    # Stack to (lines, samples, 2)
    glt = np.stack((glt_x, glt_y), axis=-1)
    output_name = os.path.join(output_dir, 'GLT')

    if os.path.isfile(output_name) and not overwrite:
        raise AttributeError(f'File {output_name} already exists. Use --overwrite to replace.')

    metadata = {
        'lines': glt.shape[0],
        'samples': glt.shape[1],
        'bands': 2,
        'interleave': interleave,
        'header offset': 0,
        'file type': 'ENVI Standard',
        'data type': envi_typemap[str(glt.dtype)],
        'byte order': 0,
        'band names': ['glt_x', 'glt_y'],
        'description': '{Ground Look-up Table (GLT) extracted from EMIT netCDF}'
    }

    envi_ds = envi.create_image(envi_header(output_name), metadata, ext='', force=overwrite)
    mm = envi_ds.open_memmap(interleave='bip', writable=True)
    mm[...] = glt
    del mm, envi_ds
    print(f'Saved GLT file: {output_name} (with .hdr)')

def main(rawargs=None):
    parser = argparse.ArgumentParser(description="Extract GLT from EMIT netCDF and save as ENVI raster.")
    parser.add_argument('input_netcdf', type=str, help='Input EMIT netCDF file.')
    parser.add_argument('output_dir', type=str, help='Directory to save the GLT ENVI file.')
    parser.add_argument('--interleave', type=str, default='BIP', choices=['BIL', 'BIP', 'BSQ'], help='ENVI interleave.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files.')
    args = parser.parse_args(rawargs)

    if not os.path.isdir(args.output_dir):
        raise AttributeError(f'Output directory {args.output_dir} does not exist - please create it and try again.')

    nc_ds = netCDF4.Dataset(args.input_netcdf, 'r', format='NETCDF4')
    extract_and_save_glt(nc_ds, args.output_dir, interleave=args.interleave, overwrite=args.overwrite)

if __name__ == "__main__":
    main()