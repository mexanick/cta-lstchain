#!/usr/bin/env python3

"""
Script to create DL3 of single (merged) DL2 run along with IRFs generated from lstchain_mc_dl2_to_irf.py.

The selection cuts applied are the same as those used in generating IRFs.

 - Input: Path where the merged DL2 data HDF5 file is present
          Source name
          IRFs
 - Output: DL3 of the input DL2 data file in fits format.

Usage:
$> python lstchain_dl2_to_dl3.py
--input-data ./DL2/dl2_LST-1.Run*.h5
--output-fits-dir ./DL3/
--add-irf True
--input-irf-file ./IRF/irf.fits.gz
--source-name Crab
--config ../../data/data_selection_cuts.json
"""

import os
import json
from distutils.util import strtobool
import numpy as np
import argparse
from pathlib import Path
import logging
import sys

from lstchain.irf import create_event_list
from lstchain.io import read_data_dl2_to_QTable, read_configuration_file, get_standard_config
from lstchain.reco.utils import filter_events
from lstchain.paths import run_info_from_filename

from astropy.io import fits
import astropy.units as u
from astropy.coordinates.angle_utilities import angular_separation

from pyirf.utils import calculate_source_fov_offset
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="DL2 to DL3")

# Required arguments
parser.add_argument('--input-data', '-d', type=Path,
                    dest='input_data',
                    help='path to merged DL2 data HDF5 file',
                    default=None, required=True
                    )

parser.add_argument('--output-fits-dir', '-o', type=Path,
                    dest='output_fits_dir',
                    help='path to output fits files',
                    default=None, required=True
                    )

parser.add_argument('--add-irf', '-add-irf', action='store',
		            type=lambda x: bool(strtobool(x)), dest='add_irf',
                    help='Boolean: True to add IRF to DL3',
                    default=False, required=True
                    )

parser.add_argument('--input-irf-file', '-irf', type=Path, dest='irf',
                    help='Path to the fits.gz file of IRFs',
                    default=None, required=False
                    )

parser.add_argument('--source-name', '-s', type=str,
                    dest='source_name',
                    help='Name of the source',
                    required=True
                    )

parser.add_argument('--config', '-conf', type=Path,
                    dest='config',
                    help='Config file for selection cuts',
                    default=None, required=False
                    )
args = parser.parse_args()

def main():

    if not args.input_data.is_file():
        log.error('Input Path does not exist or is not a file')
        sys.exit(1)
    file = str(args.input_data).split('/')[-1]

    output_dir = args.output_fits_dir.absolute()
    output_dir.mkdir(exist_ok=True)

    data = read_data_dl2_to_QTable(args.input_data)

    data['reco_source_fov_offset'] = calculate_source_fov_offset(data, prefix='reco')

    # Get the run_id from the filename if it is -1 in the obs_id column
    if data['obs_id'][0] <= 0:
        run_number=run_info_from_filename(args.input_data)[1]
    else:
        run_number= data['obs_id'][0]

    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)

    if args.config is None:
        cuts = read_configuration_file(os.path.join(os.path.dirname(__file__), '../data/data_selection_cuts.json'))
    else:
        cuts = read_configuration_file(args.config)

    data = filter_events(data, cuts["events_filters"])

    # Separate cuts for angular separations, for now. Will be included later in filter_events
    data = data[data["gh_score"] > cuts["fixed_cuts"]["gh_score"][0]]

    data = data[data["reco_source_fov_offset"] < u.Quantity(**cuts["fixed_cuts"]["source_fov_offset"])]

    # Create primary HDU
    events, gti, pointing = create_event_list(data=data, run_number=run_number,
                    source_name=args.source_name)

    name_dl3_file = file.replace('dl2', 'dl3')
    name_dl3_file = name_dl3_file.replace('h5', 'fits')

    if args.add_irf:
        irf = fits.open(args.irf)
        aeff2d = irf['EFFECTIVE AREA']
        edisp2d = irf['ENERGY DISPERSION']
        # bkg2d = irf['BACKGROUND']
        # psf = irf['PSF']
        hdulist = fits.HDUList([fits.PrimaryHDU(), events, gti, pointing, aeff2d, edisp2d])
    else:
        hdulist = fits.HDUList([fits.PrimaryHDU(), events, gti, pointing])
    hdulist.writeto((args.output_fits_dir/name_dl3_file),overwrite=True)

if __name__ == '__main__':
    main()
