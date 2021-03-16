#!/usr/bin/env python3

"""
Pipeline for the reconstruction of Energy, disp and gamma/hadron
separation of events stored in a simtelarray file.
- Input: DL1 files and trained Random Forests.
- Output: DL2 data file.

Usage:

$> python lstchain_dl1_to_dl2.py 
--input-file dl1_LST-1.Run02033.0137.h5
--path-models ./trained_models

"""

import argparse
import numpy as np
import os
import pandas as pd
from tables import open_file
import joblib
from ctapipe.instrument import SubarrayDescription
from lstchain.reco.utils import filter_events, impute_pointing, add_delta_t_key
from lstchain.reco import dl1_to_dl2
from lstchain.io import (
    read_configuration_file,
    standard_config,
    replace_config,
    write_dl2_dataframe,
    get_dataset_keys,
)
from lstchain.io.io import (
    dl1_params_src_dep_lstcam_key,
    dl1_images_lstcam_key,
    dl2_params_lstcam_key,
    dl2_params_src_dep_lstcam_key,
    write_dataframe,
)

parser = argparse.ArgumentParser(description="DL1 to DL2")

# Required arguments
parser.add_argument('--input-file', '-f', type=str,
                    dest='input_file',
                    help='path to a DL1 HDF5 file',
                    default=None, required=True)

parser.add_argument('--path-models', '-p', action='store', type=str,
                    dest='path_models',
                    help='Path where to find the trained RF',
                    default='./trained_models')

# Optional arguments
parser.add_argument('--output-dir', '-o', action='store', type=str,
                    dest='output_dir',
                    help='Path where to store the reco dl2 events',
                    default='./dl2_data')

parser.add_argument('--config', '-c', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file. If none is given, a standard configuration is applied',
                    default=None, required=False)


args = parser.parse_args()


def main():
    custom_config = {}
    if args.config_file is not None:
        try:
            custom_config = read_configuration_file(os.path.abspath(args.config_file))
        except("Custom configuration could not be loaded !!!"):
            pass

    config = replace_config(standard_config, custom_config)

    dataset_keys = get_dataset_keys(args.input_file)
    if not config['source_dependent']:
        params_keys = [k for k in dataset_keys if 'parameters/' in k]
    else:
        params_keys = [k for k in dataset_keys if 'parameters_src_dependent/' in k]
    # Drop unwanted DL1 keys
    dataset_keys = [k for k in dataset_keys if 'image/' not in k]
    dataset_keys = [k for k in dataset_keys if 'parameters/' not in k]
    dataset_keys = [k for k in dataset_keys if 'parameters_src_dependent/' not in k]

    for tel_id in config['source_config']['EventSource']['allowed_tels']:
        for k in params_keys:
            if k.endswith(f'{tel_id}'):

                data = pd.read_hdf(args.input_file, key=k)

                # if real data, add deltat t to dataframe keys
                data = add_delta_t_key(data)

                # Dealing with pointing missing values. This happened when `ucts_time` was invalid.
                if 'alt_tel' in data.columns and 'az_tel' in data.columns \
                        and (np.isnan(data.alt_tel).any() or np.isnan(data.az_tel).any()):
                    # make sure there is a least one good pointing value to interp from.
                    if np.isfinite(data.alt_tel).any() and np.isfinite(data.az_tel).any():
                        data = impute_pointing(data)
                    else:
                        data.alt_tel = - np.pi/2.
                        data.az_tel = - np.pi/2.

                #Load the trained RF for reconstruction:
                fileE = f"{args.path_models}/{k.split('/')[-1]}/reg_energy.sav"
                fileD = f"{args.path_models}/{k.split('/')[-1]}/reg_disp_vector.sav"
                fileH = f"{args.path_models}/{k.split('/')[-1]}/cls_gh.sav"

                reg_energy = joblib.load(fileE)
                reg_disp_vector = joblib.load(fileD)
                cls_gh = joblib.load(fileH)

                #Apply the models to the data

                #Source-independent analysis
                if not config['source_dependent']:
                    data = filter_events(data,
                                         filters=config["events_filters"],
                                         finite_params=config['regression_features'] + config['classification_features'],
                                     )

                    dl2 = dl1_to_dl2.apply_models(data, cls_gh, reg_energy, reg_disp_vector, custom_config=config)

                #Source-dependent analysis
                if config['source_dependent']:
                    data_srcdep = pd.read_hdf(args.input_file, key=k)
                    data_srcdep.columns = pd.MultiIndex.from_tuples([tuple(col[1:-1].replace('\'', '').replace(' ','').split(",")) for col in data_srcdep.columns])

                    dl2_srcdep_dict = {}

                    for i, j in enumerate(data_srcdep.columns.levels[0]):
                        data_with_srcdep_param = pd.concat([data, data_srcdep[j]], axis=1)
                        data_with_srcdep_param = filter_events(data_with_srcdep_param,
                                                           filters=config["events_filters"],
                                                           finite_params=config['regression_features'] + config['classification_features'],
                                                       )
                        dl2_df = dl1_to_dl2.apply_models(data_with_srcdep_param, cls_gh, reg_energy, reg_disp_vector, custom_config=config)

                        dl2_srcdep = dl2_df.drop(data.keys(), axis=1)
                        dl2_srcdep_dict[j] = dl2_srcdep

                        if i==0:
                            dl2_srcindep = dl2_df.drop(data_srcdep[j].keys(), axis=1)

                os.makedirs(args.output_dir, exist_ok=True)
                output_file = os.path.join(args.output_dir, os.path.basename(args.input_file).replace('dl1','dl2'))

                with open_file(args.input_file, 'r') as h5in:
                    with open_file(output_file, 'a') as h5out:

                        # Write the selected DL1 info
                        for key_to_write in dataset_keys:
                            if not key_to_write.startswith('/'):
                                key_to_write = '/' + key_to_write

                            path, name = key_to_write.rsplit('/', 1)
                            if path not in h5out:
                                grouppath, groupname = path.rsplit('/', 1)
                                g = h5out.create_group(
                                    grouppath, groupname, createparents=True
                                    )
                            else:
                                g = h5out.get_node(path)

                            h5in.copy_node(key_to_write, g, overwrite=True)

                key_to_write = k.replace('dl1', 'dl2')
                if not config['source_dependent']:
                    write_dataframe(dl2, output_file, key_to_write)
                else:
                    write_dataframe(dl2_srcindep, output_file,
                                        key_to_write.replace('parameters_src_dependent', 'parameters'))
                    write_dataframe(pd.concat(dl2_srcdep_dict, axis=1), output_file, k)


if __name__ == '__main__':
    main()
