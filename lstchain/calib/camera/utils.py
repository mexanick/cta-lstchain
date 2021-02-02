import tables
import numpy as np

from lstchain.io.io import dl1_params_tel_mon_ped_key, dl1_params_tel_mon_cal_key


def get_bias_and_std(dl1_file):
    """
    Function to extract bias and std of pedestal from interleaved events from dl1 file.
    Parameters
    ----------
    input_filename: str
        path to dl1 file
    Returns
    -------
    bias, std: np.ndarray, np.ndarray
        bias and std in p.e.
    """
    f = tables.open_file(dl1_file)
    ped = f.root[dl1_params_tel_mon_ped_key]
    ped_charge_mean = np.array(ped.cols.charge_mean)
    ped_charge_std = np.array(ped.cols.charge_std)
    calib = f.root[dl1_params_tel_mon_cal_key]
    dc_to_pe = np.array(calib.cols.dc_to_pe)
    ped_charge_mean_pe = ped_charge_mean * dc_to_pe
    ped_charge_std_pe = ped_charge_std * dc_to_pe
    f.close()
    return ped_charge_mean_pe, ped_charge_std_pe

def get_threshold_from_dl1_file(dl1_path, sigma_clean):
    """
    Function to get picture threshold from dl1 from interleaved pedestal events.
    Parameters
    ----------
    input_filename: str
        path to dl1 file
    sigma_clean: float
        cleaning level
    Returns
    -------
    picture_thresh: np.ndarray
        picture threshold calculated using interleaved pedestal events
    """
    high_gain = 0
    ped_mean_pe, ped_std_pe = get_bias_and_std(dl1_path)
    # If something bad happed with interleaved pedestal, take pedestal from calibration run
    if ped_std_pe.shape == (2, 2, 1855):
        interleaved_events_id = 1
    else:
        interleaved_events_id = 0
    threshold_clean_pe = ped_mean_pe + sigma_clean*ped_std_pe
    # find pixels with std = 0 and mean = 0 <=> dead pixels in interleaved
    # pedestal event likely due to stars
    dead_pixel_ids = np.where(threshold_clean_pe[interleaved_events_id, high_gain, :] == 0)[0]
    # for dead pixels set max value of threshold
    threshold_clean_pe[interleaved_events_id, high_gain, dead_pixel_ids] = \
        max(threshold_clean_pe[interleaved_events_id, high_gain, :])
    # return pedestal interleaved threshold from data run for high gain
    return threshold_clean_pe[interleaved_events_id, high_gain, :]
