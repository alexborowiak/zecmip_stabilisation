import numpy as np
import pandas as pd
import xarray as xr

import utils

logger = utils.get_notebook_logger()



def base_a1_merge(ds_1pct_raw, ds_a1_raw, lower_limit=40, logginglevel='ERROR'):
    """
    Calculate the average of a specific branch in a dataset.

    This function calculates the average of a specific branch in a dataset by first identifying the branch start time
    and then selecting a time window around that branch start time for averaging.

    Args:
        ds_1pct (xarray.Dataset): Dataset containing the 1% CO2 increase data.
        ds_a1 (xarray.Dataset): Dataset containing the branch point data.
        upper_limit (int, optional): Upper time limit (number of time steps after the branch point) for averaging.
            Default is 10.
        lower_limit (int, optional): Lower time limit (number of time steps before the branch point) for averaging.
            Default is 10.
        logginglevel (str, optional): Logging level for the 'utils' module. Default is 'ERROR'.

    Returns:
        xarray.Dataset: A dataset containing the average values of the selected time window around the branch point.
    """
    ds_a1 = ds_a1_raw.copy()
    ds_1pct= ds_1pct_raw.copy()

    # Set the logging level for the 'utils' module
    utils.change_logginglevel(logginglevel)
    logger.info(f'{ds_a1.model.values} - {ds_1pct.model.values}')

    # Get the branch start time
    branch_start_time = ds_a1.time.values[0]

    # Get time values from ds_1pct
    onepct_time_values = ds_1pct.time.values
    logger.info(f'{branch_start_time=}')

    # Find the index of the branch start time in onepct_time_values
    onepct_branch_arg = np.where(onepct_time_values == branch_start_time)[0][0]

    logger.info(f'{onepct_branch_arg=}')

    # Select a time slice around the branch point
    ds_1pct_branch_slice = ds_1pct.isel(
        time=slice(onepct_branch_arg - lower_limit, onepct_branch_arg))

    ds_a1['time'] = np.arange(len(ds_a1.time))
    ds_1pct_branch_slice['time'] = np.arange(-len(ds_1pct_branch_slice.time.values), 0, 1)

    ds = ds_a1.combine_first(ds_1pct_branch_slice)


    return ds


def isolate_base_period(ds_1pct, ds_a1, selection_length=10, logginglevel='ERROR'):
    """
    Calculate the average of a specific branch in a dataset.

    This function calculates the average of a specific branch in a dataset by first identifying the branch start time
    and then selecting a time window around that branch start time for averaging.

    Args:
        ds_1pct (xarray.Dataset): Dataset containing the 1% CO2 increase data.
        ds_a1 (xarray.Dataset): Dataset containing the branch point data.
        upper_limit (int, optional): Upper time limit (number of time steps after the branch point) for averaging.
            Default is 10.
        lower_limit (int, optional): Lower time limit (number of time steps before the branch point) for averaging.
            Default is 10.
        logginglevel (str, optional): Logging level for the 'utils' module. Default is 'ERROR'.

    Returns:
        xarray.Dataset: A dataset containing the average values of the selected time window around the branch point.
    """

    # Set the logging level for the 'utils' module
    utils.change_logginglevel(logginglevel)
    logger.info(f'{ds_a1.model.values} - {ds_1pct.model.values}')

    # Get the branch start time
    branch_start_time = ds_a1.time.values[0]

    # Get time values from ds_1pct
    onepct_time_values = ds_1pct.time.values
    logger.info(f'{branch_start_time=}')

    # Find the index of the branch start time in onepct_time_values
    onepct_branch_arg = np.where(onepct_time_values == branch_start_time)[0][0]

    logger.info(f'{onepct_branch_arg=}')

    # Select a time slice around the branch point
    ds_1pct_branch_slice = ds_1pct.isel(
        time=slice(onepct_branch_arg - selection_length, onepct_branch_arg + selection_length))

    logger.debug(f'Time values around branch point: {len(ds_1pct_branch_slice.time.values)}')
    logger.debug(ds_1pct_branch_slice.time.values)


    return ds_1pct_branch_slice


def calculate_branch_average(ds_1pct, ds_a1, upper_limit=10, lower_limit=10, logginglevel='ERROR'):
    """
    Calculate the average of a specific branch in a dataset.

    This function calculates the average of a specific branch in a dataset by first identifying the branch start time
    and then selecting a time window around that branch start time for averaging.

    Args:
        ds_1pct (xarray.Dataset): Dataset containing the 1% CO2 increase data.
        ds_a1 (xarray.Dataset): Dataset containing the branch point data.
        upper_limit (int, optional): Upper time limit (number of time steps after the branch point) for averaging.
            Default is 10.
        lower_limit (int, optional): Lower time limit (number of time steps before the branch point) for averaging.
            Default is 10.
        logginglevel (str, optional): Logging level for the 'utils' module. Default is 'ERROR'.

    Returns:
        xarray.Dataset: A dataset containing the average values of the selected time window around the branch point.
    """

    # Set the logging level for the 'utils' module
    utils.change_logginglevel(logginglevel)
    logger.info(f'{ds_a1.model.values} - {ds_1pct.model.values}')

    # Get the branch start time
    branch_start_time = ds_a1.time.values[0]

    # Get time values from ds_1pct
    onepct_time_values = ds_1pct.time.values
    logger.info(f'{branch_start_time=}')

    # Find the index of the branch start time in onepct_time_values
    onepct_branch_arg = np.where(onepct_time_values == branch_start_time)[0][0]

    logger.info(f'{onepct_branch_arg=}')

    # Select a time slice around the branch point
    ds_1pct_branch_slice = ds_1pct.isel(
        time=slice(onepct_branch_arg - lower_limit, onepct_branch_arg + upper_limit))

    logger.debug(f'Time values around branch point: {len(ds_1pct_branch_slice.time.values)}')
    logger.debug(ds_1pct_branch_slice.time.values)

    # Calculate the mean of the selected time window
    branch_point_mean = ds_1pct_branch_slice.mean(dim='time')
    logger.info('\n')

    return branch_point_mean
