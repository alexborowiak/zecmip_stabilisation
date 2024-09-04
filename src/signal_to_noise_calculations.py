import os
import sys
import dask
import itertools
import pandas as pd
import numpy as np
import xarray as xr
from functools import partial
from itertools import takewhile
from typing import Optional, List
from numpy.typing import ArrayLike

# Custom Module Imports
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'zecmip_stabilisation'))
import constants
sys.path.append(constants.MODULE_DIR)
import utils
import xarray_extender as xe
logger = utils.get_notebook_logger()
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'list_xarray'))
from listXarray import listXarray

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module='dask.*')
warnings.filterwarnings("ignore", category=Warning)




def grid_gradient(
    arr: ArrayLike, 
    axis: int, 
    xs: ArrayLike = None, 
    mean_xs: float = None, 
    denominator: float = None
) -> float:
    """
    Calculate a gradient-like signal along a specified axis in a 2D array.
    This is a more efficient method for calculating the gradient than inbuilt Python methods.

    Args:
    - arr (ArrayLike): Input array.
    - axis (int): Axis along which to calculate the gradient.
    - xs (ArrayLike, optional): Array of indices along the specified axis. Defaults to np.arange(arr.shape[axis]).
    - mean_xs (float, optional): Mean of xs. Defaults to np.nanmean(xs).
    - denominator (float, optional): Denominator for the gradient calculation. Defaults to np.mean(xs) ** 2 - np.mean(xs ** 2).

    Returns:
    - float: The calculated gradient signal.
    """
    def mult_func(arr1: ArrayLike, arr2: ArrayLike) -> ArrayLike:
        """Element-wise multiplication function."""
        return arr1 * arr2

    if xs is None:
        xs = np.arange(arr.shape[axis])
    if denominator is None:
        denominator = np.mean(xs) ** 2 - np.mean(xs ** 2)
    if mean_xs is None:
        mean_xs = np.nanmean(xs)

    xs_mult_arr = np.apply_along_axis(mult_func, axis=axis, arr=arr, arr2=xs)
    numerator = mean_xs * np.nanmean(arr, axis=axis) - np.nanmean(xs_mult_arr, axis=axis)
    return numerator / denominator


def adjust_time_from_rolling(data, window, logginglevel='ERROR'):
        """
        Adjusts time points in the dataset by removing NaN values introduced by rolling operations.
    
        Parameters:
        - window (int): The size of the rolling window.
        - logginglevel (str): The logging level for debugging information ('ERROR', 'WARNING', 'INFO', 'DEBUG').
    
        Returns:
        - data_adjusted (xarray.Dataset): Dataset with adjusted time points.
    
        Notes:
        - This function is designed to handle cases where rolling operations introduce NaN values at the edges of the
        dataset.
        - The time points are adjusted to remove NaN values resulting from rolling operations with a specified window
        size.
        - The position parameter controls where the adjustment is made: 'start', 'start', or 'end'.
    
        """
        # Change the logging level based on the provided parameter
        utils.change_logging_level(logginglevel)
    
        # Calculate the adjustment value for the time points
        time_adjust_value = int((window - 1) / 2) + 1

        # If the window is even, adjust the time value back by one
        if window % 2:
            time_adjust_value = time_adjust_value - 1
    
        # Log the adjustment information
        logger.debug(f'Adjusting time points by {time_adjust_value}')
    
        # Remove NaN points on either side introduced by rolling with min_periods
        data_adjusted = data.isel(time=slice(time_adjust_value, -time_adjust_value))
    
        # Ensure the time coordinates match the adjusted data
        # The default option is the middle
        adjusted_time_length = len(data_adjusted.time.values)

        time_slice = slice(0, adjusted_time_length)
        new_time = data.time.values[time_slice]
        data_adjusted['time'] = new_time
    
        return data_adjusted


def rolling_signal(
    data: xr.DataArray, 
    window: int, 
    min_periods: int = 0, 
    start_align:bool = True, 
    method: str = 'gradient', 
    logginglevel: str = 'ERROR'
) -> xr.DataArray:
    """
    Calculate a rolling signal in a dataset based on a specified method.

    Args:
    - data (xr.DataArray): Input dataset.
    - window (int, optional): Rolling window size. Defaults to 20.
    - min_periods (int, optional): Minimum number of periods. Defaults to 0.
    - start_align (bool): Wether the value is assigned to the start or the middle. Defauls to True
    - method (str, optional): Calculation method. Defaults to 'gradient'.
    - logginglevel (str, optional): Logging level. Defaults to 'ERROR'.

    Returns:
    - xr.DataArray: The calculated rolling signal.
    """
    utils.change_logging_level(logginglevel)
    logger.info(f"Calculating the rolling signal with method {method}")

    if min_periods == 0: min_periods = window
    elif min_periods == 'half': min_periods = int(window/2)

    logger.debug(f"{window=}, {min_periods=}\ndata=\n{data}")

    # New x values
    xs = np.arange(window)
    # Mean of the x-values
    mean_xs = np.nanmean(xs)
    # Denominator can actually always be the same
    denominator = np.mean(xs) ** 2 - np.mean(xs ** 2)
    signal_da = (data
                 .rolling(time=window, min_periods=min_periods, center=True)
                 .reduce(grid_gradient, xs=xs, mean_xs=mean_xs, denominator=denominator
                        )) 
    # Multiply by window length to get signal from gradient
    signal_da = signal_da * window


    if start_align == True:
        signal_da = adjust_time_from_rolling(signal_da, window, logginglevel)
    # else:
    #     signal_da = signal_da.dropna(dim='time')


    signal_da.name = 'signal'
    signal_da = signal_da.expand_dims('window').assign_coords(window=('window', [window]))
    return signal_da


def rolling_noise(data, window:int, min_periods=0, start_align:bool=True,logginglevel='ERROR') -> xr.DataArray:
    '''
    - start_align (bool): Wether the value is assigned to the start or the middle. Defauls to True
    '''
    
    utils.change_logging_level(logginglevel)

    logger.info("Calculting the rolling noise")

    
    if min_periods == 0: min_periods = window
    elif min_periods == 'half': min_periods = int(window/2)
    
    # Rolling standard deviation
    noise_da = \
       data.rolling(time = window, min_periods = min_periods, center = True).std()

    if start_align == True:
        noise_da = adjust_time_from_rolling(noise_da, window=window, logginglevel=logginglevel)
    # if center == True:
    #     noise_da = adjust_time_from_rolling(noise_da, window=window, logginglevel=logginglevel)
    # else:
    #     noise_da = noise_da.dropna(dim='time')
    
    noise_da.name = 'noise'
    
    noise_da = noise_da.expand_dims('window').assign_coords(window=('window', [window]))
    
    return noise_da


def static_noise(data, logginglevel='ERROR', *args, **kwargs) -> xr.DataArray:
    
    utils.change_logging_level(logginglevel)

    logger.info("Calculting the static noise")

    noise_da = data.std(dim='time')
    noise_da.name = 'noise'
    #noise_da = noise_da.expand_dims('window').assign_coords(window=('window', [window]))
    
    return noise_da


def signal_to_noise_ratio(
    ds: xr.Dataset, 
    window:int,
    detrended_data: xr.Dataset = None, 
    start_align:bool=True,
    noise_type = 'rolling',
    return_all: bool = False,
    logginglevel='ERROR',
    *args,
    **kwargs
) -> xr.Dataset:
    """
    Calculate the signal-to-noise ratio for a given dataset.

    Args:
    - ds (xr.Dataset): Input dataset.
    - window(int): The window length for the calculations.
    - detrended_data (xr.Dataset, optional): Detrended dataset. Defaults to None.
    - return_all (bool, optional): Whether to return all datasets (signal, noise, and ratio). Defaults to False.

    Returns:
    - xr.Dataset: The signal-to-noise ratio dataset. If return_all is True, returns a tuple of signal, noise, and
    ratio datasets.
    """
    # Calculate the rolling signal
    utils.change_logginglevel(logginglevel)
    signal_ds = rolling_signal(ds, window, start_align=start_align, logginglevel=logginglevel, *args, **kwargs)  
    # Use the rolling_signal function to calculate the signal

    # Calculate the rolling noise
    # If detrended data is provided, use it; otherwise, use the original dataset
    if noise_type == 'rolling':
        noise_func = rolling_noise
        noise_func = partial(noise_func, start_align=start_align)
        noise_func = partial(noise_func, window=window)
    elif noise_type == 'static': noise_func = static_noise
    logger.info(f'{noise_type=}')
    noise_ds = noise_func(ds if detrended_data is None else detrended_data, logginglevel=logginglevel, *args, **kwargs)  
    logger.debug(noise_ds)
    # Calculate the signal-to-noise ratio
    sn_ratio_ds = signal_ds / noise_ds  # Divide the signal by the noise to get the ratio

    sn_ratio_ds.name = 'sn'
    # Return the desired datasets
    if return_all:
        return signal_ds, noise_ds, sn_ratio_ds  # Return all datasets if requested
    return sn_ratio_ds  # Otherwise, return only the signal-to-noise ratio dataset


def upper_and_lower_bounds(ds, qlower, qupper):
    # Choose percentile function based on whether data is chunked
    percentile_func = xe.dask_percentile if ds.chunks else np.nanpercentile

    # Calculate upper and lower bounds of SNR using specified percentiles
    ub_ds = ds.reduce(percentile_func, q=qupper, dim='time').compute()
    lb_ds = ds.reduce(percentile_func, q=qlower, dim='time').compute()

    # Merge upper and lower bounds into a single dataset
    bounds_ds = xr.merge(
        [
            lb_ds.to_dataset(name='lower_bound'),
            ub_ds.to_dataset(name='upper_bound')
        ], 
        compat='override'
    ).compute()

    return bounds_ds


def signal_to_noise_ratio_bounds(ds, window:int, **kwargs):
    """
    Calculate the upper and lower bounds of the signal-to-noise ratio (SNR) for a given dataset.

    Parameters:
    - ds (xr.Dataset): Input dataset.
    - window (int): Window size for calculating SNR.
    - **kwargs: Additional keyword arguments to be passed to the `signal_to_noise_ratio` function.
                `qlower` and `qupper` will be used for percentile calculation.

    Returns:
    - xr.Dataset: Dataset containing the upper and lower bounds of SNR.

    Notes:
    - This function calculates the upper and lower bounds of the SNR using percentiles.
    - The SNR is calculated using the `signal_to_noise_ratio` function.
    """

    # Extract qlower and qupper from kwargs
    qlower = kwargs.pop('qlower', 1)
    qupper = kwargs.pop('qupper', 99)

    
    if 'chunks' in kwargs: 
        chunks = kwargs.pop('chunks')

    # print(kwargs)


    # Calculate signal-to-noise ratio
    sn_ratio_ds = signal_to_noise_ratio(ds=ds, window=window, **kwargs).compute()

    sn_ratio_ds = sn_ratio_ds.chunk(chunks)

    sn_ratio_bounds_ds = upper_and_lower_bounds(sn_ratio_ds, qlower, qupper)

    return sn_ratio_bounds_ds


def signal_to_noise_ratio_bounds_multi_window(
    ds, 
    windows: ArrayLike, 
    **kwargs
) -> xr.Dataset:
    """
    Calculate signal-to-noise ratio bounds for multiple windows.

    Parameters:
    ds (xr.Dataset): Input dataset
    windows (ArrayLike): List of window sizes
    **kwargs: Additional keyword arguments to pass to signal_to_noise_ratio_bounds

    Returns:
    xr.Dataset: Dataset containing signal-to-noise ratio bounds for each window
    """
    # Suppress warnings for All-NaN slices
    # Suppress warnings for All-NaN slices
    logginglevel = kwargs.pop('logginglevel', 'ERROR')
    utils.change_logginglevel(logginglevel)

    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered",
                                category=RuntimeWarning)
        
        # Initialize list to store results
        to_concat = []

        parallel = kwargs.pop('parallel', False)
        if parallel:
            signal_to_noise_ratio_bounds_func = dask.delayed(signal_to_noise_ratio_bounds)
        else:
            signal_to_noise_ratio_bounds_func = signal_to_noise_ratio_bounds
        # Loop over each window size
        for window in windows:
            logger.info(window)
            out_data = signal_to_noise_ratio_bounds_func(ds, window, **kwargs)
            # Calculate signal-to-noise ratio bounds for current window
            to_concat.append(out_data)
        if parallel: to_concat = dask.compute(*to_concat)
        # Concatenate results along a new dimension named 'window'
        outpout_ds = xr.concat(to_concat, dim='window')
    return outpout_ds
    
#####!!!!!! Batch processing - better version but has not been tests.
# def signal_to_noise_ratio_bounds_multi_window(
#     ds, 
#     windows: ArrayLike, 
#     **kwargs
# ) -> xr.Dataset:
#     """
#     Calculate signal-to-noise ratio bounds for multiple windows.

#     Parameters:
#     ds (xr.Dataset): Input dataset
#     windows (ArrayLike): List of window sizes
#     **kwargs: Additional keyword arguments to pass to signal_to_noise_ratio_bounds

#     Returns:
#     xr.Dataset: Dataset containing signal-to-noise ratio bounds for each window
#     """
#     # Suppress warnings for All-NaN slices
#     logginglevel = kwargs.pop('logginglevel', 'ERROR')
#     utils.change_logginglevel(logginglevel)

#     # Handle warning suppression
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", message="All-NaN slice encountered",
#                                 category=RuntimeWarning)
        
#         # Initialize list to store results
#         to_concat = []

#         parallel = kwargs.pop('parallel', False)
#         batch_size = kwargs.pop('batch', None)  # Retrieve batch size from kwargs

#         # Determine the function to use based on parallel flag
#         if parallel:
#             signal_to_noise_ratio_bounds_func = dask.delayed(signal_to_noise_ratio_bounds)
#         else:
#             signal_to_noise_ratio_bounds_func = signal_to_noise_ratio_bounds

#         def process_batch(windows_batch, ds, **kwargs):
#             # Process a batch of windows
#             to_concat_batch = []
#             for window in windows_batch:
#                 logger.info(window)
#                 out_data = signal_to_noise_ratio_bounds_func(ds, window, **kwargs)
#                 to_concat_batch.append(out_data)
#             # Compute results if running in parallel
#             to_concat_batch = dask.compute(*to_concat_batch)
#             return to_concat_batch

#         # Process windows in batches if batch_size is specified
#         if parallel and batch_size and isinstance(batch_size, int):
#             for i in range(0, len(windows), batch_size):
#                 batch_windows = windows[i:i + batch_size]  # Get current batch of windows
#                 to_concat.extend(process_batch(batch_windows, ds, **kwargs))
#         else:
#             # Process all windows at once if batch size is not specified
#             for window in windows:
#                 logger.info(window)
#                 out_data = signal_to_noise_ratio_bounds_func(ds, window, **kwargs)
#                 to_concat.append(out_data)
#             if parallel:
#                 to_concat = dask.compute(*to_concat)

#         # Concatenate results along a new dimension named 'window'
#         outpout_ds = xr.concat(to_concat, dim='window')

    return outpout_ds


def multi_window_func(
    func,
    ds, 
    windows: ArrayLike, 
    parallel=True,
    **kwargs
) -> xr.Dataset:
    """
    Calculate signal-to-noise ratio for multiple windows.

    Parameters:
    ds (xr.Dataset): Input dataset
    windows (ArrayLike): List of window sizes
    **kwargs: Additional keyword arguments to pass to signal_to_noise_ratio

    Returns:
    xr.Dataset: Dataset containing signal-to-noise ratio for each window
    """
    logginglevel = kwargs.pop('logginglevel', 'ERROR')
    utils.change_logginglevel(logginglevel)

    logger.debug(f'Multi window func - applying function {func}')

    # Using dask delayed or not?
    func = dask.delayed(func) if parallel else func
    
  
    # Initialize list to store results
    to_concat = []
    
    # Loop over each window size
    for window in windows:
        logger.info(window)
        # Calculate signal-to-noise ratio for current window
        if isinstance(ds, listXarray):
            output_data = ds(func, window, **kwargs)
        else:
            output_data = func(ds, window, **kwargs)
        to_concat.append(output_data)

    # Compute the dask object
    if parallel:  to_concat = dask.compute(*to_concat)
    # Concatenate results along a new dimension named 'window' and compute
    if isinstance(to_concat[0], (xr.Dataset, xr.DataArray)):
        result_ds = xr.concat(to_concat, dim='window').compute()
    else:
        result_ds = to_concat
    return result_ds





def multi_window_func_with_model_split(
    func,
    ds, 
    windows: ArrayLike, 
    parallel=True,
    **kwargs
) -> xr.Dataset:
    """
    Calculate signal-to-noise ratio for multiple windows.

    Parameters:
    ds (xr.Dataset): Input dataset
    windows (ArrayLike): List of window sizes
    **kwargs: Additional keyword arguments to pass to signal_to_noise_ratio

    Returns:
    xr.Dataset: Dataset containing signal-to-noise ratio for each window
    """
    model_output_list = []
    for model in ds.model.values:
        logger.info(model)
        if isinstance(esmpi_data, listXarray):
            ds_model = ds[model].dropna(dim='time')
            # Call the first function
            result_ds = multi_window_func(func, ds_model, windows, parallel, **kwargs)
        else: 
            ds_model = ds.sel(model=model).dropna(dim='time')
            # Call the first function
            result_ds = multi_window_func(func, ds_model, windows, parallel, **kwargs)
        model_output_list.append(result_ds)
        
    to_retrun_ds = xr.concat(model_output_list, dim='model')
    return to_retrun_ds



def signal_to_noise_ratio_multi_window(
    ds, 
    windows: ArrayLike,
    parallel=False,
    **kwargs
) -> xr.Dataset:
    """
    Calculate signal-to-noise ratio for multiple windows.

    Parameters:
    ds (xr.Dataset): Input dataset
    windows (ArrayLike): List of window sizes
    **kwargs: Additional keyword arguments to pass to signal_to_noise_ratio

    Returns:
    xr.Dataset: Dataset containing signal-to-noise ratio for each window
    """
    
    logginglevel = kwargs.get('logginglevel', 'ERROR')
    utils.change_logginglevel(logginglevel)

    # Using dask delayed or not?
    signal_to_noise_ratio_func = dask.delayed(signal_to_noise_ratio) if parallel else signal_to_noise_ratio
    
    # Suppress warnings for All-NaN slices
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
    
        # Initialize list to store results
        to_concat = []
        
        # Loop over each window size
        for window in windows:
            # logger.info(window)
            # Calculate signal-to-noise ratio for current window
            sn_ratio_ds = signal_to_noise_ratio_func(ds, window, **kwargs)
            to_concat.append(sn_ratio_ds)

        # Compute the dask object (for some reason make it list of list)
        if parallel: 
            to_concat = dask.compute(to_concat)[0]
        # Concatenate results along a new dimension named 'window' and compute
        output_ds = xr.concat(to_concat, dim='window').compute()
    output_ds.name = 'sn'
    return output_ds



def get_increase_and_decreasing_stability_number(ds:xr.Dataset) -> xr.DataArray:
    '''
    Dataset needs to have the data_vars: 'signal_to_noise', 'upper_bound', 'lower_bounds'
    Divides the datset into inncreasing unstable, decreasing unstable, and stable.
    
    These can then be counted to view the number of unstable and stable models at
    any point.
    
    '''
    decreasing_unstable_da = ds.where(ds.signal_to_noise < ds.lower_bound).signal_to_noise
    increasing_unstable_da = ds.where(ds.signal_to_noise > ds.upper_bound).signal_to_noise
    
    
    stable_da = ds.utils.between('signal_to_noise',
                                 less_than_var='upper_bound', greater_than_var='lower_bound').signal_to_noise
    unstable_da = ds.utils.above_or_below(
        'signal_to_noise', greater_than_var='upper_bound', less_than_var='lower_bound').signal_to_noise
    
    return xr.concat([decreasing_unstable_da, increasing_unstable_da, unstable_da, stable_da], 
                     pd.Index(['decreasing', 'increasing', 'unstable', 'stable'], name='stability'))


def get_average_after_stable_year(arr, year):
    """
    Calculate the average of a subset of the input array, starting from the given year and extending 20 years forward.

    Parameters:
    arr (numpy array): The input array
    year (int or float): The starting year (will be converted to integer)

    Returns:
    float: The average of the subset array, or the input year if it's NaN

    """
    # Check if the input year is NaN, return it as is
    if np.isnan(year): return year
    
    # Convert the year to an integer
    year = int(year)
    
    # Extract a subset of the array, starting from the given year and extending 20 years forward
    arr_subset = arr[year:year+20]
    
    # Calculate and return the mean of the subset array
    return np.mean(arr_subset)




def count_consecutive_ones(arr: List[int]) -> int:
    """
    Counts the number of consecutive 1s at the start of the array.

    Parameters:
    arr (List[int]): The input list containing 1s and 0s.

    Returns:
    int: The count of consecutive 1s at the beginning of the array.
    """
    # Use takewhile to generate an iterator that stops when the first 0 is encountered
    consecutive_ones = takewhile(lambda x: x == 1, arr)
    
    # Count the elements returned by takewhile
    count = sum(1 for _ in consecutive_ones)
    
    return count

def find_stability_index(arr: np.ndarray, window: int, fraction: float = 0.5) -> int:
    """
    Determines the index at which stability is achieved in the input array.

    The algorithm works by sliding a window of a specified length over the array,
    checking each subset to determine if the fraction of unstable values (i.e., 
    values that are not finite) is below a certain threshold. If the fraction of 
    unstable values in a subset is below the threshold, the index where stability 
    is achieved is determined based on the last unstable value within the subset.

    If stability is achieved in the current subset, the function further checks if 
    the next element after the stable period is also stable by counting the number 
    of consecutive stable elements. The final index of stability is adjusted 
    accordingly.

    If stability is never achieved within the array, the function returns an 
    index calculated as the last checked index plus half the window length.

    Parameters:
    arr (np.ndarray): The input array containing numerical values.
    window (int): The length of the window used for stability checking.
    fraction (float): The threshold fraction of instability (default is 0.5).

    Returns:
    int: The index at which stability is first achieved.
    """
    finite_arr = np.isfinite(arr)

    lenght_of_selection = np.min([20, window])#window#int(window/2)

    for sel_start in range(len(finite_arr)-lenght_of_selection):
        # Make sub-selection
        finite_arr_selection = finite_arr[sel_start:sel_start+lenght_of_selection]
        # Number that are unstable (unstable =1, stable = 0)
        number_unstable = np.sum(finite_arr_selection)
        # Fraction that is unstalbe
        fraction_unstable = number_unstable/lenght_of_selection
        
        # If < 0.2 is unstalbe, then call stability immediately. Do not add consecs     
        if fraction_unstable < 0.33:
            return sel_start

        # If the fraction is less than fraciton - stability is achieved
        if fraction_unstable < fraction:
            # Figure out where the last unstable value occurs in selection
            last_arg = np.argwhere(finite_arr_selection==1)
            last_arg = last_arg[-1][-1] if len(last_arg) > 0 else 0
            last_arg = last_arg+1 # Always one behind, so add 1
            stable_year = sel_start + last_arg
            
            # The next element is one after the stable year - so why should it be stable there?
            if finite_arr[stable_year+1] == 1: 
                cosec_after_stabilisation = count_consecutive_ones(finite_arr[stable_year+1:])
                stable_year = stable_year + cosec_after_stabilisation+1
            return stable_year

    # Stability never achieved return arg + the offset
    return sel_start + int(window/2)

def get_percent_non_nan(arr, window):
    """
    Calculates the percentage of non-NaN (finite) values within a moving window subset of the array.
    
    Parameters:
    ----------
    arr : numpy.ndarray
        1D array in which to calculate the fraction of non-NaN values.
    window : int
        The size of the window used to create the subset of the array.
        
    Returns:
    -------
    numpy.ndarray
        1D array of the same length as the input array, containing the fraction of non-NaN values 
        within each moving window. The last `subset_legnth` elements are filled with NaN to maintain 
        the same length as the input array.
    """
    # Calculate the length of the subset based on the window size
    subset_legnth = np.min([int(window/2), 10])
    
    # Initialize an array to hold the fraction of non-NaN values for each subset
    fraction_non_nan_arr = []
    
    # Iterate over the array with the defined window size
    for t in range(len(arr) - subset_legnth):
        # Extract the current subset of the array
        arr_subset = arr[t:t + subset_legnth]
        
        # Count the number of non-NaN values in the subset
        number_non_nan = np.sum(np.isfinite(arr_subset))
        
        # Calculate the fraction of non-NaN values
        fraction_non_nan = number_non_nan / subset_legnth
        
        # Store the result
        fraction_non_nan_arr.append(fraction_non_nan)
    
    # Pad the result with NaN values to match the length of the input array
    fraction_non_nan_arr = np.concatenate([fraction_non_nan_arr,
                                           np.tile(np.nan, subset_legnth)])
    
    return np.array(fraction_non_nan_arr)


def get_percent_non_zero(arr, window):
    """
    Calculates the percentage of non-NaN (finite) values within a moving window subset of the array.
    
    Parameters:
    ----------
    arr : numpy.ndarray
        1D array in which to calculate the fraction of non-NaN values.
    window : int
        The size of the window used to create the subset of the array.
        
    Returns:
    -------
    numpy.ndarray
        1D array of the same length as the input array, containing the fraction of non-NaN values 
        within each moving window. The last `subset_legnth` elements are filled with NaN to maintain 
        the same length as the input array.
    """
    # Calculate the length of the subset based on the window size
    subset_legnth = np.min([int(window/2), 10])
    
    # Initialize an array to hold the fraction of non-NaN values for each subset
    fraction_non_nan_arr = []
    
    # Iterate over the array with the defined window size
    for t in range(len(arr) - subset_legnth):
        # Extract the current subset of the array
        arr_subset = arr[t:t + subset_legnth]
        
        # Count the number of non-NaN values in the subset
        number_non_nan = np.sum(arr_subset)
        
        # Calculate the fraction of non-NaN values
        fraction_non_nan = number_non_nan / subset_legnth
        
        # Store the result
        fraction_non_nan_arr.append(fraction_non_nan)
    
    # Pad the result with NaN values to match the length of the input array
    fraction_non_nan_arr = np.concatenate([fraction_non_nan_arr,
                                           np.tile(np.nan, subset_legnth)])
    
    return np.array(fraction_non_nan_arr)


def get_last_arg_v2(arr):
    """
    Finds the last index where the array has a value of 1 and returns the 1-based index.
    
    Parameters:
    ----------
    arr : numpy.ndarray
        1D array in which to find the last occurrence of the value 1.
        
    Returns:
    -------
    int
        The 1-based index of the last occurrence of 1 in the array. 
        Returns 1 if no such occurrence is found.
    """
    # Find all indices where the array equals 1
    last_arg = np.argwhere(arr == 1)
    
    # Select the last index, or 0 if no 1s are found
    last_arg = last_arg[-1][-1] if len(last_arg) > 0 else 0
    
    # Add 1 to the result to convert to a 1-based index
    last_arg = last_arg + 1
    
    return last_arg


def find_stable_year_unsable_window_sel(unstable_pattern_arr, unstable_fraction_arr, windows):
    """
    This function finds the first year in which all windows become stable. It checks when a pattern
    becomes stable across different windows and returns the total year when stability is achieved.

    Parameters:
    ----------
    unstable_pattern_arr : numpy.ndarray
        2D array representing unstable patterns across different windows and years.
    unstable_fraction_arr : numpy.ndarray
        2D array representing the fraction of instability across different windows and years.
        The shape should be the transpose of `unstable_pattern_arr`.

    Returns:
    -------
    int
        The first year in which all windows become stable.

    Raises:
    ------
    AssertionERROR:
        If the shape of `unstable_pattern_arr` does not match the reversed shape of
        `unstable_fraction_arr`.
    """
    
    # Ensure that the shapes of the arrays are compatible
    assert unstable_pattern_arr.shape == unstable_fraction_arr.shape[::-1]
    
    # Sum instability across years for each window
    stable_num_arr = np.sum(unstable_fraction_arr, axis=1)

    # Find the first year where all windows are stable (no instability)
    first_year_all_stable = np.where(stable_num_arr == 0)[0][0]

    # If the first year is stable from the start, return this year
    if first_year_all_stable == 0:
        return first_year_all_stable

    # The year before the first fully stable year
    stable_point_query_year = first_year_all_stable - 1

    # Find windows that are unstable in the year before full stability
    windows_that_are_unstable = unstable_fraction_arr[stable_point_query_year, :]

    # Get indices of the unstable windows
    window_unstable_args = np.where(windows_that_are_unstable == 1)[0]

    larst_arg_list = []

    for sarg in window_unstable_args:
        # Select the window size for analysis
        window = windows[sarg]

        # Set the length of the selection window, max of 20 or the window size
        lenght_of_selection = np.min([20, window])

        # Select the analysis window: data from the unstable window at the query year onwards
        # Ensure the dimension order is window first
        anlsysis_window = unstable_pattern_arr[sarg, stable_point_query_year:stable_point_query_year + lenght_of_selection]

        # Find the last stable index (where the data is finite) in the analysis window
        last_arg = get_last_arg_v2(np.isfinite(anlsysis_window))
        larst_arg_list.append(last_arg)

    larst_arg_list = np.array(larst_arg_list)

    # Calculate the maximum extension of stability required across the unstable windows
    stable_year_addition = np.max(larst_arg_list)

    # Calculate the total year when stability is achieved
    total_year_stable = first_year_all_stable + stable_year_addition
    
    return total_year_stable


def frac_non_zero(time_arr, period_length: int = 10, logginglevel='ERROR'):
    """
    Calculate the percentage of non-NaN values within specified windows for each time series.

    Args:
        time_window_arr (np.ndarray): 2D array where each column represents a different time series.
        windows (list of int): List of window sizes to apply across the time series.
        period_length (int): Length of the period over which the fraction is calculated. Default is 10.

    Returns:
        np.ndarray: 2D array with the same shape as `time_window_arr` where each element represents 
                    the percentage of non-NaN values within the corresponding window.
    """
    if isinstance(logginglevel, str): utils.change_logginglevel(logginglevel)
    
    # Calculate the fraction of non-NaN values in each window.
    # sel_length = period_length
    number_non_zero = np.array([
        np.nansum(time_arr[t:t + period_length]) for t in range(len(time_arr) - period_length + 1)
    ])
    logger.info(number_non_zero)
    frac_non_zero = number_non_zero / period_length
    logger.info(frac_non_zero)

    frac_non_zero = np.concatenate([frac_non_zero, np.tile(np.nan, len(time_arr)-len(frac_non_zero))])

    return frac_non_zero


def frac_non_zero_window(time_window_arr, period_length: int = 10, logginglevel='ERROR'):
    """
    Calculate the percentage of non-NaN values within specified windows for each time series.

    Args:
        time_window_arr (np.ndarray): 2D array where each column represents a different time series.
        windows (list of int): List of window sizes to apply across the time series.
        period_length (int): Length of the period over which the fraction is calculated. Default is 10.

    Returns:
        np.ndarray: 2D array with the same shape as `time_window_arr` where each element represents 
                    the percentage of non-NaN values within the corresponding window.
    """
    if isinstance(logginglevel, str):
        utils.change_logginglevel(logginglevel)
    
    # Initialize an array to store the percentage of non-NaN values.
    frac_non_zero_2d = np.zeros_like(time_window_arr).astype(np.float64)#, np.nan)

    # Iterate through each column (time series).
    for num in range(time_window_arr.shape[1]):
        # Extract the time series data for the current window.
        time_arr = time_window_arr[:, num]

        # Calculate the fraction of non-NaN values in each window.
        # sel_length = period_length
        number_non_zero = np.array([
            np.nansum(time_arr[t:t + period_length]) for t in range(len(time_arr) - period_length + 1)
        ])
        logger.info(number_non_zero)
        frac_non_zero = number_non_zero / period_length
        logger.info(frac_non_zero)


        # Place the computed fractions back into the result array.
        frac_non_zero_2d[:len(frac_non_zero), num] = frac_non_zero

    return frac_non_zero_2d


def remove_periods_below_threshold(year_vals, threshold=10):
    """
    Remove values from the array where the difference between consecutive elements
    in stable periods is below a given threshold. Stable periods are defined as the
    segments between changes in differences. The function removes the values associated
    with stable periods that have lengths below the threshold.

    Parameters:
    year_vals (list or np.array): The input list or array of values to be filtered.
    threshold (int, optional): The minimum allowed length of stable periods. Default is 10.

    Returns:
    np.array: The filtered array with stable periods below the threshold removed.
    """
    # Convert the input to a numpy array for easier manipulation
    year_vals = np.array(year_vals)

    # Compute the differences between consecutive values
    lengths = np.diff(year_vals)

    # Extract the stable period lengths (even indexed differences)
    period_lengths = lengths[::2]

    # Identify stable periods below the threshold
    length_below_threshold = period_lengths <= threshold

    # Determine indices to remove based on the stable periods
    small_period_args = np.where(length_below_threshold)[0]
    to_remove = np.sort(np.concatenate([small_period_args*2, small_period_args*2+1]))

    # Remove the identified indices from the array
    year_vals = np.delete(year_vals, to_remove)

    return year_vals


# def remove_closest_pairs_below_threshold(year_list, threshold=10):
#     """
#     Remove pairs of consecutive values from a list where the difference between the values is below a given threshold.
#     The function removes the two values with the smallest difference iteratively until all remaining differences 
#     between consecutive values are equal to or above the threshold.

#     Parameters:
#     year_list (list or np.array): The input list or array of years to be filtered.
#     threshold (int, optional): The minimum allowed difference between consecutive values. Default is 10.

#     Returns:
#     np.array: The filtered array with all differences between consecutive values equal to or above the threshold.
#     """
#     # Convert the input to a numpy array for easier manipulation
#     year_list = np.array(year_list)

#     # Calculate differences between consecutive values
#     diffs = np.diff(year_list)

#     # Iterate while there are any differences below the threshold
#     while np.any(diffs < threshold):
#         # Find the index of the smallest difference
#         min_diff_arg = np.argmin(diffs)

#         # Remove the two values with the smallest difference
#         # First, remove the element at min_diff_arg + 1
#         year_list = np.delete(year_list, min_diff_arg + 1)
#         # Then, remove the element at min_diff_arg (which has shifted one position left)
#         year_list = np.delete(year_list, min_diff_arg)

#         # Recalculate differences after removal
#         diffs = np.diff(year_list)

#     return year_list


def search_for_instability(time_window_arr, frac_unstable_arr, windows, period_length, logginglevel='ERROR'):
    
    utils.change_logginglevel(logginglevel)
        
    logger.info('\nInstability Search\n------\n')

    # Set a threshold where instability is defined as the fraction of unstable points > 0.4.
    instability_condition = frac_unstable_arr >= 0.5
    logger.debug(f' - Instability_condition shape: {instability_condition.shape}')

    # Create a binary array where 1 indicates instability.
    frac_unstable_threshold_arr = np.where(instability_condition, 1, 0)

    # If no instability is detected, break the loop.
    if np.all(frac_unstable_threshold_arr == 0): return np.nan

    # Count the number of unstable points across all windows.
    number_unstable_across_window = np.nansum(frac_unstable_threshold_arr, axis=1)
    logger.debug(f' - number_unstable_across_window\n{number_unstable_across_window}')

    # Find the first year where instability is detected.
    first_year_condition_met = np.where(number_unstable_across_window > 0)[0][0]
    logger.debug(f' - {first_year_condition_met=}')

    window_args = np.where(frac_unstable_threshold_arr[first_year_condition_met, :]==1)[0]
    logger.debug(f' - {window_args=}')
    logger.debug(f' - windows that are unstable\n{windows[window_args]}')


    first_arg_list = []
    for sarg in window_args:
        # length_of_selection = period_length#10#np.min([10, int(window)])
        # Select the window size for analysis
        window = windows[sarg]
        logger.debug(f' - {sarg=} {window=}')
        anlsysis_window =\
            time_window_arr[first_year_condition_met:first_year_condition_met + period_length+1, sarg]
         # time_window_arr[:, sarg]

        logger.debug(f' - anlsysis_window {anlsysis_window.shape}\n{anlsysis_window}')
        first_unstable_point = np.where(anlsysis_window==1)[0][0]
        first_arg_list.append(first_unstable_point)
    logger.debug(f' - first_arg_list\n{first_arg_list}')
    first_arg_list = np.array(first_arg_list)
    year_addition = np.nanmin(first_arg_list)
    logger.debug(f' - {year_addition=}')

    year_val = first_year_condition_met+year_addition+1
    return year_val


def search_for_stability(time_window_arr, frac_unstable_arr, windows, period_length, i, logginglevel='ERROR'):
    utils.change_logginglevel(logginglevel)
    logger.info('\nStability Search\n----------\n')
    # Set a threshold where stability is defined as the fraction of unstable points <= 0.2.
    # Shape time x window  
    frac_stable_time_window = np.where(frac_unstable_arr < 0.5, 1, 0)

    # Count the number of unstable points across all windows.
    # Shape time
    number_windows_stable = np.nansum(frac_stable_time_window, axis=1)
    logger.debug(f'number_windows_stable:\n{number_windows_stable}')

    # If every window is stable
    # Shape: time
    where_all_windows_stable = number_windows_stable == len(windows)
    logger.debug(f'The locations where all windows are stable \n{where_all_windows_stable}')

    if i == 0:
        logger.debug('')
        first_unstable_location = np.where(where_all_windows_stable==0)[0][0]
        logger.debug(f'  - first_unstable_location\n{first_unstable_location}')

        where_all_windows_stable = where_all_windows_stable[first_unstable_location:]
        frac_stable_time_window = frac_stable_time_window[first_unstable_location:]
        time_window_arr = time_window_arr[first_unstable_location:]

    # print(where_all_windows_stable)
    # If stability is not found, break the loop.
    if np.all(where_all_windows_stable == False): return np.nan

    # Find the first year where stability is detected.
    first_year_condition_met = np.where(where_all_windows_stable)[0][0]

    #Identify the year before the first fully stable year.
    point_query_year = first_year_condition_met - 1
    logger.debug(f' - {point_query_year=}')
    assert point_query_year >= 0 , ('The point query year is less than 0')


    # Extract values from the time series at the query year.
    # val_at_query_windows = time_window_arr[point_query_year, :]
    val_at_query_windows = frac_stable_time_window[point_query_year, :]
    logger.debug(f'Values at window {val_at_query_windows}')

    # Find the windows that are stable at the query year.
    # This shoudl be == 0, as we are looking for when they 
    # are not stable (e.g unstable)
    window_args = np.where(val_at_query_windows == 0)[0]
    logger.debug(f' - windows that are unstable\n{windows[window_args]}')
    logger.debug(f' - window_args\n{window_args}')
    

    larst_arg_list = []
    # print(window_args)
    for sarg in window_args:
        # Get the window size for analysis.
        window = windows[sarg]

        # Select the analysis window starting from the query year.
        analysis_window = time_window_arr[point_query_year:point_query_year + period_length, sarg]

        # Find the last stable index in the analysis window.
        last_arg = get_last_arg_v2(analysis_window)
        larst_arg_list.append(last_arg)

    # Calculate the additional years needed for stability.
    larst_arg_list = np.array(larst_arg_list)
    year_addition = np.max(larst_arg_list)
    year_val = first_year_condition_met + year_addition + 1
    if i == 0:
        year_val = year_val + first_unstable_location

    return year_val


def calcuate_year_stable_and_unstable(time_window_arr, frac_unstable_arr, windows, period_length:int=10, number_attempts: int = 7, 
                                      max_val:int=50, logginglevel='ERROR'):
    """
    Calculate the years of stability and instability in a time series.
    
    Parameters:
    - time_window_arr (np.ndarray): A 2D array representing the time series data for different windows.
    - windows (np.ndarray): An array containing the size of each window.
    - number_attempts (int): The maximum number of iterations for searching stability/instability. Default is 5.
    - period_length (int): The length of that stabilility has to occur for before the climate is
                             consdiered stable.
    
    Returns:
    - year_list (np.ndarray): A cumulative sum array representing the years at which stability or instability is identified.
      If the number of identified years is less than the number of attempts, the remaining positions in the array will be NaN.


    Notes on window subtraction
    - Looking for stability.
        - THis should occur one after the last time there is a one

    - Looking for instability
        - This should occur the first time there is a 1
    """
    # Convert the time series data to a binary array where 1 represents
    # finite values and 0 represents NaNs or infinite values.
    utils.change_logging_level(logginglevel)
    assert time_window_arr.shape[-1] == len(windows), (
    f"Assertion failed: The last dimension of time_window_arr has length {time_window_arr.shape[-1]} "
    f"but the nubmer of windows is {len(windows)}. The -1 dimensions shoudl be window. Data may need to be transpoed.")


    # This is not used functinoally, but used defensivly
    window_shape = time_window_arr.shape[-1]

    # Number of attempts is needed later
    number_attemps_2 = number_attempts
    
    # time_window_arr = np.where(np.isfinite(time_window_arr), 1, 0)
    
    next_year_list = []

    # Calculate the fraction of unstable points within each window.
    # time x window
    logger.debug(f'{period_length=}')
    # frac_unstable_arr = frac_non_zero_window(time_window_arr, period_length, None)
    logger.info(f'Shapes - time_window_arr = {time_window_arr.shape}, frac_unstable_arr = {frac_unstable_arr.shape}')
    # Using 5 years here, as if the very last of the 10 years has an unstable fraction
    
    initial_fracs = frac_unstable_arr[:period_length, :]
    logger.info(f'initial_fracs shape {initial_fracs.shape}')
    with np.printoptions(threshold=np.inf):
        logger.debug(f' - inital_fracs (shape = {initial_fracs.shape})\n{initial_fracs}')

    if np.any(initial_fracs >= 0.5):
        logger.info('Fracs above 0.5 found - test for stability')
        # If any of the fraction are greater than 0.5, then we are unstable 
        # and need to start looking for stability
        TEST_FOR_STABILITY = True
        TEST_FOR_INSTABILITY = False
    else:
        logger.info('Fracs above 0.5 not found - test for instability')

        # Otherwise, we have started off with being stable
        # so start testing for instability
        TEST_FOR_STABILITY = False
        TEST_FOR_INSTABILITY = True
        # Append stable year 0 to array
        next_year_list.append(0)
        # We have already used one attempt
        number_attempts = number_attempts - 1
        
    i = 0
    while number_attempts >= 0:
        if i != 0:
            # Check if the end of the time series has been reached.
            if next_year_list[-1] >= time_window_arr.shape[0]: break

            
            if next_year_list[-1] == 0: selection = 0
            else: selection = next_year_list[-1]-1

            # The first time we are doing this, we don't want to subtract as this
            # has not occured year
            if next_year_list[-1] > 0: next_year_list[-1] = next_year_list[-1]-1

            # Subset the two datasets
            time_window_arr = time_window_arr[selection:, :]
            frac_unstable_arr = frac_unstable_arr[selection:, :]
    
            # Break if the remaining time series is too short for further analysis.
            if time_window_arr.shape[0] < (period_length+5): break
  
        if TEST_FOR_INSTABILITY: 
            year_val = search_for_instability(time_window_arr, frac_unstable_arr, windows, period_length, logginglevel)
            logger.debug(f' - {year_val=}')
            TEST_FOR_STABILITY = True
            TEST_FOR_INSTABILITY = False
    
        elif TEST_FOR_STABILITY: 
            year_val = search_for_stability(time_window_arr, frac_unstable_arr, windows,period_length, i, logginglevel)
            logger.debug(f' - {year_val=}')
            TEST_FOR_STABILITY = False
            TEST_FOR_INSTABILITY = True
        
        

        if np.isnan(year_val): break
        # If getting close to max value => break
        if len(next_year_list)>1:
            if np.cumsum(next_year_list)[-1] > max_val-5: break

        i+=1
        number_attempts = number_attempts - 1

        if year_val == 1: year_val = 0
        next_year_list.append(year_val)

        logger.info(f' - next_year_list\n{next_year_list}')
        logger.info(f'- Currunt cumsum\n{np.cumsum(next_year_list)}')
        # If the condition analysed is more than 45, break
        if np.cumsum(next_year_list)[-1] > max_val-5: break
        logger.info('Complete\n')


    logger.info(f'\n - Search complete - final processing')
    # Calculate the cumulative sum of the years to get the year list.
    logger.info(f' = nex_year_list final form\n{next_year_list}')
    year_list = np.cumsum(next_year_list)
    logger.info(f' = year_list (cumsum applied )\n{year_list}')
    # Due to the need to select one year before the year of change
    # this introduced small ERRORs that accumulate
    # position 1 will be too large by one year
    # position 2 will be too large by 2 years
    # etc.
    if len(year_list) > 1:
        for num in range(len(year_list)):
            if num%2:
                year_list[num] = year_list[num]+1 #- num

    year_list = remove_periods_below_threshold(year_list, int(np.ceil(period_length/2)))
    
    # End point conditions
    # # If there are any values greater than 50, replace them with 50
    if np.any(year_list>(max_val)):
        year_list[np.where(year_list>(max_val))] = max_val
        logger.debug(f'Value greater than max ({max_val=}) found and removed \n{year_list}')

    # IF the last is an unstable, but it has become unstalbe just beofre the end
    # then get rid of
    # if len(year_list)%2 and len(year_list)>1:
    #     if year_list[-1] > max_val-3:
    #         # Thus, remove the last value
    #         year_list = year_list[:-1]
        
    
    # # If not even (e.g. 1, 3) then finishing on instability
    # Need to assign unstable
    if not len(year_list)%2 and len(year_list)>1: 
        # Instability has been assigned at the last year
        # So chop of the last point
        if year_list[-1] > max_val-3:
            year_list = year_list[:-1]
        else:
            year_list = np.concatenate([year_list,[max_val]])

    # The model nevera actuyall stabilised, so assign 50
    if len(year_list) == 0:
        year_list = np.array([max_val])
        
    # Check how many 50s there are
    number_of_max_vals = len(np.where(year_list==max_val)[0])
    # IF there is more than one 50, remove
    if number_of_max_vals > 1:
        year_list = year_list[:-(number_of_max_vals-1)]

    
    # Ensure the year list has the same size as the number of attempts.
    # I is upwards counts of number of attempts
    if len(year_list) < number_attemps_2:
        year_list = np.concatenate([year_list, np.tile(np.nan, number_attemps_2 - len(year_list))])

    if len(year_list) != number_attemps_2:
        logger.info(f' - year_list length {len(year_list)} {number_attemps_2=}')
        logger.info(f' - year_list\n{year_list}')
        logger.info('\n')

    
    logger.info(year_list.shape)
    logger.info(f'final result {year_list}')
    return year_list


###########

def search_for_instability_v2(time_window_arr, frac_unstable_arr, windows, period_length, logginglevel='ERROR'):
    
    utils.change_logginglevel(logginglevel)
        
    logger.info('\nInstability Search\n------\n')

    # Set a threshold where instability is defined as the fraction of unstable points > 0.4.
    # instability_condition = frac_unstable_arr >= 0.5
    instability_condition = frac_unstable_arr
    logger.debug(f' - Instability_condition shape: {instability_condition.shape}')

    # Create a binary array where 1 indicates instability.
    frac_unstable_threshold_arr = np.where(instability_condition, 1, 0)

    # If no instability is detected, break the loop.
    if np.all(frac_unstable_threshold_arr == 0): return np.nan

    # Count the number of unstable points across all windows.
    number_unstable_across_window = np.nansum(frac_unstable_threshold_arr, axis=1)
    logger.debug(f' - number_unstable_across_window\n{number_unstable_across_window}')

    # Find the first year where instability is detected.
    first_year_condition_met = np.where(number_unstable_across_window > 0)[0][0]
    logger.debug(f' - {first_year_condition_met=}')

    window_args = np.where(frac_unstable_threshold_arr[first_year_condition_met, :]==1)[0]
    logger.debug(f' - {window_args=}')
    logger.debug(f' - windows that are unstable\n{windows[window_args]}')

    # Extract rows for the time window
    row_extraction = np.take(time_window_arr, range(first_year_condition_met, first_year_condition_met + period_length), axis=0)
    logger.debug(f'row_extraction ({row_extraction.shape})\n{row_extraction}')
    
    # Extract specific columns using np.take() with window_args
    column_row_extraction = np.take(row_extraction, window_args, axis=1)
    logger.debug(f'column_row_extraction ({column_row_extraction.shape})\n{column_row_extraction}')

    last_arg_array_sum = np.sum(column_row_extraction, axis=1)
    logger.debug(f'last_arg_array_sum ({last_arg_array_sum.shape})\n{last_arg_array_sum}')

    first_arg = np.where(last_arg_array_sum!=0)[0][0]
    # first_arg_list = []
    # for sarg in window_args:
    #     # length_of_selection = period_length#10#np.min([10, int(window)])
    #     # Select the window size for analysis
    #     window = windows[sarg]
    #     logger.debug(f' - {sarg=} {window=}')
    #     anlsysis_window =\
    #         time_window_arr[first_year_condition_met:first_year_condition_met + period_length+1, sarg]
    #      # time_window_arr[:, sarg]

    #     logger.debug(f' - anlsysis_window {anlsysis_window.shape}\n{anlsysis_window}')
    #     first_unstable_point = np.where(anlsysis_window==1)[0][0]
    #     first_arg_list.append(first_unstable_point)
    logger.debug(f' - first_arg\n{first_arg}')
    # first_arg_list = np.array(first_arg)
    # year_addition = np.nanmin(first_arg)
    # logger.debug(f' - {year_addition=}')

    year_val = first_year_condition_met+first_arg+1
    return year_val


def search_for_stability_v2(time_window_arr, frac_unstable_arr, windows, period_length, i, logginglevel='ERROR'):
    utils.change_logginglevel(logginglevel)
    logger.info('\nStability Search\n----------\n')
    # Set a threshold where stability is defined as the fraction of unstable points <= 0.2.
    # Shape time x window  
    # frac_stable_time_window = np.where(frac_unstable_arr < 0.5, 1, 0)

    # Count the number of unstable points across all windows.
    # Shape time
    number_windows_unstable = np.nansum(frac_unstable_arr, axis=1)
    logger.debug(f'number_windows_unstable:\n{number_windows_unstable}')

    # True or flase if all windows are stable are not
    # We are looking for the arg that == 0
    # Shape: time
    where_all_windows_stable = number_windows_unstable == 0 #len(windows)


    if i == 0:
        logger.info(f'{i=} identified')
        first_unstable_location = np.where(number_windows_unstable!=0)[0][0]
        logger.debug(f'  - {first_unstable_location=}')
        
        where_all_windows_stable = where_all_windows_stable[first_unstable_location:]
        frac_unstable_arr = frac_unstable_arr[first_unstable_location:]
        time_window_arr = time_window_arr[first_unstable_location:]

    logger.debug(f'The locations where all windows are stable \n{where_all_windows_stable}')
    # If stability is not found, break the loop.
    if np.all(where_all_windows_stable == False): return np.nan

    # Find the first year where stability is detected.
    first_year_condition_met = np.where(where_all_windows_stable)[0][0]

    #Identify the year before the first fully stable year.
    point_query_year = first_year_condition_met - 1
    logger.debug(f' - {point_query_year=}')
    assert point_query_year >= 0 , ('The point query year is less than 0')


    # Extract values from the time series at the query year.
    # val_at_query_windows = time_window_arr[point_query_year, :]
    val_at_query_windows = frac_unstable_arr[point_query_year, :]
    logger.debug(f'Values at window {val_at_query_windows}')

    # Find the windows that are stable at the query year.
    # This shoudl be == 0, as we are looking for when they 
    # are not stable (e.g unstable)
    window_args = np.where(val_at_query_windows != 0)[0]
    logger.debug(f' - windows that are unstable\n{windows[window_args]}')
    logger.debug(f' - window_args\n{window_args}')   

    # Extract rows for the time window
    row_extraction = np.take(time_window_arr, range(point_query_year, point_query_year + period_length), axis=0)
    logger.debug(f'row_extraction ({row_extraction.shape})\n{row_extraction}')
    
    # Extract specific columns using np.take() with window_args
    column_row_extraction = np.take(row_extraction, window_args, axis=1)
    logger.debug(f'column_row_extraction ({column_row_extraction.shape})\n{column_row_extraction}')

    last_arg_array_sum = np.sum(column_row_extraction, axis=1)
    logger.debug(f'last_arg_array_sum ({last_arg_array_sum.shape})\n{last_arg_array_sum}')

    last_arg = np.where(last_arg_array_sum!=0)[0][-1]
    logger.debug(f'{last_arg=}')


    year_val = first_year_condition_met + last_arg + 2
    logger.info(f'{year_val=}')
    if i == 0:
        year_val = year_val + first_unstable_location

    return year_val
    return year_val


def calcuate_year_stable_and_unstable_v2(time_window_arr, frac_unstable_arr, windows, period_length:int=10, number_attempts: int = 7, 
                                      max_val:int=50, logginglevel='ERROR'):
    """
    Calculate the years of stability and instability in a time series.
    
    Parameters:
    - time_window_arr (np.ndarray): A 2D array representing the time series data for different windows.
    - windows (np.ndarray): An array containing the size of each window.
    - number_attempts (int): The maximum number of iterations for searching stability/instability. Default is 5.
    - period_length (int): The length of that stabilility has to occur for before the climate is
                             consdiered stable.
    
    Returns:
    - year_list (np.ndarray): A cumulative sum array representing the years at which stability or instability is identified.
      If the number of identified years is less than the number of attempts, the remaining positions in the array will be NaN.


    Notes on window subtraction
    - Looking for stability.
        - THis should occur one after the last time there is a one

    - Looking for instability
        - This should occur the first time there is a 1
    """
    # Convert the time series data to a binary array where 1 represents
    # finite values and 0 represents NaNs or infinite values.
    utils.change_logging_level(logginglevel)
    assert time_window_arr.shape[-1] == len(windows), (
    f"Assertion failed: The last dimension of time_window_arr has length {time_window_arr.shape[-1]} "
    f"but the nubmer of windows is {len(windows)}. The -1 dimensions shoudl be window. Data may need to be transpoed.")


    # This is not used functinoally, but used defensivly
    window_shape = time_window_arr.shape[-1]

    # Number of attempts is needed later
    number_attemps_2 = number_attempts
    
    # time_window_arr = np.where(np.isfinite(time_window_arr), 1, 0)
    
    next_year_list = []

    # Calculate the fraction of unstable points within each window.
    # time x window
    logger.debug(f'{period_length=}')
    # frac_unstable_arr = frac_non_zero_window(time_window_arr, period_length, None)
    logger.info(f'Shapes - time_window_arr = {time_window_arr.shape}, frac_unstable_arr = {frac_unstable_arr.shape}')
    # Using 5 years here, as if the very last of the 10 years has an unstable fraction
    
    initial_fracs = frac_unstable_arr[:int(np.ceil(period_length/2)), :]
    # print('difa')
    logger.info(f'initial_fracs shape {initial_fracs.shape}')
    with np.printoptions(threshold=np.inf):
        logger.debug(f' - inital_fracs (shape = {initial_fracs.shape})\n{initial_fracs}')

    if np.any(initial_fracs):
        logger.info('Fracs above threshold found - test for stability')
        # If any of the fraction are greater than 0.5, then we are unstable 
        # and need to start looking for stability
        TEST_FOR_STABILITY = True
        TEST_FOR_INSTABILITY = False
    else:
        logger.info('Fracs not found - test for instability')

        # Otherwise, we have started off with being stable
        # so start testing for instability
        TEST_FOR_STABILITY = False
        TEST_FOR_INSTABILITY = True
        # Append stable year 0 to array
        next_year_list.append(0)
        # We have already used one attempt
        number_attempts = number_attempts - 1
        
    i = 0
    while number_attempts >= 0:
        if i != 0:
            # Check if the end of the time series has been reached.
            if next_year_list[-1] >= time_window_arr.shape[0]: break

            
            if next_year_list[-1] == 0: selection = 0
            else: selection = next_year_list[-1]-1

            # The first time we are doing this, we don't want to subtract as this
            # has not occured year
            if next_year_list[-1] > 0: next_year_list[-1] = next_year_list[-1]-1

            # Subset the two datasets
            time_window_arr = time_window_arr[selection:, :]
            frac_unstable_arr = frac_unstable_arr[selection:, :]
    
            # Break if the remaining time series is too short for further analysis.
            if time_window_arr.shape[0] < (period_length+5): break
  
        if TEST_FOR_INSTABILITY: 
            year_val = search_for_instability_v2(time_window_arr, frac_unstable_arr, windows, period_length, logginglevel)
            logger.debug(f' - {year_val=}')
            TEST_FOR_STABILITY = True
            TEST_FOR_INSTABILITY = False
    
        elif TEST_FOR_STABILITY: 
            year_val = search_for_stability_v2(time_window_arr, frac_unstable_arr, windows,period_length, i, logginglevel)
            logger.debug(f' - {year_val=}')
            TEST_FOR_STABILITY = False
            TEST_FOR_INSTABILITY = True
        
        

        if np.isnan(year_val): break
        # If getting close to max value => break
        if len(next_year_list)>1:
            if np.cumsum(next_year_list)[-1] > max_val-5: break

        i+=1
        number_attempts = number_attempts - 1

        if year_val == 1: year_val = 0
        next_year_list.append(year_val)

        logger.info(f' - next_year_list\n{next_year_list}')
        logger.info(f'- Currunt cumsum\n{np.cumsum(next_year_list)}')
        # If the condition analysed is more than 45, break
        if np.cumsum(next_year_list)[-1] > max_val-5: break
        logger.info('Complete\n')


    logger.info(f'\n - Search complete - final processing')
    # Calculate the cumulative sum of the years to get the year list.
    logger.info(f' = nex_year_list final form\n{next_year_list}')
    year_list = np.cumsum(next_year_list)
    logger.info(f' = year_list (cumsum applied )\n{year_list}')
    # Due to the need to select one year before the year of change
    # this introduced small ERRORs that accumulate
    # position 1 will be too large by one year
    # position 2 will be too large by 2 years
    # etc.
    if len(year_list) > 1:
        for num in range(len(year_list)):
            if num%2:
                year_list[num] = year_list[num]+1 #- num

    year_list = remove_periods_below_threshold(year_list, int(np.ceil(period_length/2)))
    
    # End point conditions
    # # If there are any values greater than 50, replace them with 50
    if np.any(year_list>(max_val)):
        year_list[np.where(year_list>(max_val))] = max_val
        logger.debug(f'Value greater than max ({max_val=}) found and removed \n{year_list}')

    # IF the last is an unstable, but it has become unstalbe just beofre the end
    # then get rid of
    # if len(year_list)%2 and len(year_list)>1:
    #     if year_list[-1] > max_val-3:
    #         # Thus, remove the last value
    #         year_list = year_list[:-1]
        
    
    # # If not even (e.g. 1, 3) then finishing on instability
    # Need to assign unstable
    if not len(year_list)%2 and len(year_list)>1: 
        # Instability has been assigned at the last year
        # So chop of the last point
        if year_list[-1] > max_val-3:
            year_list = year_list[:-1]
        else:
            year_list = np.concatenate([year_list,[max_val]])

    # The model nevera actuyall stabilised, so assign 50
    if len(year_list) == 0:
        year_list = np.array([max_val])
        
    # Check how many 50s there are
    number_of_max_vals = len(np.where(year_list==max_val)[0])
    # IF there is more than one 50, remove
    if number_of_max_vals > 1:
        year_list = year_list[:-(number_of_max_vals-1)]

    
    # Ensure the year list has the same size as the number of attempts.
    # I is upwards counts of number of attempts
    if len(year_list) < number_attemps_2:
        year_list = np.concatenate([year_list, np.tile(np.nan, number_attemps_2 - len(year_list))])

    if len(year_list) != number_attemps_2:
        logger.info(f' - year_list length {len(year_list)} {number_attemps_2=}')
        logger.info(f' - year_list\n{year_list}')
        logger.info('\n')

    
    logger.info(year_list.shape)
    logger.info(f'final result {year_list}')
    return year_list















# def remove_periods_below_threshold(year_vals, threshold=10):
#     """
#     Remove values from the array where the difference between consecutive elements
#     in stable periods is below a given threshold. Stable periods are defined as the
#     segments between changes in differences. The function removes the values associated
#     with stable periods that have lengths below the threshold.

#     Parameters:
#     year_vals (list or np.array): The input list or array of values to be filtered.
#     threshold (int, optional): The minimum allowed length of stable periods. Default is 10.

#     Returns:
#     np.array: The filtered array with stable periods below the threshold removed.
#     """
#     # Convert the input to a numpy array for easier manipulation
#     year_vals = np.array(year_vals)

#     # Compute the differences between consecutive values
#     lengths = np.diff(year_vals)

#     # Extract the stable period lengths (even indexed differences)
#     period_lengths = lengths[::2]

#     # Identify stable periods below the threshold
#     length_below_threshold = period_lengths < threshold

#     # Determine indices to remove based on the stable periods
#     small_period_args = np.where(length_below_threshold)[0]
#     to_remove = np.sort(np.concatenate([small_period_args*2, small_period_args*2+1]))

#     # Remove the identified indices from the array
#     year_vals = np.delete(year_vals, to_remove)

#     return year_vals


# # def remove_closest_pairs_below_threshold(year_list, threshold=10):
# #     """
# #     Remove pairs of consecutive values from a list where the difference between the values is below a given threshold.
# #     The function removes the two values with the smallest difference iteratively until all remaining differences 
# #     between consecutive values are equal to or above the threshold.

# #     Parameters:
# #     year_list (list or np.array): The input list or array of years to be filtered.
# #     threshold (int, optional): The minimum allowed difference between consecutive values. Default is 10.

# #     Returns:
# #     np.array: The filtered array with all differences between consecutive values equal to or above the threshold.
# #     """
# #     # Convert the input to a numpy array for easier manipulation
# #     year_list = np.array(year_list)

# #     # Calculate differences between consecutive values
# #     diffs = np.diff(year_list)

# #     # Iterate while there are any differences below the threshold
# #     while np.any(diffs < threshold):
# #         # Find the index of the smallest difference
# #         min_diff_arg = np.argmin(diffs)

# #         # Remove the two values with the smallest difference
# #         # First, remove the element at min_diff_arg + 1
# #         year_list = np.delete(year_list, min_diff_arg + 1)
# #         # Then, remove the element at min_diff_arg (which has shifted one position left)
# #         year_list = np.delete(year_list, min_diff_arg)

# #         # Recalculate differences after removal
# #         diffs = np.diff(year_list)

# #     return year_list

# def calcuate_year_stable_and_unstable(time_window_arr, windows, period_length:int=10, number_attempts: int = 7, 
#                                       max_val:int=50, logginglevel='ERROR'):
#     """
#     Calculate the years of stability and instability in a time series.
    
#     Parameters:
#     - time_window_arr (np.ndarray): A 2D array representing the time series data for different windows.
#     - windows (np.ndarray): An array containing the size of each window.
#     - number_attempts (int): The maximum number of iterations for searching stability/instability. Default is 5.
#     - period_length (int): The length of that stabilility has to occur for before the climate is
#                              consdiered stable.
    
#     Returns:
#     - year_list (np.ndarray): A cumulative sum array representing the years at which stability or instability is identified.
#       If the number of identified years is less than the number of attempts, the remaining positions in the array will be NaN.


#     Notes on window subtraction
#     - Looking for stability.
#         - THis should occur one after the last time there is a one

#     - Looking for instability
#         - This should occur the first time there is a 1
#     """
#     # Convert the time series data to a binary array where 1 represents
#     # finite values and 0 represents NaNs or infinite values.
#     utils.change_logging_level(logginglevel)
#     assert time_window_arr.shape[-1] == len(windows), (
#     f"Assertion failed: The last dimension of time_window_arr has length {time_window_arr.shape[-1]} "
#     f"but the nubmer of windows is {len(windows)}. The -1 dimensions shoudl be window. Data may need to be transpoed.")


#     # This is not used functinoally, but used defensivly
#     window_shape = time_window_arr.shape[-1]

#     # Number of attempts is needed later
#     number_attemps_2 = number_attempts
    
#     time_window_arr = np.where(np.isfinite(time_window_arr), 1, 0)
    
#     bump_start=0
#     next_year_list = []

#     # Calculate the fraction of unstable points within each window.
#     # time x window
#     logger.debug(f'{period_length=}')
#     frac_unstable_arr = frac_non_zero_window(time_window_arr, period_length, None)
#     logger.info(f'Shapes - time_window_arr = {time_window_arr.shape}, frac_unstable_arr = {frac_unstable_arr.shape}')
#     # Using 5 years here, as if the very last of the 10 years has an unstable fraction
    
#     initial_fracs = frac_unstable_arr[:period_length, :]
#     logger.info(f'initial_fracs shape {initial_fracs.shape}')
#     with np.printoptions(threshold=np.inf):
#         logger.debug(f' - inital_fracs (shape = {initial_fracs.shape})\n{initial_fracs}')

#     if np.any(initial_fracs >= 0.5):
#         logger.info('Fracs above 0.5 found - test for stability')
#         # If any of the fraction are greater than 0.5, then we are unstable 
#         # and need to start looking for stability
#         TEST_FOR_STABILITY = True
#         TEST_FOR_INSTABILITY = False
#     else:
#         logger.info('Fracs above 0.5 not found - test for instability')

#         # Otherwise, we have started off with being stable
#         # so start testing for instability
#         TEST_FOR_STABILITY = False
#         TEST_FOR_INSTABILITY = True
#         # Append stable year 0 to array
#         next_year_list.append(0)
#         # We have already used one attempt
#         number_attempts = number_attempts - 1
        


#     i = 0
#     while number_attempts >= 0:
#         if i != 0:
#             # Check if the end of the time series has been reached.
#             if next_year_list[-1] >= time_window_arr.shape[0]: break
    
#             # Cut the time series starting from the last identified year to the end.
#             # The instability last at least as long as the window the fraction is taken
#             # over
#             #There are issues that can occur if the number of points
#             # are all in a row, the stable year and the unstalbe year,
#             # then both become the same. Thus, putting the negative one 
#             # here fixes this

#             # This selection condition results in values being too large.
#             if next_year_list[-1] == 0: selection = 0
#                 # number_required_for_sub = 3
#             else: selection = next_year_list[-1]-1
#                 # number_required_for_sub = 2
#             # if selection<0: selection = 0

#             # The first time we are doing this, we don't want to subtract as this
#             # has not occured year
#             if next_year_list[-1] > 0:
#                 # and TEST_FOR_INSTABILITY:  and len(next_year_list)>=2 # len(next_year_list) >=
#                 # number_required_for_sub: 
#                 next_year_list[-1] = next_year_list[-1]-1
#             # if selection > 1: selection = selection -1
#             time_window_arr = time_window_arr[selection:, :]
#             frac_unstable_arr = frac_unstable_arr[selection:, :]
    
#             # Break if the remaining time series is too short for further analysis.
#             if time_window_arr.shape[0] < (period_length+5): break
#         # if len(next_year_list)>=2: # At lesat two entries
#         #     # There has only been least three years since the last change
#         #     if next_year_list[-1] <= period_length: # Only ten years since last condition (changed from 3)
#         # #         # Erase last two values - the are basically ontop
#         # #         # e.g. the first condition didn't really occur
#         # #         if bump_start > 0:
#         # #             logger.info(f'Bump start currently {bump_start=}')
#         # #         bump_start += np.nansum(next_year_list[-2:])
#         # #         logger.info(f'Bump start now {bump_start=}')
#         # #         logger.info(f'Current next_year_list\n{next_year_list}')
#         # #         next_year_list = next_year_list[:-2]
#         # #         logger.info(f'Values erased next_year_list\n {next_year_list}')
#         # #         bump_start = 0
#         #         number_attempts += 2
#         #     else: bump_start=0
    
#         # Calculate the fraction of unstable points within each window.
#         # frac_unstable_arr = frac_non_zero_window(time_window_arr, period_length, logginglevel)
    
#         if TEST_FOR_INSTABILITY:  # Searching for instability
#             logger.info('\nInstability Search\n------\n')
    
#             # Set a threshold where instability is defined as the fraction of unstable points > 0.4.
#             instability_condition = frac_unstable_arr >= 0.5
#             logger.debug(f' - Instability_condition shape: {instability_condition.shape}')
    
#             # Create a binary array where 1 indicates instability.
#             frac_unstable_threshold_arr = np.where(instability_condition, 1, 0)
    
#             # If no instability is detected, break the loop.
#             if np.all(frac_unstable_threshold_arr == 0): break
    
#             # Count the number of unstable points across all windows.
#             number_unstable_across_window = np.nansum(frac_unstable_threshold_arr, axis=1)
#             logger.debug(f' - number_unstable_across_window\n{number_unstable_across_window}')
    
#             # Find the first year where instability is detected.
#             first_year_condition_met = np.where(number_unstable_across_window > 0)[0][0]
#             logger.debug(f' - {first_year_condition_met=}')
    
#             window_args = np.where(frac_unstable_threshold_arr[first_year_condition_met, :]==1)[0]
#             logger.debug(f' - {window_args=}')
#             logger.debug(f' - windows that are unstable\n{windows[window_args]}')

    
#             first_arg_list = []
#             for sarg in window_args:
#                 # length_of_selection = period_length#10#np.min([10, int(window)])
#                 # Select the window size for analysis
#                 window = windows[sarg]
#                 logger.debug(f' - {sarg=} {window=}')
#                 anlsysis_window =\
#                     time_window_arr[:, sarg]
#                     # time_window_arr[first_year_condition_met:first_year_condition_met + period_length+1, sarg]

#                 logger.debug(f' - anlsysis_window {anlsysis_window.shape}\n{anlsysis_window}')
#                 first_unstable_point = np.where(anlsysis_window==1)[0][0]
#                 first_arg_list.append(first_unstable_point)
#             logger.debug(f' - first_arg_list\n{first_arg_list}')
#             first_arg_list = np.array(first_arg_list)
#             year_addition = np.nanmin(first_arg_list)
#             logger.debug(f' - {year_addition=}')
    
#             year_val = first_year_condition_met+year_addition+1
#             logger.debug(f' - {year_val=}')
#             # If the condition analysed is more than 45, break
#             if np.cumsum(next_year_list)[-1] > max_val-5: break

#             # Instability found, start looking for instability
#             TEST_FOR_STABILITY = True
#             TEST_FOR_INSTABILITY = False
#             # number_attempts = number_attempts - 1
#             # i += 1
    
#         elif TEST_FOR_STABILITY:  # Searching for stability
#             logger.info('\nStability Search\n----------\n')
#             # Set a threshold where stability is defined as the fraction of unstable points <= 0.2.
#             # Shape time x window  
#             frac_stable_time_window = np.where(frac_unstable_arr < 0.5, 1, 0)
    
#             # Count the number of unstable points across all windows.
#             # Shape time
#             number_windows_stable = np.nansum(frac_stable_time_window, axis=1)
#             logger.debug(f'number_windows_stable:\n{number_windows_stable}')
    
#             # If every window is stable
#             # Shape: time
#             where_all_windows_stable = number_windows_stable == len(windows)
#             logger.debug(f'The locations where all windows are stable \n{where_all_windows_stable}')

    
#             # if i == 0:
#                 # During the first instance of this, there is a special conditino
#                 # This results by the stability occuring immediately at the start
#                 # followed by instabiliyt.
#                 # This results in the stability occuring early
#                 # Therefore we only want to start searching for stability
#                 # once the initial instability has occured
#             if i == 0:
#                 logger.debug('')
#                 first_unstable_location = np.where(where_all_windows_stable==0)[0][0]
#                 logger.debug(f'  - first_unstable_location\n{first_unstable_location}')
    
#                 where_all_windows_stable = where_all_windows_stable[first_unstable_location:]
#                 frac_stable_time_window = frac_stable_time_window[first_unstable_location:]
#                 time_window_arr = time_window_arr[first_unstable_location:]

#             # print(where_all_windows_stable)
#             # If stability is not found, break the loop.
#             if np.all(where_all_windows_stable == False): break
    
#             # Find the first year where stability is detected.
#             first_year_condition_met = np.where(where_all_windows_stable)[0][0]
    
#             #Identify the year before the first fully stable year.
#             point_query_year = first_year_condition_met - 1
#             logger.debug(f' - {point_query_year=}')
#             assert point_query_year >= 0 , ('The point query year is less than 0')

    
#             # Extract values from the time series at the query year.
#             # val_at_query_windows = time_window_arr[point_query_year, :]
#             val_at_query_windows = frac_stable_time_window[point_query_year, :]
#             logger.debug(f'Values at window {val_at_query_windows}')
    
#             # Find the windows that are stable at the query year.
#             # This shoudl be == 0, as we are looking for when they 
#             # are not stable (e.g unstable)
#             window_args = np.where(val_at_query_windows == 0)[0]
#             logger.debug(f' - windows that are unstable\n{windows[window_args]}')
#             logger.debug(f' - window_args\n{window_args}')
            
    
#             larst_arg_list = []
#             # print(window_args)
#             for sarg in window_args:
#                 # Get the window size for analysis.
#                 window = windows[sarg]
    
#                 # Determine the length of the selection window, max of 10 or the window size.
#                 # length_of_selection = 10#np.min([10, int(window)])
#                 # length_of_selection = period_length
    
#                 # print(window)
#                 # Select the analysis window starting from the query year.
#                 analysis_window = time_window_arr[point_query_year:point_query_year + period_length, sarg]
    
#                 # Find the last stable index in the analysis window.
#                 last_arg = get_last_arg_v2(analysis_window)
#                 larst_arg_list.append(last_arg)

#             # Calculate the additional years needed for stability.
#             larst_arg_list = np.array(larst_arg_list)
#             year_addition = np.max(larst_arg_list)
#             year_val = first_year_condition_met + year_addition + 1
#             if i == 0:
#                 year_val = year_val + first_unstable_location

#             # Year of stabilisation found, start looking for other
#             TEST_FOR_STABILITY = False
#             TEST_FOR_INSTABILITY = True
#         number_attempts = number_attempts - 1
        
#         i += 1
  
    
#         # Append the calculated year to the list.
#         if bump_start:
#             logger.debug(f' - {bump_start=}')
#             year_val=year_val + bump_start
#             # remove the bump start after it has been used
#             bump_start = 0

#         # Append
#         if year_val == 1: year_val = 0
#         next_year_list.append(year_val)

#         logger.info(f' - next_year_list\n{next_year_list}')
#         logger.info(f'- Currunt cumsum\n{np.cumsum(next_year_list)}')
#         # If the condition analysed is more than 45, break
#         if np.cumsum(next_year_list)[-1] > max_val-5: break
#         logger.info('Complete\n')


#     logger.info(f'\n - Search complete - final processing')
#     # Calculate the cumulative sum of the years to get the year list.
#     logger.info(f' = nex_year_list final form\n{next_year_list}')
#     year_list = np.cumsum(next_year_list)
#     logger.info(f' = year_list (cumsum applied )\n{year_list}')
#     # Due to the need to select one year before the year of change
#     # this introduced small ERRORs that accumulate
#     # position 1 will be too large by one year
#     # position 2 will be too large by 2 years
#     # etc.
#     if len(year_list) > 1:
#         for num in range(len(year_list)):
#             if num%2:
#                 year_list[num] = year_list[num]+1 #- num


#     # Remove values that are too close together
#     # year_list = remove_close_values(year_list, period_length)
#     # year_list = remove_closest_pairs_below_threshold(year_list, int(period_length/2)-2)
#     year_list = remove_periods_below_threshold(year_list, int(period_length/2))
#     # 
#     # End point conditions
#     # # If there are any values greater than 50, replace them with 50
#     if np.any(year_list>(max_val)):
#         year_list[np.where(year_list>(max_val))] = max_val
#         logger.debug(f'Value greater than max ({max_val=}) found and removed \n{year_list}')

#     # IF the last is an unstable, but it has become unstalbe just beofre the end
#     # then get rid of
#     # if len(year_list)%2 and len(year_list)>1:
#     #     if year_list[-1] > max_val-3:
#     #         # Thus, remove the last value
#     #         year_list = year_list[:-1]
        
    
#     # # If not even (e.g. 1, 3) then finishing on instability
#     # Need to assign unstable
#     if not len(year_list)%2 and len(year_list)>1: 
#         # Instability has been assigned at the last year
#         # So chop of the last point
#         if year_list[-1] > max_val-3:
#             year_list = year_list[:-1]
#         else:
#             year_list = np.concatenate([year_list,[max_val]])

#     # The model nevera actuyall stabilised, so assign 50
#     if len(year_list) == 0:
#         year_list = np.array([max_val])
        
#     # Check how many 50s there are
#     number_of_max_vals = len(np.where(year_list==max_val)[0])
#     # IF there is more than one 50, remove
#     if number_of_max_vals > 1:
#         year_list = year_list[:-(number_of_max_vals-1)]

    
#     # Ensure the year list has the same size as the number of attempts.
#     # I is upwards counts of number of attempts
#     if len(year_list) < number_attemps_2:
#         year_list = np.concatenate([year_list, np.tile(np.nan, number_attemps_2 - len(year_list))])

#     if len(year_list) != number_attemps_2:
#         logger.info(f' - year_list length {len(year_list)} {number_attemps_2=}')
#         logger.info(f' - year_list\n{year_list}')
#         logger.info('\n')

    
#     logger.info(year_list.shape)
#     logger.info(f'final result {year_list}')
#     return year_list




def generate_covered_amount_dict():
    """
    Generate a dictionary mapping window sizes to the number of initial points to cover.

    This function calculates the number of initial years (points) that should be set 
    to NaN (covered) for various window lengths. Windows shorter than 10 years 
    are removed (i.e., no years are covered for such windows).
    Anything that has a half window lenght shorter than 10 year shoudl be deleted.
    The 10-year window at year 5, will have a complete winodw lenth
    19: 1;
    18: 1 (at the year 1, the centered window will have 9 on either side  - this 10 in)
    17 - 2 (at the year 2, the center window will have 10 points inside)
    16 - 2 (at the year 2, the centered window will have have 8 either side - thus 10 in )
    Returns:
        dict: A dictionary where keys are window lengths (10 to 19) and values 
        are the number of initial years to cover.
    """
    cover_amount_dict = {}
    # Define the range of window sizes from 19 down to 10
    start_issue_windows = np.arange(19, 9, -1)
    for window in start_issue_windows:
        # Calculate the number of initial points to cover for each window
        cover_amount = int(np.ceil((20 - window) / 2)) + 1
        cover_amount_dict[window] = cover_amount
    return cover_amount_dict


cover_amount_dict = generate_covered_amount_dict()

def remove_bad_start_values_andrew_version(arr, window, logginglevel='ERROR'):
    """
    Remove initial values from an array based on the window size.

    This function sets the initial values of an array to NaN based on a specified 
    window size. The number of initial values set to NaN is half the window size, 
    rounded up.

    Args:
        arr (np.ndarray): The input array from which initial values will be removed.
        window (int): The window size used to determine the number of initial values to remove.

    Returns:
        np.ndarray: The modified array with initial values set to NaN.
    """

    utils.change_logginglevel(logginglevel)
    arr2 = arr.copy()
    # Calculate the number of initial points to cover (set to NaN)
    cover_amount = int(np.ceil(window / 2)) - 1
    logger.info(f'{cover_amount=}')
    arr2[:cover_amount] = np.nan

    # Need to also remove the bad values at the end
    # Reverse and apply same code
    arr2 = arr2[::-1]

    # If there is another timeserie that has been merged with
    # there could already be trailing nans
    finite_indices = np.where(np.isfinite(arr2))[0]
    if finite_indices.size > 0:
        nan_end_length = finite_indices[0]
        logger.info(f'{nan_end_length=}')
        arr2[:nan_end_length+cover_amount] = np.nan
    else:
        logger.warning('All elements in arr2 are NaN. No operation performed.')
    
    arr2 = arr2[::-1]
    
    return arr2


def remove_bad_start_values(arr, window):
    """
    Remove initial values from an array based on the window size using a predefined dictionary.

    This function uses a predefined dictionary (`cover_amount_dict`) to determine the number 
    of initial values to set to NaN based on the window size.

    Args:
        arr (np.ndarray): The input array from which initial values will be removed.
        window (int): The window size used to determine the number of initial values to remove.

    Returns:
        np.ndarray: The modified array with initial values set to NaN.
    """
    arr2 = arr.copy()
    # Check if the window size is in the dictionary and get the cover amount
    if window in list(cover_amount_dict):
        cover_amount = cover_amount_dict.get(window, 0)
        arr2[:cover_amount] = np.nan
    return arr2


# def calcuate_year_stable_and_unstable(time_window_arr, windows, number_attempts: int = 7, max_val:int=50,
#                                      logginglevel='ERROR'):
#     """
#     Calculate the years of stability and instability in a time series.
    
#     Parameters:
#     - time_window_arr (np.ndarray): A 2D array representing the time series data for different windows.
#     - windows (np.ndarray): An array containing the size of each window.
#     - number_attempts (int): The maximum number of iterations for searching stability/instability. Default is 5.
    
#     Returns:
#     - year_list (np.ndarray): A cumulative sum array representing the years at which stability or instability is identified.
#       If the number of identified years is less than the number of attempts, the remaining positions in the array will be NaN.
#     """
#     # Convert the time series data to a binary array where 1 represents
#     # finite values and 0 represents NaNs or infinite values.
#     utils.change_logging_level(logginglevel)
#     time_window_arr = np.where(np.isfinite(time_window_arr), 1, 0)
    
#     bump_start=0
#     next_year_list = []
#     for i in range(number_attempts):
    
#         if i != 0:
#             # Check if the end of the time series has been reached.
#             if next_year_list[-1] >= time_window_arr.shape[0]: break
    
#             # Cut the time series starting from the last identified year to the end.
#             # The instability last at least as long as the window the fraction is taken
#             # over
#             time_window_arr = time_window_arr[next_year_list[-1]:, :]
    
#             # Break if the remaining time series is too short for further analysis.
#             if time_window_arr.shape[0] < 10:
#                 break
#         if len(next_year_list)>2: # At lesat two entries
#             # There has only been least three years since the last change
#             if next_year_list[-1] <= 3: # Only three years since last condition
#                 # Erase last two values - the are basically ontop
#                 # e.g. the first condition didn't really occur
#                 bump_start = np.sum(next_year_list[:-2])#next_year_list[-1]
#                 next_year_list = next_year_list[:-2]
#             else: bump_start=0
    
#         # Calculate the fraction of unstable points within each window.
#         frac_unstable_arr = frac_non_zero_window(time_window_arr, windows)
    
#         if i % 2:  # Searching for instability
#             # print('Instability Search')
    
#             # Set a threshold where instability is defined as the fraction of unstable points > 0.4.
#             instability_condition = frac_unstable_arr >= 0.5
    
#             # Create a binary array where 1 indicates instability.
#             frac_unstable_threshold_arr = np.where(instability_condition, 1, 0)
    
#             # If no instability is detected, break the loop.
#             if np.all(frac_unstable_threshold_arr == 0): break
    
#             # Count the number of unstable points across all windows.
#             number_unstable_across_window = np.nansum(frac_unstable_threshold_arr, axis=1)
    
#             # Find the first year where instability is detected.
#             first_year_condition_met = np.where(number_unstable_across_window > 0)[0][0]
    
#             window_args = np.where(frac_unstable_threshold_arr[first_year_condition_met, :]==1)[0]
    
#             first_arg_list = []
#             for sarg in window_args:
#                 length_of_selection = 10#np.min([10, int(window)])
#                 # Select the window size for analysis
#                 window = windows[sarg]
#                 anlsysis_window =\
#                     time_window_arr[first_year_condition_met:first_year_condition_met + length_of_selection, sarg]
            
#                 first_unstable_point = np.where(anlsysis_window==1)[0][0]
#                 first_arg_list.append(first_unstable_point)
#             first_arg_list = np.array(first_arg_list)
#             year_addition = np.min(first_arg_list)
    
#             year_val = first_year_condition_met+year_addition+1
#             # If the condition analysed is more than 45, break
#             if np.cumsum(next_year_list)[-1] > max_val-5: break
    
      
#         else:  # Searching for stability
#             # print('Stability Search')
#             if i == 0:
#                 # First lets check if it is already ultra stable
#                 frac_below_hard_threshold = np.where(frac_unstable_arr[:10, :] <= 0.25, 1, 0)

#             # IF all fraction are below 0.2, stabilise immediately
#             if np.all(frac_below_hard_threshold == 1) and i==0:
#                     year_val = 0
#             else:
#                 # Set a threshold where stability is defined as the fraction of unstable points <= 0.2.
#                 # Shape time x window  
#                 frac_below_threshold = np.where(frac_unstable_arr < 0.5, 1, 0)
        
#                 # Count the number of unstable points across all windows.
#                 # Shape time
#                 number_windows_stable = np.nansum(frac_below_threshold, axis=1)
#                 logger.debug(f'number_windows_stable:\n{number_windows_stable}')
        
#                 # If every window is stable
#                 # Shape: time
#                 stability_condition = number_windows_stable == len(windows)
    
#                 logger.debug(f'stability condition array\n{stability_condition}')
#                 # print(stability_condition)
#                 # If stability is not found, break the loop.
#                 if np.all(stability_condition == False): break
        
#                 # Find the first year where stability is detected.
#                 first_year_condition_met = np.where(stability_condition)[0][0]
        
#                 #Identify the year before the first fully stable year.
#                 point_query_year = first_year_condition_met# - 1
#                 # if point_query_year <0: point_query_year +=1
#                 logger.info(f'{point_query_year}')
        
#                 # Extract values from the time series at the query year.
#                 # val_at_query_windows = time_window_arr[point_query_year, :]
#                 val_at_query_windows = frac_below_threshold[point_query_year, :]
#                 logger.debug(f'Values at window {val_at_query_windows}')
        
#                 # Find the windows that are stable at the query year.
#                 window_args = np.where(val_at_query_windows == 1)[0]
                
        
#                 larst_arg_list = []
#                 # print(window_args)
#                 for sarg in window_args:
#                     # Get the window size for analysis.
#                     window = windows[sarg]
        
#                     # Determine the length of the selection window, max of 10 or the window size.
#                     length_of_selection = 10#np.min([10, int(window)])
        
#                     # print(window)
#                     # Select the analysis window starting from the query year.
#                     analysis_window = time_window_arr[point_query_year:point_query_year + length_of_selection, sarg]
        
#                     # Find the last stable index in the analysis window.
#                     last_arg = get_last_arg_v2(analysis_window)
#                     larst_arg_list.append(last_arg)
    
#                 # Calculate the additional years needed for stability.
#                 larst_arg_list = np.array(larst_arg_list)
#                 year_addition = np.max(larst_arg_list)
#                 year_val = first_year_condition_met + year_addition

            
    
#         # Append the calculated year to the list.
#         if bump_start:
#             # print(year_val, bump_start)
#             year_val=year_val + bump_start

#         # Append
#         if year_val == 1: year_val = 0
#         next_year_list.append(year_val)

#         # If the condition analysed is more than 45, break
#         if np.cumsum(next_year_list)[-1] > max_val-5: break


#     # Calculate the cumulative sum of the years to get the year list.
#     year_list = np.cumsum(next_year_list)


#     # End point conditions
#     # # If there are any values greater than 50, replace them with 50
#     if np.any(year_list>max_val):
#         year_list[np.where(year_list>max_val)] = max_val
        
    
#     # # If not even (e.g. 1, 3) then finishing on instability
#     # Need to assign unstable
#     if not len(year_list)%2 and len(year_list)>1: 
#         # Instability has been assigned at the last year
#         # So chop of the last point
#         if year_list[-1] == max_val:
#             year_list = year_list[:-1]
#         else:
#             year_list = np.concatenate([year_list,[max_val]])

#     # The model nevera actuyall stabilised, so assign 50
#     if len(year_list) == 0:
#         year_list = np.array([max_val])
        
#     # Check how many 50s there are
#     number_of_max_vals = len(np.where(year_list==max_val)[0])
#     # IF there is more than one 50, remove
#     if number_of_max_vals > 1:
#         year_list = year_list[:-(number_of_max_vals-1)]

    
#     # Ensure the year list has the same size as the number of attempts.
#     if len(year_list) < number_attempts:
#         year_list = np.concatenate([year_list, np.tile(np.nan, number_attempts - len(year_list))])
#     logger.info('\n')
#     return year_list

#####!!!!!!!!!!!!!!
##### DO NOT DELETE
##### These are old algorithms to determine the stability of the climate. 
##### They are (should not) be used in notebooks, but are worth keep
#####!!!!!!!!!!!!!!


# def get_year_stable(arr:ArrayLike, window:int=None, time:Optional[ArrayLike]=None, stable_length:int=None, 
#                     logginglevel='ERROR'
#                     ) -> int:
#     """
#     This function calculates the year when stability occurs.

#     Parameters:
#     arr (ArrayLike): Input array to check for stability.
#     stable_length (int, optional): The minimum length of stability. Defaults to 20.
#     time (ArrayLike, optional): Time array corresponding to arr. Defaults to None.

#     Returns:
#     int: The year when stability occurs. Returns np.nan if no stability is found.

#     Example:
#     >>> arr = [1, 2, 3, np.nan, np.nan, np.nan, np.nan, 4, 5, 6, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7, 8, 9]
#     >>> get_year_stable(arr)
#     3
#     """
#     utils.change_logginglevel(logginglevel)
#     if stable_length is None:stable_length = int(window/2)

#     # Create a mask where arr is finite (not NaN or infinity)
#     condition = np.where(np.isfinite(arr), True, False)
#     # Group the mask by consecutive values
#     condition_groupby = []
#     for key, group in itertools.groupby(condition):
#         condition_groupby.append((key, len(list(group))))

#     logger.debug(condition_groupby)
#     # Find the indices where stability occurs (i.e., where the condition is False and the length is greater than stable_length)
#     condition_groupby_stable_arg = [arg for arg, (key, length) in enumerate(condition_groupby) 
#                                    if not key and length > stable_length]
#     logger.debug(condition_groupby_stable_arg)

#     # If stability is found, get the index of the first occurrence
#     if len(condition_groupby_stable_arg) > 0: 
#         condition_groupby_stable_arg = condition_groupby_stable_arg[0]
#     # There has not been a length long enough, but the last window, is a stable period
#     elif len(condition_groupby_stable_arg) == 0 and condition_groupby[-1][0] == False: 
#         condition_groupby_stable_arg = len(condition_groupby)-1
#     else: return np.nan  # Return NaN if no stability is found

    
#     # Calculate the year when stability occurs
#     stable_arg = np.sum(list(map(lambda x:x[-1], condition_groupby[:condition_groupby_stable_arg])))
    
#     #np.nansum([length for key, length in condition_groupby[:condition_groupby_stable_arg]])

    

#     if np.isnan(stable_arg): return stable_arg
#     stable_arg = int(stable_arg) if isinstance(stable_arg, float) else stable_arg
#     if time is not None: return time[stable_arg]

#     return stable_arg


# def convert_arr_to_groupby_condition(condition):
#     """
#     Convert the condition array to a grouped condition array.

#     Parameters:
#     condition (ArrayLike): The input condition array.

#     Returns:
#     list: A list of tuples containing the condition and the length of each group.
#     """
    
#     # Create a mask where arr is finite (not NaN or infinity)
    
#     # Group the mask by consecutive values
#     condition_groupby = []
#     for key, group in itertools.groupby(condition):
#         condition_groupby.append((key, len(list(group))))
#     return condition_groupby


# def extract_arg(condition_groupby, stable_length):
#     """
#     Extract the index of the first stable group from the grouped condition array.

#     Parameters:
#     condition_groupby (list): The grouped condition array.
#     stable_length (int): The minimum length for a stable group.

#     Returns:
#     int: The index of the first stable group. Returns NaN if no stability is found.
#     """
    
#     condition_groupby_stable_arg = [arg for arg, (key, length) in enumerate(condition_groupby) 
#                                    if not key and length > stable_length]
#     # If stability is found, get the index of the first occurrence
#     if len(condition_groupby_stable_arg) > 0: 
#         condition_groupby_stable_arg = condition_groupby_stable_arg[0]
#     # There has not been a length long enough, but the last window, is a stable period
#     elif len(condition_groupby_stable_arg) == 0 and condition_groupby[-1][0] == False: 
#         condition_groupby_stable_arg = len(condition_groupby)-1
#     else: 
#         return np.nan  # Return NaN if no stability is found
    
#     # Calculate the year when stability occurs
#     stable_arg = np.sum(list(map(lambda x:x[-1], condition_groupby[:condition_groupby_stable_arg])))
    
#     return stable_arg

# def get_year_stable_v2(arr:ArrayLike, window:int=None, time:Optional[ArrayLike]=None) -> int:
#     """
#     This function calculates the index of the first year with stable data.
#     Note: This is different from get_stable_year, as there is now a condition on
#     how long everything must be stable for.

#     Parameters:
#     arr (ArrayLike): The input array.
#     window (int): The window size for stability calculation. Default is None.
#     time (Optional[ArrayLike]): The time array. Default is None.

#     Returns:
#     int: The index of the first year with stable data. If time is provided, returns the corresponding time value.
#     Example:
#     >>> arr = [1, 2, 3, np.nan, np.nan, np.nan, np.nan, 4, 5, 6, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7, 8, 9]
#     >>> get_year_stable(arr)
#     3
#     """
    
#     # Calculate the stable length based on the window size
#     stable_length = int(window/4)
#     if stable_length < 10: 
#         stable_length = 10  # Minimum stable length is 10
    
#     # Create a condition array where finite values in arr are True and non-finite are False
#     condition = np.where(np.isfinite(arr), True, False)
    
#     # Initialize a flag for stability
#     found_stability = False
    
#     # Loop until stability is found
#     while not found_stability:
#         # Group the condition array and extract the index of the first stable group
#         condition_groupby = convert_arr_to_groupby_condition(condition)
#         stable_arg = extract_arg(condition_groupby, stable_length)
        
#         # If the first stable group is at index 0, return 0
#         if stable_arg == 0 or np.isnan(stable_arg): 
#             return stable_arg
        
#         # Calculate the start of the sample period
#         start_of_sample = stable_arg - 10
#         start_of_sample = start_of_sample if start_of_sample > 0 else 0
        
#         # Count the number of stable points in the sample period
#         #print(start_of_sample,stable_arg)
#         total_stable_in_period = np.count_nonzero(condition[start_of_sample:stable_arg])
        
#         # If there are at least 5 stable points, mark as stable
#         if total_stable_in_period >= 5:
#             found_stability = True
#         else:
#             # If not enough points for stability, mark the period as unstable
#             condition[start_of_sample:stable_arg] = False
    
#     # If time is provided, return the corresponding time value
#     if time is not None: 
#         return time[stable_arg]
    
#     # Otherwise, return the index of the first year with stable data
#     return stable_arg
