import os, sys
import dask
import itertools
import numpy as np
import xarray as xr
from typing import Optional
from numpy.typing import ArrayLike

# Custom Module Imports
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'zecmip_stabilisation_drafts'))
import constants
sys.path.append(constants.MODULE_DIR)
import utils
import xarray_extender as xe
logger = utils.get_notebook_logger()

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
    This is a more efficeitn for calculting the gradient than inbuilt python methods

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
    center: bool = True, 
    method: str = 'gradient', 
    logginglevel: str = 'ERROR'
) -> xr.DataArray:
    """
    Calculate a rolling signal in a dataset based on a specified method.

    Args:
    - data (xr.DataArray): Input dataset.
    - window (int, optional): Rolling window size. Defaults to 20.
    - min_periods (int, optional): Minimum number of periods. Defaults to 0.
    - center (bool, optional): Whether to center the rolling window. Defaults to True.
    - method (str, optional): Calculation method. Defaults to 'gradient'.
    - logginglevel (str, optional): Logging level. Defaults to 'ERROR'.

    Returns:
    - xr.DataArray: The calculated rolling signal.
    """
    utils.change_logging_level(logginglevel)
    logger.info(f"Calculating the rolling signal with method {method}")

    if min_periods == 0:
        min_periods = window

    logger.debug(f"{window=}, {min_periods=}\ndata=\n{data}")

    # New x values
    xs = np.arange(window)
    # Mean of the x-values
    mean_xs = np.nanmean(xs)
    # Denominator can actually always be the same
    denominator = np.mean(xs) ** 2 - np.mean(xs ** 2)
    signal_da = (data
                 .rolling(time=window, min_periods=min_periods, center=center)
                 .reduce(grid_gradient, xs=xs, mean_xs=mean_xs, denominator=denominator
                        )) 
    # Multiply by window length to get signal from gradient
    signal_da = signal_da* window
    if center:
        signal_da = adjust_time_from_rolling(signal_da, window, logginglevel)
    else:
        signal_da = signal_da.dropna(dim='time')


    signal_da.name = 'signal'
    signal_da = signal_da.expand_dims('window').assign_coords(window=('window', [window]))
    return signal_da


def rolling_noise(data, window:int, min_periods = 0,center=True,logginglevel='ERROR') -> xr.DataArray:
    
    utils.change_logging_level(logginglevel)

    logger.info("Calculting the rolling noise")

    
    # If no min_periods, then min_periods is just roll_period.
    if ~min_periods:min_periods = window
    
    # Rolling standard deviation
    noise_da = \
       data.rolling(time = window, min_periods = min_periods, center = True).std()

    if center == True:
        noise_da = adjust_time_from_rolling(noise_da, window=window, logginglevel=logginglevel)
    else:
        noise_da = noise_da.dropna(dim='time')
    
    noise_da.name = 'noise'
    
    noise_da = noise_da.expand_dims('window').assign_coords(window=('window', [window]))
    
    return noise_da



def signal_to_noise_ratio(
    ds: xr.Dataset, 
    window:int,
    detrended_data: xr.Dataset = None, 
    return_all: bool = False
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
    signal_ds = rolling_signal(ds, window)  # Use the rolling_signal function to calculate the signal

    # Calculate the rolling noise
    # If detrended data is provided, use it; otherwise, use the original dataset
    noise_ds = rolling_noise(ds if detrended_data is None else detrended_data, window)  

    # Calculate the signal-to-noise ratio
    sn_ratio_ds = signal_ds / noise_ds  # Divide the signal by the noise to get the ratio

    # Return the desired datasets
    if return_all:
        return signal_ds, noise_ds, sn_ratio_ds  # Return all datasets if requested
    return sn_ratio_ds  # Otherwise, return only the signal-to-noise ratio dataset


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
    qlower = kwargs.pop('qlower', 5)
    qupper = kwargs.pop('qupper', 95)

    # Calculate signal-to-noise ratio
    sn_ratio_ds = signal_to_noise_ratio(ds=ds, window=window, **kwargs)
    sn_ratio_ds = sn_ratio_ds.compute()

    # Choose percentile function based on whether data is chunked
    percentile_func = xe.dask_percentile if sn_ratio_ds.chunks else np.nanpercentile

    # Calculate upper and lower bounds of SNR using specified percentiles
    sn_ratio_ub_ds = sn_ratio_ds.reduce(percentile_func, q=qupper, dim='time').compute()
    sn_ratio_lb_ds = sn_ratio_ds.reduce(percentile_func, q=qlower, dim='time').compute()

    # Merge upper and lower bounds into a single dataset
    sn_ratio_bounds_ds = xr.merge(
        [
            sn_ratio_lb_ds.to_dataset(name='lower_bound'),
            sn_ratio_ub_ds.to_dataset(name='upper_bound')
        ], 
        compat='override'
    ).compute()

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
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        
        # Initialize list to store results
        to_concat = []
        
        # Loop over each window size
        for window in windows:
            # logger.info(window)
            print(window, end='')
            # Calculate signal-to-noise ratio bounds for current window
            to_concat.append(signal_to_noise_ratio_bounds(ds, window, **kwargs))
        
        # Concatenate results along a new dimension named 'window'
        outpout_ds = xr.concat(to_concat, dim='window')
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
    func = dask.delayed(func) if parallel else signal_to_noise_ratio
    
  
    # Initialize list to store results
    to_concat = []
    
    # Loop over each window size
    for window in windows:
        logger.info(window)
        # Calculate signal-to-noise ratio for current window
        output_ds = func(ds, window, **kwargs)
        to_concat.append(output_ds)

    # Compute the dask object (for some reason make it list of list)
    if parallel: 
        to_concat = dask.compute(*to_concat)
    # Concatenate results along a new dimension named 'window' and compute
    result_ds = xr.concat(to_concat, dim='window').compute()
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
    logginglevel = kwargs.pop('logginglevel', 'ERROR')
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
    return output_ds


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

def get_year_stable(arr:ArrayLike, window:int=None, time:Optional[ArrayLike]=None, stable_length:int=None, 
                    logginglevel='ERROR'
                    ) -> int:
    """
    This function calculates the year when stability occurs.

    Parameters:
    arr (ArrayLike): Input array to check for stability.
    stable_length (int, optional): The minimum length of stability. Defaults to 20.
    time (ArrayLike, optional): Time array corresponding to arr. Defaults to None.

    Returns:
    int: The year when stability occurs. Returns np.nan if no stability is found.

    Example:
    >>> arr = [1, 2, 3, np.nan, np.nan, np.nan, np.nan, 4, 5, 6, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7, 8, 9]
    >>> get_year_stable(arr)
    3
    """
    utils.change_logginglevel(logginglevel)
    if stable_length is None:stable_length = int(window/2)

    # Create a mask where arr is finite (not NaN or infinity)
    condition = np.where(np.isfinite(arr), True, False)
    # Group the mask by consecutive values
    condition_groupby = []
    for key, group in itertools.groupby(condition):
        condition_groupby.append((key, len(list(group))))

    logger.debug(condition_groupby)
    # Find the indices where stability occurs (i.e., where the condition is False and the length is greater than stable_length)
    condition_groupby_stable_arg = [arg for arg, (key, length) in enumerate(condition_groupby) 
                                   if not key and length > stable_length]
    logger.debug(condition_groupby_stable_arg)

    # If stability is found, get the index of the first occurrence
    if len(condition_groupby_stable_arg) > 0: 
        condition_groupby_stable_arg = condition_groupby_stable_arg[0]
    # There has not been a length long enough, but the last window, is a stable period
    elif len(condition_groupby_stable_arg) == 0 and condition_groupby[-1][0] == False: 
        condition_groupby_stable_arg = len(condition_groupby)-1
    else: return np.nan  # Return NaN if no stability is found

    
    # Calculate the year when stability occurs
    stable_arg = np.sum(list(map(lambda x:x[-1], condition_groupby[:condition_groupby_stable_arg])))
    
    #np.nansum([length for key, length in condition_groupby[:condition_groupby_stable_arg]])

    

    if np.isnan(stable_arg): return stable_arg
    stable_arg = int(stable_arg) if isinstance(stable_arg, float) else stable_arg
    if time is not None: return time[stable_arg]

    return stable_arg


def convert_arr_to_groupby_condition(condition):
    """
    Convert the condition array to a grouped condition array.

    Parameters:
    condition (ArrayLike): The input condition array.

    Returns:
    list: A list of tuples containing the condition and the length of each group.
    """
    
    # Create a mask where arr is finite (not NaN or infinity)
    
    # Group the mask by consecutive values
    condition_groupby = []
    for key, group in itertools.groupby(condition):
        condition_groupby.append((key, len(list(group))))
    return condition_groupby


def extract_arg(condition_groupby, stable_length):
    """
    Extract the index of the first stable group from the grouped condition array.

    Parameters:
    condition_groupby (list): The grouped condition array.
    stable_length (int): The minimum length for a stable group.

    Returns:
    int: The index of the first stable group. Returns NaN if no stability is found.
    """
    
    condition_groupby_stable_arg = [arg for arg, (key, length) in enumerate(condition_groupby) 
                                   if not key and length > stable_length]
    # If stability is found, get the index of the first occurrence
    if len(condition_groupby_stable_arg) > 0: 
        condition_groupby_stable_arg = condition_groupby_stable_arg[0]
    # There has not been a length long enough, but the last window, is a stable period
    elif len(condition_groupby_stable_arg) == 0 and condition_groupby[-1][0] == False: 
        condition_groupby_stable_arg = len(condition_groupby)-1
    else: 
        return np.nan  # Return NaN if no stability is found
    
    # Calculate the year when stability occurs
    stable_arg = np.sum(list(map(lambda x:x[-1], condition_groupby[:condition_groupby_stable_arg])))
    
    return stable_arg

def get_year_stable_v2(arr:ArrayLike, window:int=None, time:Optional[ArrayLike]=None) -> int:
    """
    This function calculates the index of the first year with stable data.
    Note: This is different from get_stable_year, as there is now a condition on
    how long everything must be stable for.

    Parameters:
    arr (ArrayLike): The input array.
    window (int): The window size for stability calculation. Default is None.
    time (Optional[ArrayLike]): The time array. Default is None.

    Returns:
    int: The index of the first year with stable data. If time is provided, returns the corresponding time value.
    Example:
    >>> arr = [1, 2, 3, np.nan, np.nan, np.nan, np.nan, 4, 5, 6, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 7, 8, 9]
    >>> get_year_stable(arr)
    3
    """
    
    # Calculate the stable length based on the window size
    stable_length = int(window/4)
    if stable_length < 10: 
        stable_length = 10  # Minimum stable length is 10
    
    # Create a condition array where finite values in arr are True and non-finite are False
    condition = np.where(np.isfinite(arr), True, False)
    
    # Initialize a flag for stability
    found_stability = False
    
    # Loop until stability is found
    while not found_stability:
        # Group the condition array and extract the index of the first stable group
        condition_groupby = convert_arr_to_groupby_condition(condition)
        stable_arg = extract_arg(condition_groupby, stable_length)
        
        # If the first stable group is at index 0, return 0
        if stable_arg == 0 or np.isnan(stable_arg): 
            return stable_arg
        
        # Calculate the start of the sample period
        start_of_sample = stable_arg - 10
        start_of_sample = start_of_sample if start_of_sample > 0 else 0
        
        # Count the number of stable points in the sample period
        #print(start_of_sample,stable_arg)
        total_stable_in_period = np.count_nonzero(condition[start_of_sample:stable_arg])
        
        # If there are at least 5 stable points, mark as stable
        if total_stable_in_period >= 5:
            found_stability = True
        else:
            # If not enough points for stability, mark the period as unstable
            condition[start_of_sample:stable_arg] = False
    
    # If time is provided, return the corresponding time value
    if time is not None: 
        return time[stable_arg]
    
    # Otherwise, return the index of the first year with stable data
    return stable_arg