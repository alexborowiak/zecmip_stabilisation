# Standard library imports
import sys
import logging
from typing import Dict, List, Union, Optional, Callable, Tuple

# Third-party library imports
import numpy as np
import xarray as xr
import cftime


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib import ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


ArrayLike = List[float]

# Local imports (if applicable)
import utils
# logging.basicConfig(format="%(message)s", filemode='w', stream = sys.stdout)
# logger = logging.getLogger()

logger = utils.get_notebook_logger()

sys.path.append('../')
# import constants
import plotting_functions
from listXarray import listXarray

from classes import ObsoleteFunctionError

# NOTE: this comes from zecmip_stability_global_(single_ensemble).ipynb
ZECMIP_MODEL_PARAMS = {'GISS-E2-1-G-CC': {'value': -0.09, 'color': '#add8e6', 'linestyle': '-'},
 'CanESM5': {'value': -0.101, 'color': '#87ceeb', 'linestyle': '--'},
 'MIROC-ES2L': {'value': -0.108, 'color': '#6495ed', 'linestyle': '-'},
 'GFDL-ESM4': {'value': -0.204, 'color': '#4169e1', 'linestyle': '--'},
 'MPI-ESM1-2-LR': {'value': -0.27, 'color': '#1e90ff', 'linestyle': '-'},
 'CESM2': {'value': -0.31, 'color': '#0000cd', 'linestyle':'--'},
 'NorESM2-LM': {'value': -0.333, 'color': '#00008b','linestyle': '-'},
 'ACCESS-ESM1-5': {'value': 0.011,
  'color': [0.99137255, 0.6972549 , 0.48392157, 1.], 'linestyle': '--'},
 'UKESM1-0-LL': {'value': 0.288,
  'color': [0.78666667, 0.11294118, 0.07294118, 1.], 'linestyle': '-'}}


# # Usually use a red cmap, so making sure the lines are not red.



REGION_STYLE_DICT = {
    'land': {'color': 'brown', 'linestyle': '--'}, 
    'ocean': {'color': 'blue', 'linestyle': '--'},
    'nh': {'color': 'm', 'linestyle': (0, (1, 1))}, 
    'sh': {'color': 'orange', 'linestyle': (0, (1, 1))},
    'tropics': {'color': 'darkgreen', 'linestyle': '--'}, 
    'mid_nh': {'color': 'red', 'linestyle': '-'}, 
    'mid_sh': {'color': 'tomato', 'linestyle': '--'},
    'arctic': {'color': 'lightblue', 'linestyle': '-'}, 
    'antarctic': {'color': 'darkblue'},
    'global_warm': {'color': 'darkred'},
    'global_cool': {'color': 'darkblue'},
    'gl': {'color': 'black', 'linestyle': '-', 'linewidth': 5, 'zorder': 100}
}


NO_RED_COLORS = ('k', 'green','m', 'mediumpurple', 'black',
                 'lightgreen','lightblue', 'greenyellow')



MODEL_PROFILES = { 'zecmip': ZECMIP_MODEL_PARAMS, 'region': REGION_STYLE_DICT}


colors = [(1, 1, 1, 0), (0, 0, 0, 1)]  # RGBA format: (red, green, blue, alpha)
black_white_cmap = mcolors.ListedColormap(colors)




plot_kwargs = dict(height=12, width=22, hspace=0.3, #vmin=-8, vmax=8, step=2, 
                   cmap = 'RdBu_r', line_color = 'limegreen', line_alpha=0.65, 
                   ax2_ylabel = 'Anomaly', cbar_label = 'Signal-to-Noise', cbartick_offset=0,
                   axes_title='', 
                   title='', label_size=12, extend='both', xlowerlim=None, xupperlim=None,  filter_max=True,)


colors = [(0, 0, 0, 0), (0.5, 0.5, 0.5, 0.5)]  # (R, G, B, Alpha)
cmap_binary = mcolors.LinearSegmentedColormap.from_list("binary_no_color", colors)

def grey_mask(ax, da):
    da.plot(ax=ax, cmap=cmap_binary, add_colorbar=False)

def dict_to_title(d):
    lat_direction = 'N' if d['lat'] >= 0 else 'S'
    lon_direction = 'E' if d['lon'] >= 0 else 'W'
    return f"{abs(d['lat'])}°{lat_direction}, {abs(d['lon'])}°{lon_direction}"

def format_ticks_as_years(ax, xvalues, major_base:int=10, minor_base:int=5, logginglevel='ERROR'):

    utils.change_logginglevel(logginglevel)

    logger.info('Calling function format_ticks_as_years')
    logger.debug(f'{xvalues=}')
    if isinstance(xvalues[0] , cftime.datetime):
        xlabels = xvalues.dt.year.values-1 # The ticks neet to be set back one year so year 1 is year 0
    else:
        xlabels=xvalues

    logger.debug(f'{xlabels=}')
    
    # Set major ticks every 10 units
    logger.info(f'{major_base=}')
    major_locator = mticker.MultipleLocator(base=major_base)
    ax.xaxis.set_major_locator(major_locator)
        
    # Set minor ticks every 5 units
    logger.info(f'{minor_base=}')
    minor_locator = mticker.MultipleLocator(base=minor_base)
    ax.xaxis.set_minor_locator(minor_locator)

def format_xticks(ax, locations, labels, increment:int=10):
    ax.set_xticks(locations[::increment])
    ax.set_xticklabels(labels[::increment])

def plot_all_coord_lines(da: xr.DataArray, coord='model', exp_type=None,
                         fig=None, ax:plt.Axes=None, figsize:tuple=(15,7),
                         font_scale=1, consensus=True, xlabel=None, ylabel=None, yticks_right:list=None, labelpad=60,
                         bbox_to_anchor=(1.02,1), ncol=4, add_legend=True, xlim=None, ylim=None, title=None, 
                         increment:int=10, linestyle='-', colors:Union[str, Tuple, Dict]=None, logginglevel='ERROR',params=None,
                         linewidth=2,
                        **kwargs):
    '''
    Plots all of the values in time for a coordinate. E.g. will plot all of the models values
    in time for the global average or for a given grid cell.
    '''

    logger.debug(locals())
    utils.change_logging_level(logginglevel)
    
    fig = plt.figure(figsize=figsize) if not fig else fig
    ax = fig.add_subplot(111) if not ax else ax

    coord_values = da[coord].values.flatten() # Flatten in-case 0D array
    
    time = da.time.values

    # CFTIME
    if isinstance(time[0], cftime.datetime): time  = da.time.dt.year.values

    # I ahve created color profiles for all the models 
    if exp_type:
        logger.debug(f'{exp_type=}')
        params = MODEL_PROFILES[exp_type]
        coord_values = [cv for cv in list(params) if cv in coord_values]

    # Consensus needs to go first. So that it appears first in the legend
    if consensus and len(coord_values) > 1:
        ax.plot(time, da.median(dim=coord).values, 
                alpha=kwargs['line_alpha'] if 'line_alpha' in kwargs else 1,
                zorder=500, label='Median', linewidth=4,  
                c='black')
        
    for i, coord_value in enumerate(coord_values):
        logger.debug(f'{i=}, {coord_value=}')

        # linewidth = 2
        zorder  = 100
        if exp_type or params:
            logger.info('Using custom params')
            c = params[coord_value]['color']
            ls =  params[coord_value].get('linestyle', '-')
            # linewidth = params[coord_value].get('linewidth', 3)
            if 'zorder' in params[coord_value]: zorder = params[coord_value]['zorder']
            logger.debug(f'   - {c=}, {ls=}')
        else:
            c = NO_RED_COLORS[i]
            if isinstance(linestyle, str): ls=linestyle
            elif isinstance(linestyle, dict): ls = linestyle[coord_value]
            else: ls = linestyle[i]

        da_to_plot = da.loc[{coord:coord_value}].values if len(coord_values) > 1 else da.values

        ax.plot(time, da_to_plot,
                alpha=kwargs['line_alpha'] if 'line_alpha' in kwargs else 1,
                zorder=zorder, label=coord_value, linewidth=linewidth,  linestyle=ls, 
                c=c)

    ax.grid(True, linestyle='--', color='gray', alpha=0.2)

    
    if isinstance(xlim, tuple): ax.set_xlim(time[xlim[0]], time[xlim[-1]])
    if isinstance(ylim, tuple): ax.set_ylim(ylim)
    if yticks_right is not None: ax.set_yticks(yticks_right)
    if len(coord_values) > 1 and add_legend:
        leg = ax.legend(ncol=ncol, bbox_to_anchor=bbox_to_anchor,
                        fontsize=kwargs.get('legend_fontsize', 12)*font_scale)
        
        leg.set_title(coord.capitalize())
        leg.get_title().set_fontsize(plotting_functions.PlotConfig.legend_title_size*font_scale)
        
    plotting_functions.format_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title, labelpad=labelpad, xlabelpad=15,
                                               # invisible_spines=['top', 'right']
                                   font_scale=font_scale)

    format_ticks_as_years(ax, da.time)

    return (fig, ax) if 'leg' not in locals() else (fig, ax, leg)



def create_levels(vmax:float, vmin:float=None, step:float=1)->np.ndarray:
    '''
    Ensures that all instances of creating levels using vmax + step as the max.
    '''
    vmin = -vmax if vmin is None else vmin
    return np.arange(vmin, vmax + step, step)



def format_colorbar(pcolor, gs=None, cax:plt.Axes=None, tick_symbol:str='%'):
    '''
    Creates a colorbar that takes up all columns in the 0th row.
    The tick labels are percent
    
    Reason
    ------
    In 07_exploring_consecutive_metrics_all_models_(nb_none) a colorbar 
    of this type is used repeatedly. 
    '''
    cax = plt.subplot(gs[0,:]) if not cax else cax

    cbar = plt.colorbar(pcolor, cax=cax, orientation='horizontal')
    xticks = cbar.ax.get_xticks()
    cbar.ax.set_xticks(xticks)
    if tick_sybmol: cbar.ax.set_xticklabels([str(int(xt)) + tick_symbol for xt in xticks]);
    cbar.ax.tick_params(labelsize=labelsize)
    
    return cbar


def plot_heatmap(da:xr.DataArray, fig:plt.figure=None, gs=None, ax:plt.Axes=None, cax:plt.Axes=None,
                 figsize:tuple=None, cmap='Blues', extend='neither', max_color_lim:int=None,
                 levels:list=None, vmin=None, vmax=None, step=None, yticks:list=None,
                 xlims:tuple=None, font_scale=1, alpha:float=1, 
                 cbar_tile:str='', tick_labels=None, add_colorbar=True, cbar_label=None,
                 tick_offset=None, cut_ticks=1, patch=False, hspace=0,
                 title:str=None, axes_title:str=None, labelpad=100, rotation=0,
                 ylabel='Window Length\n(Years)', xlabel='Time After Emission Cessation (Years)', return_all=True,
                 logginglevel='ERROR', **kwargs):
    '''
    Plots a heatmatp of ds. Lots of options for entering different arguements
    '''
    
    utils.change_logginglevel(logginglevel)
    logger.info(utils.function_name())
    logger.info(f'**{__file__}')

    
    figsize = figsize if figsize is not None else (plot_kwargs['width'], plot_kwargs['height'])
    fig = fig if fig is not None else plt.figure(figsize=figsize)
    gs = (gs if gs is not None else gridspec.GridSpec(2,1, height_ratios=[1, 0.1], hspace=plot_kwargs['hspace']+hspace))
    
    ax = ax if ax is not None else fig.add_subplot(gs[0])
    
    if xlims is not None: da = da.isel(time=slice(*xlims))
    if not np.issubdtype(da.time.dtype, np.int64): da['time'] = da.time.dt.year.values
    if max_color_lim: da = da.isel(time=slice(None, max_color_lim))
    
    
    if levels is not None: colormap_kwargs = dict(levels=levels)
    elif vmax is not None and step is not None:
        levels = create_levels(vmin=vmin, vmax=vmax, step=step)
        colormap_kwargs = dict(levels=levels)
    else: colormap_kwargs = dict(robust=True)
    logger.info(f'{colormap_kwargs=}')
        
    # ----> Plotting the heatmaps
    cs = da.plot(ax=ax, cmap=cmap, extend=extend, add_colorbar=False, alpha=alpha, levels=levels) # **colormap_kwargs
    
    # ---> Labelling
    plotting_functions.format_axis(ax, xlabel=xlabel, ylabel=ylabel,labelpad=labelpad, font_scale=font_scale, rotation=rotation)
    format_ticks_as_years(ax, da.time)
    fig.suptitle(title, fontsize=plotting_functions.PlotConfig.title_size*font_scale, y=0.92)
    ax.set_title(axes_title, fontsize=plotting_functions.PlotConfig.title_size*font_scale)
    if xlims is not None: ax.set_xlim(xlims)
    if yticks is not None: ax.set_yticks(yticks)
 
    # ---> Artist
    if patch: ax.add_artist(Rectangle((max_color_lim, 0), xlims[-1]-max_color_lim, 200, color='grey', alpha=0.2, zorder=-1000))
    
    # ---> Colorbar
    if add_colorbar:
        cax = cax if cax is not None else fig.add_subplot(gs[1])
        cbar = plotting_functions.create_colorbar(
            cs, cax=cax, levels=levels, extend=extend, orientation='horizontal',
            font_scale=font_scale, cbar_title=cbar_label, tick_offset=tick_offset,
            cut_ticks=cut_ticks, logginglevel=logginglevel)
        
    if return_all: return fig, gs, ax, cax


def sn_multi_window_in_time(da:xr.DataArray, exp_type:str=None,
                            temp_da:Union[xr.DataArray, xr.Dataset]=None,
                            stable_point_ds:xr.Dataset=None,
                            fig:plt.figure=None, gs=None, ax:plt.Axes=None, cax:plt.Axes=None,
                            figsize:tuple=None, cmap='Blues', extend='neither', max_color_lim:int=None,
                            levels:list=None, vmin=None, vmax=None, step=None, 
                            xlims:tuple=(None,None), font_scale=1.5, yticks:list=None, yticks_right:list=None,
                            cbar_tile:str='', tick_labels=None, add_colorbar=True, cbar_label=None,
                            tick_offset=None, cut_ticks=1, patch=False, hspace=0,
                            title:str=None, axes_title:str=None,rotation=0, 
                            ylabel='Window Length\n(Years)', xlabel='Time After Emission Cessation (Years)',
                            ax2_ylabel = 'Anomaly', add_legend=True, labelpad_left=100, labelpad_right=50,
                            bbox_to_anchor=(1, 1.3), stable_year_kwargs=dict(),
                            logginglevel='ERROR', return_all=True):
    '''
    
    '''
    # mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use('default')
    utils.change_logging_level(logginglevel)
    logger.info(utils.function_name())
    logger.info(f'**{__file__}')
    
     
    # ---> Creating plot
    fig = plt.figure(figsize=(plot_kwargs['width'], plot_kwargs['height'])) if fig is None else fig
    gs = gridspec.GridSpec(2,1, height_ratios=[1, 0.1], hspace=plot_kwargs['hspace']) if gs is None else gs
    ax = fig.add_subplot(gs[0]) if ax is None else ax
    
    # ---> Stable Year 
    stable_year_kwargs = {'color':'k', 'linestyle':':', 'linewidth':3.5, **stable_year_kwargs}
    if stable_point_ds: stable_point_ds.time.plot(y='window', ax=ax, **stable_year_kwargs)

    # ---> Plotting colors
    fig, gs, ax, cax = plot_heatmap(da=da, fig=fig, gs=gs, ax=ax, cax=cax,
                 figsize=figsize, cmap=cmap, extend=extend, max_color_lim=max_color_lim,
                 levels=levels, vmin=vmin, vmax=vmax, step=step, 
                 xlims=xlims, font_scale=font_scale, yticks=yticks,
                 cbar_tile=cbar_tile, tick_labels=tick_labels, add_colorbar=add_colorbar, cbar_label=cbar_label,
                 tick_offset=tick_offset, cut_ticks=cut_ticks, patch=patch, hspace=hspace,
                 title=title, axes_title=axes_title, rotatiogn=rotation, 
                 ylabel=ylabel, xlabel=xlabel, labelpad=labelpad_left, logginglevel=logginglevel)

    # ---> Temperature Anomaly
    if isinstance(temp_da, xr.DataArray):
        ax2 = ax.twinx()
        temp_da = temp_da.isel(time=slice(*xlims))
        if not np.issubdtype(temp_da.time.dtype, np.int64): temp_da['time'] = temp_da.time.dt.year.values-1
        plot_all_coord_lines(da=temp_da, ax=ax2, fig=fig, exp_type=exp_type, add_legend=add_legend,
                             font_scale=font_scale, bbox_to_anchor=bbox_to_anchor, yticks_right=yticks_right)
        
        plotting_functions.format_axis(ax2, xlabel=xlabel, ylabel=ax2_ylabel, font_scale=font_scale, labelpad=labelpad_right, rotation=rotation)
        ax2.set_title(None)
        plotting_functions.match_ticks(ax, ax2, 'left') # Note: Has to be left
        
    if return_all:
        try: return (fig, [ax, ax2, cax])
        except NameError: return (fig, [ax, cax])







def map_plot_all_for_coords_3(da: xr.DataArray, levels: ArrayLike, dim: str=None, ncols:int = 3,
                              fig: Optional[plt.Figure] = None, axes: Optional[List[plt.Axes]] = None,
                              gs: Optional[gridspec.GridSpec] = None, cax: Optional[plt.Axes] = None,
                              add_colorbar: bool = True, projection:Callable = ccrs.Robinson,
                              cmap: str = 'RdBu_r', extend: str = 'both', font_scale:float = 1.4,
                              cbar_title: Optional[str] = None, return_all:bool = True, add_label: bool = True,
                              title_tag:str='', add_title=True, ptype='contourf',
                              debug: bool = False, max_stabilisation_year:Optional[int] = None,
                              stabilisation_method: Optional[str] = None, wspace=0.1, 
                              stipling_da: Optional[xr.DataArray] = None,
                              logginglevel='ERROR') -> Optional[Tuple[plt.Figure, List[plt.Axes], plt.Axes]]:
    '''
    Plots rows x columns of the data for the dim coordinate. This can take any projection and
    can also take a new figure or generate a figure.

    Parameters:
        da (xr.DataArray): The data array to plot.
        dim (str): The dimension along which to plot the data.
        levels (ArrayLike): The contour levels to use for plotting.
        ncols (int): Number of columns for the plot grid (default: 3).
        fig (plt.Figure, optional): The matplotlib figure object to use for the plot (default: None).
        axes (List[plt.Axes], optional): List of matplotlib axes objects to use for the subplots (default: None).
        gs (gridspec.GridSpec, optional): The gridspec object defining the layout of subplots (default: None).
        cax (plt.Axes, optional): The axes object to use for the colorbar (default: None).
        add_colorbar (bool): Whether to add a colorbar to the plot (default: True).
        projection (callable): The projection to use for the plot (default: ccrs.Robinson).
        cmap (str): The colormap to use for the plot (default: 'RdBu_r').
        extend (str): The colorbar extension option (default: 'both').
        font_scale (float): The scaling factor for the font sizes (default: 1.4).
        cbar_title (str, optional): The title for the colorbar (default: None).
        return_all (bool): Whether to return all created objects (default: True).
        add_label (bool): Whether to add labels to the subplots (default: True).
        debug (bool): Whether to print debug information (default: False).
        max_stabilisation_year (int, optional): The maximum stabilization year (default: None).
        stabilisation_method (str, optional): The method for stabilization (default: None).
        stipling_da (xr.DataArray, optional): The data array for stippling (default: None).

    Returns:
        If `return_all` is True, the function returns the created figure, axes, and colorbar objects. Otherwise, None.

    '''
    utils.change_logginglevel(logginglevel)
    # Calculate the central longitude based on the input data.
    # If the input data is an xarray DataArray or Dataset, use the mean of the longitude values.
    # Otherwise, if it's an object of type `listXarray`, use the mean of its `lon` values obtained from `single_xarray()`.
    central_longitude = int(np.mean(da.lon.values)) if isinstance(da, (xr.DataArray, xr.Dataset)) else int(np.mean(da.single_xarray().lon.values))

 
    # Print the type of the input data for debugging purposes.
    logger.info(f'Input type = {type(da)}')

    # Check the type of the input data and get the values of the specified dimension (`dim`) accordingly.
    if isinstance(da, (xr.DataArray, xr.Dataset)): 
        if dim is None: raise TypeError(f'When using xarray types dim must not be none ({dim=})')
        dim_vals = da[dim].values
    elif isinstance(da, listXarray): dim_vals = da.refkeys
    else: raise TypeError(f"Expected 'da' to be of type xr.DataArray, xr.Dataset, or listXarray. Got {type(da)} instead.")
    logger.debug(f' - {dim_vals=}')

    # Calculate the number of rows based on the number of dimension values and the specified number of columns (`ncols`).
    nrows = int(np.ceil(len(dim_vals)/ncols))
    logger.debug(f' - {nrows=}')

    # If `fig` is not provided, create a new figure with the specified size (default: 18*nrows x 10*ncols).
    # If `fig` is already provided, use the existing figure.
    fig = plt.figure(figsize=(18*nrows, 10*ncols)) if fig is None else fig

    # If `gs` (GridSpec) is not provided, create a new GridSpec with the specified number of rows and columns,
    # along with height ratios for each row.
    # If `gs` is already provided, use the existing GridSpec.
    gs = gridspec.GridSpec(nrows+1, ncols, height_ratios=[1]*nrows + [0.15], wspace=wspace) if gs is None else gs

    # Calculate the total number of plots.
    num_plots = ncols * nrows

    # Create a list of subplots (axes) based on the number of plots calculated above.
    # If `axes` is already provided, use the existing list of axes.
    axes = (
        [fig.add_subplot(
            gs[i], projection=projection(central_longitude=central_longitude)) for i in range(0, num_plots)]
            if axes is None else axes)

    # Initialize a flag to keep track of whether the colorbar has been added already.
    colobar_completed = False

    # Check if the levels for all plots are the same (a single set of levels) or different (individual levels for
    # each plot).
    logger.debug(f' - {levels=}')
    matching_levels_for_all = isinstance(levels[0], (int, float, np.int64, np.float32, np.float64))
    logger.debug(f' - {matching_levels_for_all=}')

    cbars = []
    # Loop through each dimension value and create contour plots for each one.
    logger.info('=>Starting plot loop\n')
    for num, dv in enumerate(dim_vals):
        logger.info(f'{num=} - {dv=}')
        ax = axes[num]
        levels_to_use = levels if matching_levels_for_all else levels[num]
        logger.debug(f' - {levels_to_use=}')

        # Extract the data corresponding to the current dimension value.
        if isinstance(da, (xr.DataArray, xr.Dataset)): da_to_use = da.loc[{dim: dv}]
        elif isinstance(da, listXarray): da_to_use = da[dv]
        
        da_to_use = da_to_use.squeeze()
        logger.debug(da_to_use)

        # Create a filled contour plot on the current subplot (ax).
        plot_kwargs = dict(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, levels=levels_to_use, extend=extend,
                           add_colorbar=False)
        if ptype == 'contourf': c = da_to_use.plot.contourf(**plot_kwargs)
        elif ptype == 'imshow': c = c = da_to_use.plot(**plot_kwargs)
        else: raise TypeError(f'ptype must be one of [contourf, imshow]. Entered {ptype=}')


        if max_stabilisation_year:
            da_binary = xr.where(da_to_use > max_stabilisation_year, 1, 0)
            if stabilisation_method == None:
                pass
            elif stabilisation_method == 'blackout':
                ax.contourf(da_binary.lon.values, da_binary.lat.values, da_binary.values,
                            transform=ccrs.PlateCarree(), cmap=black_white_cmap, levels=[0, 0.5, 1],
                            extend='neither')
            elif stabilisation_method == 'stipple':
                plot_stippled_data(da_binary, ax)
            else:
                raise ValueError(
                    f'stabilisation_method must be one of [blackout, stipple]. Value entered {stabilisation_method}')
        utils.change_logginglevel(logginglevel)
        # Optionally, if `stipling_da` is provided, overlay stipple data on the plot.
        # The details of the `plot_stippled_data` function are not provided here.
        if isinstance(stipling_da, xr.DataArray) or isinstance(stipling_da, listXarray):
            if isinstance(stipling_da, xr.DataArray): stippling_data_to_use = stipling_da.loc[{dim: dv}]
            elif isinstance(da, listXarray): stippling_data_to_use = stipling_da[dv]
            plot_stippled_data(stippling_data_to_use, ax, sig_size=.7, stiple_reduction=1, alpha=0.8)

        # Add coastlines to the plot.
        axes[num].coastlines()

        # Set the title for the current subplot with the corresponding dimension value (`dv`).
        if add_title: axes[num].set_title(f'{dv}{title_tag}',
                                          fontsize=plotting_functions.PlotConfig.title_size*font_scale)

        # Optionally, add a figure label (e.g., "a)", "b)", etc.) to each subplot if `add_label` is True.
        if add_label: plotting_functions.add_figure_label(axes[num], f'{chr(97+num)})', font_scale=font_scale)

        # If `add_colorbar` is True and the colorbar hasn't been added yet (for cases with individual levels),
        # create and add a colorbar to the plot.
        
        if add_colorbar and not colobar_completed:
            gs_to_use = gs[nrows, :] if matching_levels_for_all else gs[nrows, num]  # One cax, or multiple
            if isinstance(cax, plt.Axes):
                cax_to_use = cax
            else:
                cax_to_use = plt.subplot(gs_to_use) if cax is None else cax[num]
            logger.debug(f' - colorbar: {levels_to_use=}')
            cbar = plotting_functions.create_colorbar(c, cax=cax_to_use, levels=levels_to_use, extend=extend,
                                                      orientation='horizontal',
                            font_scale=font_scale*1.2, cbar_title=cbar_title)
            colobar_completed = True if matching_levels_for_all else False
            cbars.append(cbar)

    # If `return_all` is True, return the figure, GridSpec, and axes objects.
    if return_all:
        to_return = (fig, gs, axes, cbars) if 'cbar' in locals() else (fig, gs, axes)
        return to_return



def extract_partial_color_map(cmap, start:int, required_legnth, scaling:int):

    number_partitions = required_legnth * scaling
    color_list = plt.get_cmap(cmap)(np.linspace(0, 1, number_partitions))
    color_list = color_list[start:start+required_legnth]
    return color_list
def generte_sn_cmap_and_levels(step:float=1/3):
    # This is a complex colorbar creation.
    # There are three colormaps, and I want to select different parts of them.
    # This is as I want a different colormap for all of the different levels above.
    step=1/3
    
    extreme_lower_levels = np.arange(-3, -2, step)
    even_lower_levels = np.arange(-2, -1, step)
    lower_levels = np.arange(-1, -step, step) 
    uppper_levels = np.arange(-step, 1, step)
    even_upper_levels = np.arange(1, 2, step)
    extreme_upper_levels = np.arange(2,3.1, step)
    
     
    negative_levels  = np.concatenate([ extreme_lower_levels, even_lower_levels, lower_levels])
    postive_levels = np.concatenate([ uppper_levels, even_upper_levels, extreme_upper_levels])
    sn_levels =  np.concatenate([negative_levels, postive_levels])
    sn_levels = np.unique(np.sort(sn_levels.round(2)))

    color_extreme_lower_levels = extract_partial_color_map('cool_r',  0, len(extreme_lower_levels), 2)
    color_even_lower_levels = extract_partial_color_map('YlGnBu',  len(even_lower_levels), len(even_lower_levels)*2,3)
    color_lower_levels = extract_partial_color_map('Blues_r',  5, len(lower_levels), 3)
    color_upper_levels = extract_partial_color_map('YlOrBr',  3, len(uppper_levels), 4)
    color_even_upper_levels = extract_partial_color_map('YlOrBr',  len(even_upper_levels)+1, len(even_upper_levels), 3)
    color_extreme_upper_levels = extract_partial_color_map('Reds',  len(uppper_levels)*2, len(uppper_levels), 3)

    negative_colors = np.concatenate([color_extreme_lower_levels, color_even_lower_levels, color_lower_levels])
    print(negative_levels)
    mcolors.LinearSegmentedColormap.from_list("my_cmap",  negative_colors)

    postivie_colors= np.concatenate([color_upper_levels, color_even_upper_levels, color_extreme_upper_levels])
    print(postive_levels)
    
    mcolors.LinearSegmentedColormap.from_list("my_cmap",  postivie_colors)

    # Merging all the colours together
    full_colorlist = np.concatenate([negative_colors, postivie_colors])

    sn_cmap = mcolors.LinearSegmentedColormap.from_list("my_cmap",  full_colorlist)

    return sn_cmap, sn_levels



def plot_average_stable_year(ds1, ds2, fig=None, ax=None, font_scale:float=1,
                             ylabel='Window Length (years)', xlabel='Time (Years)'):
    ''''
    Plotting the median year of stabilisation at each window for two 
    different datasets.
    '''
    if not fig: fig = plt.figure(figsize=(10, 8))
    if not ax: ax = fig.add_subplot(111)

    ds1_mean = ds1.median(dim='model')
    ds2_mean = ds2.median(dim='model')
    ds1_mean.plot(ax=ax, y='window', label='rolling', color='k', linewidth=1.5, linestyle='solid') 
    ds2_mean.plot(ax=ax,y='window', label='static', color='k', linewidth=1.5, linestyle='dashed') 

    ylims = np.take(ds1_mean.window.values, [0,-1])
    all_x_values = np.concatenate([ds1_mean.time.values, ds2_mean.time.values]).flatten()
    xlims = [np.min(all_x_values), np.max(all_x_values)]
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    leg = ax.legend(ncol=1, loc='best', frameon=True, facecolor='white', fontsize=14) 
    leg.set_title('Noise Type')
    leg.get_title().set_fontsize('16')
    
    plotting_functions.format_axis(ax, xlabel=xlabel, ylabel=ylabel,font_scale=font_scale,
                                   invisible_spines=['top', 'right'])
    ax.set_title('')
    
    return fig, ax


    
def calculate_mean_values(years, values):
    """
    Calculate mean values for each stabilization period and store start-end year tuples.

    Parameters:
    - years: np.ndarray, Array of year values.
    - values: np.ndarray, Array of corresponding values.

    Returns:
    - value_mean_list: list of float, Mean values for each stabilization period.
    - start_end_tuple_list: list of tuples, Each tuple contains the start and end years for a stabilization period.
    """
    years = years[np.isfinite(years)]
    value_mean_list = []
    start_end_tuple_list = []
    
    # Iterate through each year
    for i in range(len(years)):
        # Check if the index corresponds to a stabilization year (0, 2, 4, ...)
        if 1 - i % 2:  # True for even indices
            
            # Define the start year of the stabilization period
            start = years[i]
            
            # Define the end year, which is either 20 years after the start or the beginning of the next period
            if i < len(years) - 1:
                end = start + np.nanmin([20, years[i+1] - start])
            else:
                end = start + 20

            start = int(start)
            end = int(end)

            # Select the values corresponding to the stabilization period
            value_selection = values[start:end]
            
            # Calculate the mean of the selected values and add it to the list
            value_mean = np.mean(value_selection)

            start_end_tuple_list.append((start, end))
            value_mean_list.append(value_mean)

    return value_mean_list, start_end_tuple_list


def process_stability_data(data_coords, model_name, anom_data, stability_data, year_stability_data):
    """
    Processes stability-related data for a given model and coordinates.

    Parameters:
    - data_coords: dict, Coordinates for latitude and longitude.
    - model_name: str, The name of the climate model.
    - anom_data: xarray.Dataset, The dataset containing temperature anomaly data.
    - stability_data: xarray.Dataset, The dataset containing stability pattern data.
    - year_stability_data: xarray.Dataset, The dataset containing year stability data.

    Returns:
    - sm_ds: xarray.DataArray, The temperature anomaly data for the selected model and coordinates.
    - sm_patt: xarray.DataArray, The stability pattern data for the selected model and coordinates.
    - year_stable_vals: np.ndarray, The stable year values, filtered for finite values.
    - t1_val: float, The mean temperature anomaly before the first stable year.
    - t2_val: float, The mean temperature anomaly after the second stable year.
    """
    # Extract temperature anomaly data based on coordinates and model
    sm_ds = anom_data[model_name].sel(**data_coords).squeeze() #.sel(time=slice(0, None))

    # Extract stability pattern data based on coordinates and model
    sm_patt = stability_data[model_name].sel(**data_coords)
    # sm_patt['time'] = sm_patt.time.dt.year.values

    # Extract and filter stable year values, converting to integer
    year_stable_vals = year_stability_data[model_name].sel(**data_coords).squeeze().values
    year_stable_vals = year_stable_vals[np.isfinite(year_stable_vals)].astype(int)

    # Calculate mean temperature anomalies for specific periods before and after stability
    # t1_val = sm_ds.isel(time=slice(year_stable_vals[1] - 20, year_stable_vals[1])).mean(dim='time').values.round(2).item()
    # t2_val = sm_ds.isel(time=slice(year_stable_vals[2], year_stable_vals[2] + 20)).mean(dim='time').values.round(2).item()
    value_mean_list, start_end_tuple_list = calculate_mean_values(year_stable_vals, sm_ds.values)

    return sm_ds, sm_patt, year_stable_vals, value_mean_list, start_end_tuple_list

def plot_stabilization(data_coords, model_name, anom_data, stability_data, year_stability_data):
    """
    Plots the stabilization and temperature anomaly data for a given model and coordinates.

    Parameters:
    - data_coords: dict, Coordinates for latitude and longitude.
    - model_name: str, The name of the climate model.
    - anom_data: xarray.Dataset, The dataset containing temperature anomaly data.
    - stability_data: xarray.Dataset, The dataset containing stability pattern data.
    - year_stability_data: xarray.Dataset, The dataset containing year stability data.

    Returns:
    - fig: matplotlib.figure.Figure, The figure object containing the plots.
    - ax1: matplotlib.axes.Axes, The axes object for the first subplot.
    - ax2: matplotlib.axes.Axes, The axes object for the second subplot.
    """
    # Process stability data to get relevant datasets and values
    sm_ds, sm_patt, year_stable_vals, value_mean_list, start_end_tuple_list = process_stability_data(
        data_coords, model_name, anom_data, stability_data, year_stability_data
    )

    sn_cmap, sn_levels = generte_sn_cmap_and_levels(1/3)
    sn_cmap = 'RdBu_r'
    
    # Calculate mean values and start-end year tuples
    value_mean_list, start_end_tuple_list = calculate_mean_values(year_stable_vals, sm_ds.values)
    
    fig = plt.figure(figsize=(9, 6))
    gs = gridspec.GridSpec(4, 1, height_ratios=[0.1, 0.15, 1, 0.5], hspace=0)
    ax1 = fig.add_subplot(gs[2])
    ax2 = fig.add_subplot(gs[3])

    # Plot the stability pattern on the first subplot
    c1 = sm_patt.plot(y='window', ax=ax1, alpha=0.8, levels=sn_levels, cmap=sn_cmap, add_colorbar=False, 
                      extend='both')

    # Add vertical lines for each stable year in both subplots
    for year in year_stable_vals:
        ax1.axvline(year, color='green')
        ax2.axvline(year, color='green')

    # Check if the time variable uses cftime or numerical values
    if isinstance(sm_ds.time.values[0], (int, float)):time = sm_ds.time.values
    else: time = sm_ds.time.dt.year.values
    # Plot the GMST anomaly and its 20-year rolling mean on the second subplot
    # xmin = np.nanmin(time)
    ax2.plot(time, sm_ds.values)
    # ax2.plot(time, sm_ds.rolling(time=20, center=True).mean().values, color='green', linewidth=3)

    # Add lines for temperature anomalies based on the new value lists
    counter = 0
    for (start, end), value_mean in zip(start_end_tuple_list, value_mean_list):
        ax2.plot([start, end], [value_mean, value_mean], color='magenta')
        if counter > 0:
            diff = value_mean_list[counter] - value_mean_list[counter-1]
            
            # yval = value_mean * 1.25 if value_mean >0 else value_mean*0.75
            yval = ax2.get_ylim()[0] * 1.25 if ax2.get_ylim()[0] >0 else ax2.get_ylim()[0]*0.75
            # ax2.annotate(f'{diff:.1f}' + r'$^\circ$C', xy=(start+int((end-start)/2)+1,yval) , size=12, color='magenta',
            #             ha='center', va='center')
            sign = '+' if diff > 0 else '-'
            ax2.annotate(f'{sign}{diff:.1f}' + r'$^\circ$C', xy=(start+int((end-start)/2)+1,yval) , size=12,
                         color='magenta', ha='center', va='center')
        counter += 1

    # Set up the x-axis of the first subplot, moving labels to the top
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()

    # Apply common settings to both subplots, like limits and grid
    # for ax in [ax1, ax2]:
    #     # ax.set_xlim(-1, 80)
    #     ax.set_xlim(xmin, 100)
    #     ax.grid(True, linestyle='--', alpha=0.6, color='grey')

    # Add a grey shaded region to the first subplot
    # ax1.axvspan(50, 80, color='grey')
    # ax1.axvspan(100-np.abs(xmin), 100, color='grey')

    # Label the axes of both subplots
    ax1.set_xlabel('')
    ax2.set_xlabel('Time After Emission Cessation (Years)')
    ax1.set_ylabel('Window\nLength\n(Years)')
    ax2.set_ylabel('GMST\nAnomaly' + r'($^\circ C$)')

    # Set the title for the first subplot, including model name and coordinates
    ax1.set_title(f'{model_name} - ({dict_to_title(data_coords)})')


    cax=fig.add_subplot(gs[0])

    cbar = plt.colorbar(c1, cax=cax, orientation='horizontal')
    # cbar.set_title('S/N Ratio')

    return fig, ax1, ax2
