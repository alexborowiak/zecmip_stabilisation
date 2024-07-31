import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
