import numpy as np
import pandas as pd
import os
import scipy.stats
from netCDF4 import Dataset
import random # use this when debugging

# NOTE: onSegment, orientation and doIntersect are from Vikas Chitturi
# A Python3 program to check if a given point
# lies inside a given polygon
# Refer https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
# for explanation of functions onSegment(),
# orientation() and doIntersect()

def onSegment(p, q, r) -> bool:
    """
    Given three colinear points p, q, r,
    the function checks if point q lies
    on line segment 'pr'
    """

    if ((q[0] <= max(p[0], r[0])) &
        (q[0] >= min(p[0], r[0])) &
        (q[1] <= max(p[1], r[1])) &
        (q[1] >= min(p[1], r[1]))):
        return True

    return False

def orientation(p, q, r) -> int:
    """
    To find orientation of ordered triplet (p, q, r).
    The function returns following values
    0 --> p, q and r are colinear
    1 --> Clockwise
    2 --> Counterclockwise
    """

    val = (((q[1] - p[1]) *
        (r[0] - q[0])) -
        ((q[0] - p[0]) *
        (r[1] - q[1])))

    if val == 0:
        return 0
    if val > 0:
        return 1 # Collinear
    else:
        return 2 # Clock or counterclock

def doIntersect(p1, q1, p2, q2):
    """
    Check for intersection
    """

    # Find the four orientations needed for
    # general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases
    # p1, q1 and p2 are colinear and
    # p2 lies on segment p1q1
    if (o1 == 0) and (onSegment(p1, p2, q1)):
        return True

    # p1, q1 and p2 are colinear and
    # q2 lies on segment p1q1
    if (o2 == 0) and (onSegment(p1, q2, q1)):
        return True

    # p2, q2 and p1 are colinear and
    # p1 lies on segment p2q2
    if (o3 == 0) and (onSegment(p2, p1, q2)):
        return True

    # p2, q2 and q1 are colinear and
    # q1 lies on segment p2q2
    if (o4 == 0) and (onSegment(p2, q1, q2)):
        return True

    return False

def is_inside_polygon(points, p, int_max = 500) -> bool:
    """
    Returns true if the point p lies
    inside the polygon[] with n vertices
    """

    n = len(points)

    # There must be at least 3 vertices
    # in polygon
    if n < 3:
        return False

    # Create a point for line segment
    # from p to infinite
    extreme = (int_max, p[1])
    count = i = 0

    while True:
        next = (i + 1) % n

        # Check if the line segment from 'p' to
        # 'extreme' intersects with the line
        # segment from 'polygon[i]' to 'polygon[next]'
        if (doIntersect(points[i], points[next], p, extreme)):

            # If the point 'p' is colinear with line
            # segment 'i-next', then check if it lies
            # on segment. If it lies, return true, otherwise false
            if orientation(points[i], p, points[next]) == 0:
                return onSegment(points[i], p, points[next])
            count += 1

        i = next

        if (i == 0):
            break

    # Return true if count is odd, false otherwise
    return (count % 2 == 1)

def make_grid(prof_dir, n_max, n_min, dn = 50, ds = 50, ncdf_fname = None):
    """
    For making customized grid
    prof_dir: directory containing profiles
    n_max: in km in direction of downgoing plate
    n_min: in km in direction of upper plate
    dn: step in the n-axis
    ds: step in the s-axis
    ncdf_fname: file name of the output ncdf file

    # Returns: n_arr, s_arr, lon_arr, lat_arr
    """

    n_arr = np.arange(n_min, n_max + dn, dn)

    for subdir, dirs, files in os.walk(prof_dir):

        s_num = len(files) # number of elements in along trench direction
        s_arr = np.arange(0, s_num)*ds
        lon_arr = np.zeros([n_arr.shape[0], s_num])
        lat_arr = np.zeros([n_arr.shape[0], s_num])

        for x in files:
            filepath = subdir + os.sep + x
            x = x.replace('.txt', '')
            x = x.replace('pr_-L0-', '') # use x as label
            x = int(x)

            # open file containing profile
            prof = pd.read_csv(filepath, sep = '\t', header = None)
            prof.columns = ['LON', 'LAT', 'DIST', 'VAL1', 'VAL2']

            # loop through n
            n_idx = 0
            for n in n_arr:
                diff = np.abs(n - prof['DIST'])
                idx = np.where(diff == diff.min())[0]
                sset = prof.iloc[idx]
                #lon_arr[n_idx, x] = sset['LON'] 
                #lat_arr[n_idx, x] = sset['LAT']

                # change above 2 lines due to deprecations
                lon_arr[n_idx, x] = float(sset['LON'].iloc[0])
                lat_arr[n_idx, x] = float(sset['LAT'].iloc[0])

                n_idx += 1

    #print(s_arr)
    if ncdf_fname is not None:
        # write into netcdf file if you want
        # close if it's open
        try:
            ncfile.close()
        except:
            pass

        ncfile = Dataset(ncdf_fname, mode = 'w', format = 'NETCDF4')

        s_dim = ncfile.createDimension('s_axis', s_arr.shape[0])
        n_dim = ncfile.createDimension('n_axis', n_arr.shape[0])

        #for dim in ncfile.dimensions.items():
        #    print(dim)

        ncfile.title = 'Gridding scheme'
        s_axis = ncfile.createVariable('s_axis', np.float32, ('s_axis'))
        s_axis.units = 'km'
        n_axis = ncfile.createVariable('n_axis', np.float32, ('n_axis'))
        n_axis.units = 'km'
        lon_grid = ncfile.createVariable('lon_grid', np.float32, ('n_axis', 's_axis'))
        lon_grid.units = 'degrees'
        lat_grid = ncfile.createVariable('lat_grid', np.float32, ('n_axis', 's_axis'))
        lat_grid.units = 'degrees'

        # fill in to variables
        # the square brackets are important
        s_axis[:] = s_arr
        n_axis[:] = n_arr
        lon_grid[:,:] = lon_arr
        lat_grid[:,:] = lat_arr

        #print(s_axis[:])
        #print(n_axis[:])
        #print(lon_grid[:,:])
        ncfile.close()
        print('Dataset writing done!')

    return n_arr, s_arr, lon_arr, lat_arr


def sample_on_grid(s_ax, n_ax, lon_grid, lat_grid, dep_df, feat_dict, eq_df_col = None, eq_df = None, mode = 0, debug = False):
    """
    Inputs:
    s_ax, n_ax, lon_grid, lat_grid,
    eq_df (if needed),
    dep_df (feature that dictates content presence: important to have LON and LAT as columns).
    Set to None if you want to bypass (e.g. map data inside grid but not ne slab data)
    feat_dict - dictionary containing features to sample,
    eq_df_col - columns of earthquake dataframe you want to save
    mode - if mode is set to 0 (square grid), LON_AVE and LAT_AVE are dictated by the four points defining a grid. 
    If mode is set to 1 (along trench sampling), LON_AVE and LAT_AVE dictated by 2 points in the trench that define a grid.
    Outputs:
    grid_feat_dict
    """

    #prepare variable containing features in a grid
    grid_feat_dict = {}

    if eq_df_col is not None:
        for col in eq_df_col: # loop through all columns in eq_df_col
            grid_feat_dict[col] = []

    # use as coordinate for the heatmap/grid elements
    if dep_df is not None:
        grid_feat_dict['DEP_AVE'] = []
    grid_feat_dict['LON_AVE'] = []
    grid_feat_dict['LAT_AVE'] = []
    grid_feat_dict['S_AVE'] = []
    grid_feat_dict['N_AVE'] = []

    for col in feat_dict.keys():
        grid_feat_dict[col + '_AVE'] = []
        grid_feat_dict[col + '_STD'] = []
        grid_feat_dict[col + '_MAX'] = []
        grid_feat_dict[col + '_MIN'] = []

        grid_feat_dict[col + '_P05'] = []
        grid_feat_dict[col + '_P25'] = []
        grid_feat_dict[col + '_P50'] = []
        grid_feat_dict[col + '_P75'] = []
        grid_feat_dict[col + '_P95'] = []

        grid_feat_dict[col + '_SKW'] = []
        grid_feat_dict[col + '_KUR'] = []

    # processing in here
    n_grid = (s_ax.shape[0] - 1)*(n_ax.shape[0] - 1)

    for i in np.arange(0, n_ax.shape[0] - 1):
        for j in np.arange(0, s_ax.shape[0] - 1):

            if debug: # if debug, draw randomly whether to stop after this iteration or not
                rand_val = random.random()
                debug_dict = {}

            polygon = [(lon_grid[i, j], lat_grid[i, j]),
                       (lon_grid[i + 1, j], lat_grid[i + 1, j]),
                       (lon_grid[i + 1, j + 1], lat_grid[i + 1, j + 1]),
                       (lon_grid[i, j + 1], lat_grid[i, j + 1])]

            # the lon and lat points of the corners - for computing the average values
            lon_pts = np.array([lon_grid[i, j], lon_grid[i + 1, j], lon_grid[i + 1, j + 1], lon_grid[i, j + 1]])
            lat_pts = np.array([lat_grid[i, j], lat_grid[i + 1, j], lat_grid[i + 1, j + 1], lat_grid[i, j + 1]])

            if mode == 1: # sample along trench
                lon_pts2 = np.array([lon_grid[i + 1, j], lon_grid[i + 1, j + 1]])
                lat_pts2 = np.array([lat_grid[i + 1, j], lat_grid[i + 1, j + 1]])


            # points in the s-n coordinate system
            n_pos = 0.5*(n_ax[i + 1] + n_ax[i])
            s_pos = 0.5*(s_ax[j + 1] + s_ax[j])

            # points for corners of square hull
            poly_lon_max = lon_pts.max()
            poly_lon_min = lon_pts.min()
            poly_lat_max = lat_pts.max()
            poly_lat_min = lat_pts.min()

            if dep_df is not None:
                cond = (poly_lon_min < dep_df['LON']) & (dep_df['LON'] <= poly_lon_max) & \
                (poly_lat_min < dep_df['LAT']) & (dep_df['LAT'] <= poly_lat_max)
                dep_sset = dep_df[cond]

                if dep_sset.shape[0] == 0: # no depth values!
                    continue # go back to the start of the loop
            else: # dep_df is not used for decision making
                pass

            # if more than 0 depth values inside, proceed to next steps
            # earthquake related
            if eq_df is not None:
                cond = (poly_lon_min < eq_df['LON']) & (eq_df['LON'] <= poly_lon_max) & \
                (poly_lat_min < eq_df['LAT']) & (eq_df['LAT'] <= poly_lat_max)
                eq_sset = eq_df[cond]

            #comment this since we also look at areas with no recorded events
            #if eq_sset.shape[0] == 0:
            #    continue

            # refined search using the actual polygon
            # check depth
            if dep_df is not None:
                dep_idx_list = []
                for idx, row in dep_sset.iterrows():

                    if(is_inside_polygon(points = polygon, p = (row['LON'], row['LAT']))):
                        dep_idx_list.append(idx)

                if len(dep_idx_list) == 0: # final check for early stopping of iteration
                    continue

                # final depth dataframe
                dep_final = dep_sset.loc[dep_idx_list]
            else:
                pass

            

            # get events inside the grid element
            # earthquake related
            if eq_df is not None:
                eq_idx_list = []
                for idx, row in eq_sset.iterrows():
                    if(is_inside_polygon(points = polygon, p = (row['LON'], row['LAT']))):
                        eq_idx_list.append(idx)

                eq_final = eq_sset.loc[eq_idx_list]

                if eq_final.shape[0] > 0: # an event inside the area
                    max_eq = eq_final[eq_final['MW']  == eq_final['MW'].max()]

                    for col in eq_df_col:
                        grid_feat_dict[col].append(np.array(max_eq[col])[0])
                else: # no event inside
                    for col in eq_df_col:
                        grid_feat_dict[col].append(np.nan) # assign as nan

            # add the mean locations
            if dep_df is not None:
                grid_feat_dict['DEP_AVE'].append(dep_final['VAL'].mean())
            
            grid_feat_dict['S_AVE'].append(s_pos)
            grid_feat_dict['N_AVE'].append(n_pos)
            if mode == 0: # gridded sampling
                grid_feat_dict['LON_AVE'].append(lon_pts.mean())
                grid_feat_dict['LAT_AVE'].append(lat_pts.mean())
            elif mode == 1: # sample along trench
                grid_feat_dict['LON_AVE'].append(lon_pts2.mean())
                grid_feat_dict['LAT_AVE'].append(lat_pts2.mean())

            # compute the stats of the variables in here
            for feat in feat_dict.keys():

                feat_df = feat_dict[feat]

                # initial filter
                cond = (poly_lon_min < feat_df['LON']) & (feat_df['LON'] <= poly_lon_max) & \
                (poly_lat_min < feat_df['LAT']) & (feat_df['LAT'] <= poly_lat_max)
                feat_sset = feat_df[cond]

                # finer filter
                feat_idx_list = []
                for idx, row in feat_sset.iterrows():

                    if(is_inside_polygon(points = polygon, p = (row['LON'], row['LAT']))):
                        feat_idx_list.append(idx)

                # final feat dataframe
                feat_final = feat_sset.loc[feat_idx_list]

                if np.count_nonzero(~np.isnan(feat_final["VAL"])) > 0: # if has any non-nan values

                    if debug and rand_val < 0.5: # save now as the code will end prematurely
                        debug_dict[feat] = feat_final

                    # compute some statistics of the features here
                    grid_feat_dict[feat + '_MAX'].append(np.nanmax(feat_final['VAL']))
                    grid_feat_dict[feat + '_MIN'].append(np.nanmin(feat_final['VAL']))
                    grid_feat_dict[feat + '_AVE'].append(np.nanmean(feat_final['VAL']))
                    grid_feat_dict[feat + '_STD'].append(np.nanstd(feat_final['VAL']))
                    # quantiles
                    grid_feat_dict[feat + '_P05'].append(np.nanquantile(feat_final['VAL'], 0.05))
                    grid_feat_dict[feat + '_P25'].append(np.nanquantile(feat_final['VAL'], 0.25))
                    grid_feat_dict[feat + '_P50'].append(np.nanquantile(feat_final['VAL'], 0.50))
                    grid_feat_dict[feat + '_P75'].append(np.nanquantile(feat_final['VAL'], 0.75))
                    grid_feat_dict[feat + '_P95'].append(np.nanquantile(feat_final['VAL'], 0.95))
                    # skew and kurtosis
                    grid_feat_dict[feat + '_SKW'].append(float(scipy.stats.skew(feat_final['VAL'], nan_policy = 'omit')))
                    grid_feat_dict[feat + '_KUR'].append(float(scipy.stats.kurtosis(feat_final['VAL'], nan_policy = 'omit')))
                else: # no numerical values
                    grid_feat_dict[feat + '_MAX'].append(np.nan)
                    grid_feat_dict[feat + '_MIN'].append(np.nan)
                    grid_feat_dict[feat + '_AVE'].append(np.nan)
                    grid_feat_dict[feat + '_STD'].append(np.nan)
                    grid_feat_dict[feat + '_P05'].append(np.nan)
                    grid_feat_dict[feat + '_P25'].append(np.nan)
                    grid_feat_dict[feat + '_P50'].append(np.nan)
                    grid_feat_dict[feat + '_P75'].append(np.nan)
                    grid_feat_dict[feat + '_P95'].append(np.nan)
                    grid_feat_dict[feat + '_SKW'].append(np.nan)
                    grid_feat_dict[feat + '_KUR'].append(np.nan)

            if debug and rand_val < 0.5: # 0.4 chance of stopping early
                print("Returning prematurely while in debug mode...")
                print("Returning debug_dict...")
                print("Current polygon: ", polygon)
                return(debug_dict)

    return pd.DataFrame.from_dict(grid_feat_dict)

def map_data_to_grid(s_ax, n_ax, lon_grid, lat_grid, dep_df, in_data_df, mode = 0, rm_unmapped = True):
    """
    Map data to a grid - can be earthquake, slab, anything set as a dataframe with LON and LAT columns. 
    Inputs:
    s_ax, n_ax, lon_grid, lat_grid,
    eq_df (if needed),
    dep_df (feature that dictates content presence: important to have LON and LAT as columns). 
    Set to None if you want to bypass (e.g. map events inside grid but allow events outside slab data) 
    in_data_df - dictionary containing earthquakes
    mode - if mode is set to 0 (square grid), LON_AVE and LAT_AVE are dictated by the four points defining a grid. 
    If mode is set to 1 (along trench sampling), LON_AVE and LAT_AVE dictated by 2 points in the trench that define a grid.
    rm_unmapped - set to True if you also want to remove any unmapped data. If set to False, unmapped data will have 
    the following columns set to -1: LON_AVE, LAT_AVE, S_AVE, N_AVE.
    Outputs:
    eq_df - copy of in_data_df with the positions
    """
    eq_df = in_data_df.copy(deep = True)
    eq_df['LON_AVE'] = -1
    eq_df['LAT_AVE'] = -1
    eq_df['S_AVE'] = -1
    eq_df['N_AVE'] = -1

    # processing in here
    n_grid = (s_ax.shape[0] - 1)*(n_ax.shape[0] - 1)

    for i in np.arange(0, n_ax.shape[0] - 1):
        for j in np.arange(0, s_ax.shape[0] - 1):
            polygon = [(lon_grid[i, j], lat_grid[i, j]),
                       (lon_grid[i + 1, j], lat_grid[i + 1, j]),
                       (lon_grid[i + 1, j + 1], lat_grid[i + 1, j + 1]),
                       (lon_grid[i, j + 1], lat_grid[i, j + 1])]

            # the lon and lat points of the corners
            lon_pts = np.array([lon_grid[i, j], lon_grid[i + 1, j], lon_grid[i + 1, j + 1], lon_grid[i, j + 1]])
            lat_pts = np.array([lat_grid[i, j], lat_grid[i + 1, j], lat_grid[i + 1, j + 1], lat_grid[i, j + 1]])

            if mode == 1: # sample along trench
                lon_pts2 = np.array([lon_grid[i + 1, j], lon_grid[i + 1, j + 1]])
                lat_pts2 = np.array([lat_grid[i + 1, j], lat_grid[i + 1, j + 1]])

            # points in the s-n coordinate system
            n_pos = 0.5*(n_ax[i + 1] + n_ax[i])
            s_pos = 0.5*(s_ax[j + 1] + s_ax[j])

            # points for corners of square hull
            poly_lon_max = lon_pts.max()
            poly_lon_min = lon_pts.min()
            poly_lat_max = lat_pts.max()
            poly_lat_min = lat_pts.min()

            if dep_df is not None:
                cond = (poly_lon_min < dep_df['LON']) & (dep_df['LON'] <= poly_lon_max) & \
                (poly_lat_min < dep_df['LAT']) & (dep_df['LAT'] <= poly_lat_max)
                dep_sset = dep_df[cond]

                if dep_sset.shape[0] == 0: # no depth values!
                    continue # go back to the start of the loop
            else:
                pass # continue process

            # if more than 0 depth values inside, proceed to next steps
            # earthquake related
            cond = (poly_lon_min < eq_df['LON']) & (eq_df['LON'] <= poly_lon_max) & \
                (poly_lat_min < eq_df['LAT']) & (eq_df['LAT'] <= poly_lat_max)
            eq_sset = eq_df[cond]

            # if there are no events in element, then proceed to next element
            if eq_sset.shape[0] == 0:
                continue

            # refined search using the actual polygon
            # check depth
            if dep_df is not None:
                dep_idx_list = []
                for idx, row in dep_sset.iterrows():

                    if(is_inside_polygon(points = polygon, p = (row['LON'], row['LAT']))):
                        dep_idx_list.append(idx)

                if len(dep_idx_list) == 0: # final check for early stopping of iteration
                    continue
            else:
                pass # bypass 

            # get events inside the grid element - assured that there are some events in here
            # earthquake related
            eq_idx_list = []
            for idx, row in eq_sset.iterrows():
                if(is_inside_polygon(points = polygon, p = (row['LON'], row['LAT']))):
                    eq_idx_list.append(idx)

            if len(eq_idx_list) > 0: # an event inside the area
                # add the mean locations
                if mode == 0:
                    eq_df.loc[eq_idx_list, 'LON_AVE'] = lon_pts.mean()
                    eq_df.loc[eq_idx_list, 'LAT_AVE'] = lat_pts.mean()

                elif mode == 1: # sample along trench
                    eq_df.loc[eq_idx_list, 'LON_AVE'] = lon_pts2.mean()
                    eq_df.loc[eq_idx_list, 'LAT_AVE'] = lat_pts2.mean()

                eq_df.loc[eq_idx_list, 'S_AVE'] = s_pos
                eq_df.loc[eq_idx_list, 'N_AVE'] = n_pos

    if rm_unmapped: # remove unmapped data
        eq_df = eq_df[eq_df["S_AVE"] > -1]

    return eq_df
