import pandas as pd
import numpy as np

def parse_sted_prof(sted_prof_df):
    """
    This is for parsing a dataframe containing the prof, start, end
    Input:
    sted_prof_df - dataframe containing the start and end
    Output:
    dataframe containing the parsed sted prof
    """

    st_lon = []
    st_lat = []
    ed_lon = []
    ed_lat = []

    for idx, row in sted_prof_df.iterrows():
        # start
        lon, lat = row['START'].split("/")

        st_lon.append(float(lon))
        st_lat.append(float(lat))

        # end
        lon, lat = row['END'].split("/")

        ed_lon.append(float(lon))
        ed_lat.append(float(lat))

    sted_prof_df['ST_LON'] = st_lon
    sted_prof_df['ST_LAT'] = st_lat
    sted_prof_df['ED_LON'] = ed_lon
    sted_prof_df['ED_LAT'] = ed_lat

    sted_prof_df.drop(columns=['START', 'END'], inplace = True)
    out_df = sted_prof_df.sort_values(by = 'PROF', inplace = False)

    return out_df


