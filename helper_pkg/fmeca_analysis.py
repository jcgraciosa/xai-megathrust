import pandas as pd

def get_prof_sted(sted_prof_fname):
    """
    Gets the start and end longitude and latitude of the trench normal profiles.
    """
    sted_prof =  pd.read_csv(sted_prof_fname, header = None, sep = ' ')
    sted_prof.columns = ['PROF', 'START', 'END']

    st_lon = []
    st_lat = []
    ed_lon = []
    ed_lat = []

    for idx, row in sted_prof.iterrows():
        # start
        lon, lat = row['START'].split("/")

        st_lon.append(float(lon))
        st_lat.append(float(lat))

        # end
        lon, lat = row['END'].split("/")

        ed_lon.append(float(lon))
        ed_lat.append(float(lat))

    sted_prof['ST_LON'] = st_lon
    sted_prof['ST_LAT'] = st_lat
    sted_prof['ED_LON'] = ed_lon
    sted_prof['ED_LAT'] = ed_lat

    sted_prof.drop(columns=['START', 'END'], inplace = True)
    sted_prof = sted_prof.sort_values(by = 'PROF', inplace = False)

    return sted_prof

