import pandas as pd

def conv_lon(in_df, to_360, col_list = None):
    """
    Convert the range of the LON column in in_df from (-180, 180) to (0, 360) if to_360 is True
    and (0, 360) to (-180, 180) if to_360 is False.
    """

    if col_list is None:
        col = ["LON"]
    else:
        col = col_list

    for c in col:
        if to_360:
            in_df[c] = in_df[c]%360
        else:
            in_df[c] = (in_df[c] + 180)%360 - 180

    return in_df
