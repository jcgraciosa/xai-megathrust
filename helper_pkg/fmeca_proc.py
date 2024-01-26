import pandas as pd
import numpy as np

def create_cmt_arr(df_row):
    """
    Creates a 2D matrix representation of the cmt and computes the Frobenius norm
    """
    #print(df_row)
    cmt_arr = np.zeros([3, 3])
    cmt_arr[0, :] = np.array(df_row[['MRR', 'MRT', 'MPR']])
    cmt_arr[1, :] = np.array(df_row[['MRT', 'MTT', 'MTP']])
    cmt_arr[2, :] = np.array(df_row[['MPR', 'MTP', 'MPP']])

    # computes the Frobenius norm - our tensor magnitude
    mag = np.sqrt((cmt_arr**2).sum())

    return mag, cmt_arr

def compute_similarity(arr1, arr2, mag1, mag2):
    """
    Computes the dot product of two moment tensor
    """

    tdp = (arr1*arr2).sum()
    tdp = tdp/(mag1*mag2)

    return tdp

def prep_beachball(df):
    """
    Gets the focal mechanism components from our dataframe and packs them into an array.
    GCMT uses U-S-E coordinate system. In this coordinate system:
        MRR = MDD   MRT = MND
        MTT = MNN   MRT = -MED
        MPP = MEE   MTP = -MNE

    """
    bb =  np.array(df[['MTT', 'MPP', 'MRR', 'MTP', 'MRT', 'MPR']])
    bb[3] = -bb[3]
    bb[5] = -bb[5]

    return bb

def compute_measurables(eq_to_proc, max_pos, ds):
    """
    Computes the measurables we are interested in:
    Inputs:
    eq_to_proc: earthquake dataframe to process
    max_pos: max distance along trench
    ds: along trench distance differential

    Outputs:
    pos_all: position along trench of each event
    fin_tot_en: total energy corresponding to pos_all
    fin_max_mag: maximum magnitude corresponding to pos_all
    fin_num_ev: number of events corresponding to pos_all
    """

    eq_to_proc = eq_to_proc.drop_duplicates(subset = ['EVENT_ID'], keep = 'first')  # col 6 is the EQ event
    print(eq_to_proc.shape)

    eq_to_proc = eq_to_proc.reset_index(drop=True)
    proc_df = eq_to_proc.sort_values(by = 'ALONG', ascending = True)
    along_pos = pd.unique(proc_df['ALONG'])
    along_tot_energy = np.zeros(along_pos.shape[0])
    along_max_mag = np.zeros(along_pos.shape[0])
    along_num_ev = np.zeros(along_pos.shape[0])

    grp = proc_df.groupby('ALONG')
    event_cnt = 0
    idx = 0
    for pos in along_pos:
        to_proc = grp.get_group(pos)
        along_num_ev[idx] = to_proc.shape[0]

        for i, row in to_proc.iterrows():

            exp = np.asarray(row['EX'])
            mo = np.asarray(row['MO'])
            mw = np.asarray(row['MW'])

            energy = mo*np.power(10., exp)

            #if mw > along_max_mag[idx]: # get the max magnitude
            #    along_max_mag[idx] = mw
            if energy > along_max_mag[idx]: # get the max magnitude
                along_max_mag[idx] = energy

            along_tot_energy[idx] += energy

            event_cnt += 1

        idx += 1



    along_tot_energy = np.array(along_tot_energy)
    # convert to Mw
    #along_tot_energy = (np.log10(along_tot_energy) - 9.05)/1.5

    along_max_mag = np.array(along_max_mag)
    along_max_mag = (np.log10(along_max_mag) - 9.05)/1.5

    along_pos = np.array(along_pos)

    #along_pos = max_pos - along_pos # need to reverse shit - 07/08/2020 - needed to remove this soon

    pos_all = np.arange(0, max_pos + ds, ds)
    fin_tot_en = np.zeros(pos_all.shape)
    fin_max_mag = np.zeros(pos_all.shape)
    fin_num_ev = np.zeros(pos_all.shape)

    idx = (along_pos)//ds

    # assign to
    fin_tot_en[idx] = along_tot_energy
    fin_max_mag[idx] = along_max_mag
    fin_num_ev[idx] = along_num_ev

    return pos_all, fin_tot_en, fin_max_mag, fin_num_ev

def cum_mw_by_angle(mo, exp, angle, edges):

    """
    Get cumulative seismic moment magnitude according to the angles given in edges
    NOTE: angle and angle + 180 are considered the same

    Inputs:
    mo: array or dataframe series containing the moment of the earthquakes to process
    exp: array or dataframe series containing moment exponent of earthquakes to process
    angle: array or dataframe series containing the P axis angles with respect to trench perpendicular direction
    edges: angular bin edges

    Output:
    cum_mw: cumulative seismic moment magnitude corresponding to edges
    """

    cum_mw = np.zeros(edges.shape[0] - 1)

    cum_mo1 = cum_mo_by_angle(mo, exp, angle, edges)
    cum_mo2 = cum_mo_by_angle(mo, exp, angle + 180, edges)
    cum_mo = cum_mo1 + cum_mo2

    idx = cum_mo > 0

    mw_ss = (np.log10(cum_mo[idx]) - 9.05)/1.5

    cum_mw[idx] = mw_ss

    return cum_mw


def cum_mo_by_angle(mo, exp, angle, edges):

    """
    Get cumulative seismic moment according to the angles given in edges
    NOTE: angle and angle + 180 are different in this case

    Inputs:
    mo: array or dataframe series containing the moment of the earthquakes to process
    exp: array or dataframe series containing moment exponent of earthquakes to process
    angle: array or dataframe series containing the P axis angles with respect to trench perpendicular direction
    edges: angular bin edges

    Output:
    cum_mo: cumulative Mo corresponding to edges
    """


    st_edge = edges[:-1]
    ed_edge = edges[1:]

    cum_mo = np.zeros(st_edge.shape)

    for i, st, ed in zip(np.arange(st_edge.shape[0]), st_edge, ed_edge):

        idx = (st <= angle) & (angle < ed)

        exp_subset = np.array(exp[idx])
        mo_subset = np.array(mo[idx])

        if exp_subset.shape[0] == 0:
            continue


        energy = mo_subset*np.power(10., exp_subset)


        cum_mo[i] = energy.sum()


    return cum_mo

def assign_eq_type(t_pl, n_pl, p_pl):

    """
    Compute the square of the sines of the T,N,P plunges of the earthquakes
    and gives a classification according to Frohlich and Apperson 1992
    Notes:
    1. Thrust if T plunge is closest to vertical
    2. Normal if P plunge is closest to vertical
    3. Strike-slip if N plunge is closest to vertical
    4. Classification ignores dip of the place where earthquake occured

    Inputs:
    t_pl: tension axis plunge
    n_pl: neutral axis plunge
    p_pl: compression axis plunge

    Output:
    sin_thrust: square of the sin of the T axis
    sin_ss: square of the sin of the N axis
    sin_norm: square of the sin of the P axis
    classif: possible classif: THRUST, NORM, SS, O_THRUST, O_NORM, ODD
    """

    # try to be strict first with the classification - follow Frohlich 1992
    sin_thrust = (np.sin(t_pl*np.pi/180))**2
    sin_ss = (np.sin(n_pl*np.pi/180))**2
    sin_norm = (np.sin(p_pl*np.pi/180))**2

    classif = []

    for thrust, ss, norm in zip(sin_thrust, sin_ss, sin_norm):
        if thrust > 0.59:
        #if max(thrust, ss, norm) == thrust:
            classif.append('THRUST')
        elif norm > 0.75:
        #elif max(thrust, ss, norm) == norm:
            classif.append('NORM')
        elif ss > 0.75:
        #elif max(thrust, ss, norm) == ss:
            classif.append('SS')
        else:
            if thrust > norm and ss > norm:
                classif.append('O_THRUST')
            elif norm > thrust and ss > thrust:
                classif.append('O_NORM')
            else:
                classif.append('ODD')

    return sin_thrust, sin_ss, sin_norm, classif

def compute_corrected_tnp(eq_df, strk_df = None, mthrust_depth = 70):
    """
    Compute the corrected T, N, P plunge of the events in eq_df

    Inputs:
    eq_df: dataframe containing info on earthquake events
    must have T_PL, N_PL, P_PL, T_AZM, N_AZM, P_AZM, SLAB_DIP, ALONG, SLAB_STR
    strk_df: strike along the trench obtained by using the processing code I have
    if this is set to None, then we don't use the average strike sampled along trench
    Nov 12 2023 - add option of using the strike by sampling from the closest slab 2.0 data
    mthrust_depth: depth of the megathrust, we only correct the TNP when events are deeper than 70 km
    and in the downgoing plate (so CLASS column must also be present)
    

    Output:
    corr_t_pl: corrected T plunge
    corr_n_pl: corrected N plunge
    corr_p_pl: corrected P plunge
    """

    corr_t_pl = [] # corrected
    corr_n_pl = []
    corr_p_pl = []

    for idx, row in eq_df.iterrows():

        # get values to use in computation
        if strk_df is not None:
            al_val = row['ALONG']
        
        if row["CLASS"] == "DGOING" and row["DEPTH"] > mthrust_depth:

            if strk_df is not None: 
                trench_az = float(strk_df[strk_df['ALONG'] == al_val]['VALUE'])
            else: 
                trench_az = float(row["SLAB_STR"])  # slab strike

            dip_az = trench_az + 90

            if dip_az > 360:
                dip_az = dip_az - 360

            #print(trench_az, '  ', trench_az + 90, '  ', dip_az)

            # corrected TENSION
            az_diff = np.abs(row['T_AZM'] - dip_az)
            if az_diff <= 90: # same side
                new_pl = np.abs(row['T_PL'] - row['SLAB_DIP'])
            else: # check this one 
                new_pl = row['T_PL'] + row['SLAB_DIP']
            if new_pl > 90: # check this one
                new_pl = 180 - new_pl

            corr_t_pl.append(new_pl)

            # corrected NEUTRAL
            az_diff = np.abs(row['N_AZM'] - dip_az)
            if az_diff <= 90: # same side
                new_pl = np.abs(row['N_PL'] - row['SLAB_DIP'])
            else:
                new_pl = row['N_PL'] + row['SLAB_DIP']

            if new_pl > 90:
                new_pl = 180 - new_pl

            corr_n_pl.append(new_pl)

            # correcrted COMPRESSION
            az_diff = np.abs(row['P_AZM'] - dip_az)
            if az_diff <= 90: # same side
                new_pl = np.abs(row['P_PL'] - row['SLAB_DIP'])
            else:
                new_pl = row['P_PL'] + row['SLAB_DIP']

            if new_pl > 90:
                new_pl = 180 - new_pl

            corr_p_pl.append(new_pl)

            # debug concept
            #     if new_pl > 90:
            #         print(dip_az)
            #         print(row['P_AZM'])
            #         print(row['P_PL'])
            #         print(row['SLAB_DIP'])
            #         print(new_pl)
            #         break
        else: # no modificiations/corrections done 
            corr_t_pl.append(row["T_PL"])
            corr_n_pl.append(row["N_PL"])
            corr_p_pl.append(row["P_PL"])

    return corr_t_pl, corr_n_pl, corr_p_pl

# def compute_corrected_tnp(eq_df, strk_df):
#     """
#     Compute the corrected T, N, P plunge of the events in eq_df

#     Inputs:
#     eq_df: dataframe containing info on earthquake events
#     must have T_PL, N_PL, P_PL, T_AZM, N_AZM, P_AZM, SLAB_DIP, ALONG
#     strk_df: strike along the trench obtained by using the processing code I have

#     Output:
#     corr_t_pl: corrected T plunge
#     corr_n_pl: corrected N plunge
#     corr_p_pl: corrected P plunge
#     """

#     corr_t_pl = [] # corrected
#     corr_n_pl = []
#     corr_p_pl = []

#     for idx, row in eq_df.iterrows():

#         # get values to use in computation
#         al_val = row['ALONG']
#         trench_az = float(strk_df[strk_df['ALONG'] == al_val]['VALUE'])
#         dip_az = trench_az + 90

#         if dip_az > 360:
#             dip_az = dip_az - 360

#         #print(trench_az, '  ', trench_az + 90, '  ', dip_az)

#         # corrected TENSION
#         az_diff = np.abs(row['T_AZM'] - dip_az)
#         if az_diff <= 90: # same side
#             new_pl = np.abs(row['T_PL'] - row['SLAB_DIP'])
#         else:
#             new_pl = row['T_PL'] + row['SLAB_DIP']
#         if new_pl > 90:
#             new_pl = 180 - new_pl

#         corr_t_pl.append(new_pl)
#         # corrected NEUTRAL
#         az_diff = np.abs(row['N_AZM'] - dip_az)
#         if az_diff <= 90: # same side
#             new_pl = np.abs(row['N_PL'] - row['SLAB_DIP'])
#         else:
#             new_pl = row['N_PL'] + row['SLAB_DIP']

#         if new_pl > 90:
#             new_pl = 180 - new_pl

#         corr_n_pl.append(new_pl)

#         # correcrted COMPRESSION
#         az_diff = np.abs(row['P_AZM'] - dip_az)
#         if az_diff <= 90: # same side
#             new_pl = np.abs(row['P_PL'] - row['SLAB_DIP'])
#         else:
#             new_pl = row['P_PL'] + row['SLAB_DIP']

#         if new_pl > 90:
#             new_pl = 180 - new_pl

#         # debug concept
#     #     if new_pl > 90:
#     #         print(dip_az)
#     #         print(row['P_AZM'])
#     #         print(row['P_PL'])
#     #         print(row['SLAB_DIP'])
#     #         print(new_pl)
#     #         break

#         corr_p_pl.append(new_pl)

#     return corr_t_pl, corr_n_pl, corr_p_pl

def compute_clvd_mrk_sz(eq_df):

    """
    Computes the CLVD ratio and the mark size for plotting
    Inputs:
    eq_df - dataframe containing the T,N,P values (eigenvalues) and Mw

    Outputs:
    clvd - CLVD ratio
    mrk_sz - mark size
    """

    clvd = []
    for idx, row in eq_df.iterrows():
        val = abs(row['N_VAL'])/max(abs(row['T_VAL']), abs(row['P_VAL']))
        clvd.append(val)

    # add for size
    n = 3
    trans = eq_df['MW']**n
    max_val = 150
    min_val = 0
    m = (max_val - min_val)/(trans.max() - trans.min())
    trans = m*(trans - trans.min()) + min_val

    return clvd, trans

def get_angle_relative_to_dip(strk_df, eq_df):

    """
    TYPE_2 analysis is better
    Function compare the direction of the tension and compression axes for thrust and normal events with the fault strike
    If event is normal or oblique normal - use azimuth of tension axis
    If event is thrust or oblique thrust - use azimuth of compression axis
    If event is ss or odd - set angle to zero

    Input:
    strk_df: dataframe containing the strike along trench
    eq_df: earthquake events dataframe

    Output:
    p_angle_arr: P angle with respect to trench perpendicular (0 deg if trench perpendicular)
    t_angle_arr: T angle with respect to trench perpendicular (0 deg if trench perpendicular)

    """

    p_angle_arr = np.zeros(eq_df.shape[0])
    t_angle_arr = np.zeros(eq_df.shape[0])

    i = 0
    for idx, row in eq_df.iterrows():

        al = float(row['ALONG'])
        strike = float(strk_df[strk_df['ALONG'] == al]['VALUE'])
        #print(strike)

        if row['TYPE_2'] == 'ODD':
            p_angle_arr[i] = 0 # do not include ODD earthquakes
            t_angle_arr[i] = 0
            i += 1
            continue

        p_angle_to_use = row['P_AZM']
        t_angle_to_use = row['T_AZM']

        #print(row['P_AZM'], row['T_AZM'])
        # processing for thrust, o_thrust, norm, o_norm
        dip = strike + 90 # should be dip azimuth
        if dip >= 360:
            dip = dip - 360

        # P angle
        p_angle = dip - p_angle_to_use
        other_angle = 360 - np.abs(p_angle) # other_angle and sgn_oppose are dummy variables
        sgn_oppose = -1*p_angle/np.abs(p_angle)

        if np.abs(p_angle) <= np.abs(other_angle): # use the angle
            p_angle_arr[i] = p_angle
        else: # use the other angle and let it have the opposite sign of the angle
            p_angle_arr[i] = sgn_oppose*other_angle

        if p_angle_arr[i] < 0:
            #print(p_angle_arr[i])
            p_angle_arr[i] = 180 + p_angle_arr[i]

        # T angle
        t_angle = dip - t_angle_to_use
        other_angle = 360 - np.abs(t_angle)
        sgn_oppose = -1*t_angle/np.abs(t_angle)

        if np.abs(t_angle) <= np.abs(other_angle):
            t_angle_arr[i] = t_angle
        else:
            t_angle_arr[i] = sgn_oppose*other_angle

        if t_angle_arr[i] < 0:
            t_angle_arr[i] = 180 + t_angle_arr[i]

        # gives the deviation from the perpendicular - acute angle
#         val = np.abs(angle_arr[i])
#         if val > 90:
#             angle_arr[i] = 180 - val
#         else:
#             angle_arr[i] = val

        # use the absolute deviation - for downgoing plate - but may be changee
        #angle_arr[i] = np.abs(angle_arr[i])


        i += 1

    return p_angle_arr, t_angle_arr

