# -*- coding: utf-8 -*-
"""
Created on Aug 22 23:41:22 2023

@author: Akhil Jaini

Note: Use the requirements.txt file to update all dependencies 
to the appropriate (latest as of writing this code) versions.

"""

import os
import numpy as np
from psrqpy import QueryATNF
from astropy import wcs
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename

# Function to query the ATNF Pulsar Catalogue: https://www.atnf.csiro.au/research/pulsar/psrcat/
def QueryPSRCat():
    query = QueryATNF()
    query.query_params = ['JName', 'RaJD', 'DecJD', 'P0', 'DM', 'W50', 'W10', 'S400', 'S1400']

    # Converting the query into a Pandas dataframe
    df = query.pandas
    
    # Only having pulsars above a certain threshold and dropping the remaining ones
    df['C1'] = df.apply(lambda x: x['S1400']*x['P0']/x['W50'], axis=1)
    df['C1'].replace('', np.nan, inplace=True)
    df.dropna(subset=['C1'], inplace=True)
    df = df.drop(df[df['C1'] < 0.03].index)
    df = df.sort_values(by=['C1'], ascending=False)
    df = df[['JNAME', 'RAJD', 'RAJD_ERR', 'DECJD', 'DECJD_ERR', 'P0', 'P0_ERR', 'DM', 'DM_ERR', 
            'W50', 'W50_ERR', 'W10', 'W10_ERR', 'S400', 'S400_ERR', 'S1400', 'S1400_ERR', 'C1']]            # Rearranging the columns
    
    numstring = 'Version {} of the ATNF catalogue contains {} pulsars, and after sorting contains {} pulsars'
    print(numstring.format(query.get_version, query.num_pulsars, df.shape[0]))                              # For testing only, can be removed
    df.to_csv('psrcat_Main.csv')
    
    return df


# Function to iterate through all of the FITS header files (currently only works with files that are already in the directory, 
# but a realtime version can be implemented to read the files as they are dumped during observation). The function then reads
# the RA, DEC, lpix and mpix values to find the field of view of the corresponding candidate file. Finally, the function 
# compares this with the sources in the catalogue and returns two dataframes (also saved as CSV files): one with the sources 
# in the field of view, another with the sources out of the field of view. The next step of the pipeline will have both these 
# files as input
def check_field(dirpath=os.curdir):                                                                         # Directory path of FITS files to be provided
    fitsfiles = [f for f in os.listdir(dirpath) if f.endswith(".fits")]
    
    # Looping through the FITS files to get required info from the headers
    for i in range(len(fitsfiles)):
        header = fits.getheader(fitsfiles[i])
        world_coords = wcs.WCS(header)
        l_axis = fits.getval(fitsfiles[i], 'NAXIS1')
        m_axis = fits.getval(fitsfiles[i], 'NAXIS2')
        ra_cen = fits.getval(fitsfiles[i], 'CRVAL1')
        dec_cen = fits.getval(fitsfiles[i], 'CRVAL2')
        
        # Finding the field of view of the candidate file
        [ra_primary1, dec_primary1], [ra_primary2, dec_primary2] = world_coords.wcs_pix2world([[0,0], [l_axis-1,m_axis-1]], 0)
        
        # Adding an excess to the searchable field of view to find the alised FoV
        excess_ra, excess_dec = abs(ra_primary1 - ra_primary2), abs(dec_primary1 - dec_primary2)
        ra_aliased_min, ra_alised_max = np.minimum(ra_primary1, ra_primary2) - excess_ra, np.maximum(ra_primary1, ra_primary2) + excess_ra
        dec_aliased_min, dec_aliased_max = np.minimum(dec_primary1, dec_primary2) - excess_dec, np.maximum(dec_primary1, dec_primary2) + excess_dec
        
        # Calling the pulsar query function and adding a column based on whether the pulsars are within the FoV 
        df = QueryPSRCat()
        df['In_Field'] = df.apply(lambda x: 'YES' if ((ra_aliased_min <= x['RAJD'] <= ra_alised_max) 
                                                      & (dec_aliased_min <= x['DECJD'] <= dec_aliased_max)) else 'NO', axis=1)
        
        # Creating two dataframes, one for sources within the FoV and another for sources outside
        df_InField = df.drop(df[df['In_Field'] == 'NO'].index)
        df_InField.to_csv(f'{fitsfiles[i].split(".")[0]}_Catalogue_InField.csv')
        df_NotInField = df.drop(df[df['In_Field'] == 'YES'].index)
        df_NotInField.to_csv(f'{fitsfiles[i].split(".")[0]}_Catalogue_NotInField.csv')
    
    return df_InField, df_NotInField


# Calling the main function
if __name__=='__main__':
    check_field()
    
    