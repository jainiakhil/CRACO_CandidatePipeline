# -*- coding: utf-8 -*-
"""
Created on Aug 22 23:41:22 2023

@author: Akhil Jaini

Note: Use the requirements.txt file to update all dependencies 
to the appropriate (latest as of writing this code) versions.

"""

import os
import numpy as np
import pandas as pd
from psrqpy import QueryATNF
from astropy import wcs
from astropy.io import fits


def get_parser():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='cluster arguments', formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
    parser.add_argument('--min_flux', type=float, help='Minimum signal flux for querying the catalogue', default='0.03')
    parser.add_argument('--ra_err', type=float, help='Maximum error to be allowed in RA measurement for querying the catalogue', default='2e-3')
    parser.add_argument('--dec_err', type=float, help='Maximum error to be allowed in DEC measurement for querying the catalogue', default='2e-3')
    return parser

# Function to query the ATNF Pulsar Catalogue: https://www.atnf.csiro.au/research/pulsar/psrcat/
# Taking in arguments of flux, ra_err and dec_err to give threshold values for pulsar filtering from the catalogue
# Default values are taken as Min Flux = 0.03, Min_RA_Err = 2e-3, Min_DEC_Err = 2e-3, as given in argparse above
def QueryPSRCat(flux, ra_err, dec_err):
    if os.path.exists('psrcat_Main.csv'):
        print("PSRCAT Catalogue file already exists")
    else:
        query = QueryATNF()
        query.query_params = ['JName', 'RaJD', 'DecJD', 'P0', 'DM', 'W50', 'W10', 'S400', 'S1400']

        # Converting the query into a Pandas dataframe
        df = query.pandas

        # Only having pulsars that have RA and DEC measured above a certain astrometric precision, and whose error values are provided
        df['RAJD_ERR'].replace('', np.nan, inplace=True)
        df.dropna(subset=['RAJD_ERR'], inplace=True)
        df = df.drop(df[df['RAJD_ERR'] > ra_err].index)
        df['DECJD_ERR'].replace('', np.nan, inplace=True)
        df.dropna(subset=['DECJD_ERR'], inplace=True)
        df = df.drop(df[df['DECJD_ERR'] > dec_err].index)
        
        # Only having pulsars above a certain threshold flux and dropping the remaining ones
        df['C1'] = df.apply(lambda x: x['S1400']*x['P0']/x['W50'], axis=1)
        df['C1'].replace('', np.nan, inplace=True)
        df.dropna(subset=['C1'], inplace=True)
        df = df.drop(df[df['C1'] < flux].index)
        df = df.sort_values(by=['C1'], ascending=False)
        df = df[['JNAME', 'RAJD', 'RAJD_ERR', 'DECJD', 'DECJD_ERR', 'P0', 'P0_ERR', 'DM', 'DM_ERR', 
                'W50', 'W50_ERR', 'W10', 'W10_ERR', 'S400', 'S400_ERR', 'S1400', 'S1400_ERR', 'C1']]            # Rearranging the columns
        
        numstring = 'Version {} of the ATNF Catalogue contains {} pulsars, and after filtering contains {} pulsars.'
        print(numstring.format(query.get_version, query.num_pulsars, df.shape[0]))                              # For testing only, can be removed
        df.to_csv('psrcat_Main.csv')

        return df
    
def QueryRACSCat(flux, ra_err, dec_err):
    if os.path.exists('racscat_Main.csv'):
        print("RACS Catalogue file already exists")
    else:
        # Catalogue of sources in galactic region
        df1 = pd.read_csv('AS110_Derived_Catalogue_racs_dr1_sources_galacticregion_v2021_08_v02_5726.csv')
        
        df1 = df1[['id', 'source_name', 'sbid', 'ra', 'e_ra', 'dec', 'e_dec', 'total_flux_source', 's_code']]     # Dropping unwanted columns
        
        # Only having sources that have RA and DEC measured above a certain astrometric precision, and whose error values are provided
        df1['e_ra'].replace('', np.nan, inplace=True)
        df1.dropna(subset=['e_ra'], inplace=True)
        df1 = df1.drop(df1[df1['e_ra'] > ra_err*3600].index)
        df1['e_dec'].replace('', np.nan, inplace=True)
        df1.dropna(subset=['e_dec'], inplace=True)
        df1 = df1.drop(df1[df1['e_dec'] > dec_err*3600].index)
        
        # Only having sources above a certain threshold flux and dropping the remaining ones
        df1['total_flux_source'].replace('', np.nan, inplace=True)
        df1.dropna(subset=['total_flux_source'], inplace=True)
        df1 = df1.drop(df1[df1['total_flux_source'] < flux*1e3].index)
        
        # Catalpogue of sources away from galactic region
        df2 = pd.read_csv('AS110_Derived_Catalogue_racs_dr1_sources_galacticcut_v2021_08_v02_5725.csv')
        
        df2 = df2[['id', 'source_name', 'sbid', 'ra', 'e_ra', 'dec', 'e_dec', 'total_flux_source', 's_code']]     # Dropping unwanted columns
        
        # Only having sources that have RA and DEC measured above a certain astrometric precision, and whose error values are provided
        df2['e_ra'].replace('', np.nan, inplace=True)
        df2.dropna(subset=['e_ra'], inplace=True)
        df2 = df2.drop(df2[df2['e_ra'] > ra_err*3600].index)
        df2['e_dec'].replace('', np.nan, inplace=True)
        df2.dropna(subset=['e_dec'], inplace=True)
        df2 = df2.drop(df2[df2['e_dec'] > dec_err*3600].index)
        
        # Only having sources above a certain threshold flux and dropping the remaining ones
        df2['total_flux_source'].replace('', np.nan, inplace=True)
        df2.dropna(subset=['total_flux_source'], inplace=True)
        df2 = df2.drop(df2[df2['total_flux_source'] < flux*1e3].index)
        
        df = pd.concat([df1, df2])
        
        print(f'The RACS DR1 Catalogue contains {df.shape[0]} sources after filtering.')
        df.to_csv('racscat_Main.csv')
    
        return df

# For CRACO right now, integration time is 100ms (or 0.1sec), SEFD is 2000/24antennas (= 83.333), 
# bandwidth is 120MHz (or 120e6Hz), and number of polarizations is 2
def CalcSNR(s400, s1400, w50, p0, t_int=0.1, sefd=83.333, bw=120e6, npol=2):
    if s1400:
        s800 = 10**(np.log10(s1400)*np.log10(1400)/np.log10(800))
    else:
        s800 = 10**(np.log10(s400)*np.log10(400)/np.log10(800))  
    
    if t_int > p0:
        flux_800 = s800
    else:
        if t_int > (w50/1000):
            flux_800 = s800*p0/t_int
        else:
            flux_800 = s800*p0/(w50/1000)
    img_rms = 1e3*sefd/np.sqrt(bw*t_int*npol)
    snr = flux_800/img_rms
    
    return snr    

# Function to iterate through all of the FITS header files (currently only works with files that are already in the directory, 
# but a realtime version can be implemented to read the files as they are dumped during observation). The function then reads
# the RA, DEC, lpix and mpix values to find the field of view of the corresponding candidate file. Finally, the function 
# compares this with the sources in the catalogue and returns two dataframes (also saved as CSV files): one with the sources 
# in the field of view, another with the sources out of the field of view. The next step of the pipeline will have both these 
# files as input
def CheckField(dirpath=os.curdir):                                                                         # Directory path of FITS files to be provided
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
        
        # Adding an excess that is twice the primary FoV to the searchable FoV to find the alised FoV, 
        # and ensuring that the RA and DEC are within their bounds
        excess_ra, excess_dec = abs(ra_primary2 - ra_primary1), abs(dec_primary2 - dec_primary1)
        
        # if ra_primary1 > ra_primary2:
        #     ra_aliased_min = ((np.minimum(ra_primary1, ra_primary2) + excess_ra) % 360 + 360) % 360
        #     ra_aliased_max = ((np.maximum(ra_primary1, ra_primary2) - excess_ra) % 360 + 360) % 360
        # else:    
        ra_aliased_min = ((np.minimum(ra_primary1, ra_primary2) - excess_ra) % 360 + 360) % 360
        ra_aliased_max = ((np.maximum(ra_primary1, ra_primary2) + excess_ra) % 360 + 360) % 360
        dec_aliased_min = np.clip(np.minimum(dec_primary1, dec_primary2) - excess_dec, -90, 90)
        dec_aliased_max = np.clip(np.maximum(dec_primary1, dec_primary2) + excess_dec, -90, 90)
        
        # Calling the pulsar query catalogue and adding a column based on whether the pulsars are within the FoV 
        df_psrcat = pd.read_csv('PSRCat_Main.csv')
        
        # For cases when the field is crossing the RA = 0deg line, or when DEC is close to +-90deg, special conditions are applied
        if (ra_aliased_max < ra_aliased_min):
            df_psrcat['In_Field'] = df_psrcat.apply(lambda x: 'YES' if (((0 <= x['RAJD'] <= ra_aliased_max) & (ra_aliased_min <= x['RAJD'] <= 360)) 
                                                      & (dec_aliased_min <= x['DECJD'] <= dec_aliased_max)) else 'NO', axis=1)
        elif (dec_aliased_min == -90 or dec_aliased_max == +90):
            df_psrcat['In_Field'] = df_psrcat.apply(lambda x: 'YES' if (dec_aliased_min <= x['DECJD'] <= dec_aliased_max) else 'NO', axis=1)
        else:
            df_psrcat['In_Field'] = df_psrcat.apply(lambda x: 'YES' if ((ra_aliased_min <= x['RAJD'] <= ra_aliased_max) 
                                                      & (dec_aliased_min <= x['DECJD'] <= dec_aliased_max)) else 'NO', axis=1)    
        
        # Creating two dataframes, one for sources within the FoV and another for sources outside
        df_psrcat_InField = df_psrcat.drop(df_psrcat[df_psrcat['In_Field'] == 'NO'].index)
        df_psrcat_InField['Observable'] = df_psrcat_InField.apply(lambda x: (CalcSNR(x['S400'], x['S1400'], x['W50'], x['P0'])) , axis=1)
        df_psrcat_InField.to_csv(f'{fitsfiles[i].split(".")[0]}_PSRCat_InField.csv')
        df_psrcat_NotInField = df_psrcat.drop(df_psrcat[df_psrcat['In_Field'] == 'YES'].index)
        df_psrcat_NotInField.to_csv(f'{fitsfiles[i].split(".")[0]}_PSRCat_NotInField.csv')
        
        # Calling the RACS query catalogue and adding a column based on whether the sources are within the FoV 
        df_racscat = pd.read_csv('RACSCat_Main.csv')
        
        # For cases when the field is crossing the RA = 0deg line, or when DEC is close to +-90deg, special conditions are applied
        if (ra_aliased_max < ra_aliased_min):
            df_racscat['In_Field'] = df_racscat.apply(lambda x: 'YES' if (((0 <= x['ra'] <= ra_aliased_max) & (ra_aliased_min <= x['ra'] <= 360)) 
                                                      & (dec_aliased_min <= x['dec'] <= dec_aliased_max)) else 'NO', axis=1)
        elif (dec_aliased_min == -90 or dec_aliased_max == +90):
            df_racscat['In_Field'] = df_racscat.apply(lambda x: 'YES' if (dec_aliased_min <= x['dec'] <= dec_aliased_max) else 'NO', axis=1)
        else:
            df_racscat['In_Field'] = df_racscat.apply(lambda x: 'YES' if ((ra_aliased_min <= x['ra'] <= ra_aliased_max) 
                                                      & (dec_aliased_min <= x['dec'] <= dec_aliased_max)) else 'NO', axis=1) 
        
        # Creating two dataframes, one for sources within the FoV and another for sources outside
        df_racscat_InField = df_racscat.drop(df_racscat[df_racscat['In_Field'] == 'NO'].index)
        df_racscat_InField.to_csv(f'{fitsfiles[i].split(".")[0]}_RACSCat_InField.csv')
        df_racscat_NotInField = df_racscat.drop(df_racscat[df_racscat['In_Field'] == 'YES'].index)
        df_racscat_NotInField.to_csv(f'{fitsfiles[i].split(".")[0]}_RACSCat_NotInField.csv')
    
    print(f'The candidate file "{fitsfiles[i].split(".")[0]}" has {df_psrcat_InField.shape[0]} pulsars and {df_racscat_InField.shape[0]} RACS sources within its field.')
    return df_psrcat_InField, df_psrcat_NotInField, df_racscat_InField, df_racscat_NotInField


# Calling the main function
if __name__=='__main__':
    args = get_parser().parse_args()
    QueryPSRCat(args.min_flux, args.ra_err, args.dec_err)
    QueryRACSCat(args.min_flux, args.ra_err, args.dec_err)
    CheckField()
    
    
    