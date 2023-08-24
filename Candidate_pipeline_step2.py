import numpy as np
import pandas as pd
from psrqpy import QueryATNF
from astropy.coordinates import SkyCoord
import astropy.units as u

def cross_matching(input_file, output_file, catalogue = None, threshold = 30):
    '''
    @author: Yu Wing Joshua Lee
    Important: upgrade numpy(1.25.2) and astropy(5.3.2) before running.
    Input_file and output_file are the path to the candidate list file.
    Threshold is in arcsecond.
    Catalogue should be in csv format.
    '''
    #Check numpy and astropy version
    numpy_required_version = "1.25.2"
    astropy_required_version = "5.3.2"
    if astropy.__version__ < astropy_required_version or np.__version__ < numpy_required_version:
        raise ImportError(f"Astropy version {astropy_required_version} and Numpy version {numpy_required_version} or higher are required.")
    
    #Read candidate list and adjust threshold unit
    df = pd.read_csv(input_file, index_col=None, sep='\t+',engine='python')
    candidates = df.drop(df.index[-1])
    threshold = threshold/3600
    
    #Read from psrcat if no catalogue is supplied. Generate a catalogue of pulsar around the beam center.
    #else convert the catalogue into pandas data frame.
    if catalogue == None:
        ra_mean = candidates["ra_deg"].mean()
        dec_mean = candidates["dec_deg"].mean()
        limit = "RAJD > " + str(ra_mean - 5) + "&& RAJD < " + str(ra_mean + 5) + "&& DECJD > " +str(dec_mean - 5) +"&& DECJD < " + str(dec_mean + 5)
        query = QueryATNF(params=["NAME", "RAJD", "DECJD"], condition=limit)
        pulsar_list = query.pandas
    else:
        pulsar_list = pd.read_csv(catalogue)
    
    #Cross-checking using astropy match_to_catalog_sky
    cand_radec = SkyCoord(ra=candidates['ra_deg'], dec=candidates['dec_deg'], unit=(u.degree, u.degree), frame='icrs')
    pulsar_radec = SkyCoord(ra = pulsar_list['RAJD'], dec=pulsar_list['DECJD'], unit=(u.degree, u.degree), frame = 'icrs')
    idx, sep2d, sep3d = cand_radec.match_to_catalog_sky(pulsar_radec)
    combined = [[idx, sep2d.deg] if sep2d.deg<threshold else[None, None] for idx, sep2d in zip(idx, sep2d)]
    
    #add column to candidate list and save as a new file
    new_columns = {'Name':[], 'Separation':[]}
    for index, pair in enumerate(combined):
        pulsar_name, pulsar_distance = pair
        new_columns['Name'].append(pulsar_list.loc[pulsar_name, 'NAME'] if pulsar_name is not None else None)
        new_columns['Separation'].append(pulsar_distance)
    new_df = pd.DataFrame(new_columns)
    result_df = pd.concat([candidates, new_df],axis=1)
    result_df.to_csv(output_file, index = False, sep='\t')
    print("Cross-matching completed.")
