#%% objective
# organize neural activation data into a dataframe with video names
# check video mapping


#%% packages
import os
import numpy as np
import pandas as pd

from utils.load_roi_activation import load_neural_data

#%% functions
def get_single_roi_df(sub_id, phase, RoI, side, aggregated=False, display=True, index=''):
    """for each video and RoI, get (averaged) voxel-wise activation pattern
    Args:
        sub_id (str): subject index
        phase (str): train or test
        index (str): odd or even (only for test phase)
        RoI (str): targeted ROI
        side (str): left or right
        display (bool): whether to display data shapes
    Returns:
        pandas.DataFrame: (length: video x voxel) DataFrame containing ROI activations
    """
    # Load beta and RoI mask
    beta_data, mask_data = load_neural_data(sub_id=sub_id, phase=phase, index=index, 
                                   RoI=RoI, side=side, display=display)

    n_videos = beta_data.shape[3]
    
    # store activation in dataframe
    if aggregated:
        # If aggregated, average across voxels within ROI mask
        roi_activations = np.zeros(n_videos)
        for video in range(n_videos):
            video_data = beta_data[:,:,:,video]
            roi_activations[video] = np.nanmean(video_data[mask_data == 1])
            
        df = pd.DataFrame({
            'subj': [sub_id] * n_videos,
            'roi': [RoI] * n_videos, 
            'side': [side] * n_videos,
            'phase': [phase] * n_videos,
            'video': range(n_videos),
            'beta': roi_activations
        })
        
    else:
        # If not aggregated, keep all voxels within ROI mask
        roi_voxels = np.where(mask_data == 1)
        n_voxels = len(roi_voxels[0])
        
        all_activations = []
        for video in range(n_videos):
            video_data = beta_data[:,:,:,video]
            voxel_activations = video_data[roi_voxels]
            
            for voxel_idx in range(n_voxels):
                all_activations.append({
                    'subj': sub_id,
                    'roi': RoI,
                    'side': side,
                    'phase': phase, 
                    'video': video,
                    'voxel_id': voxel_idx,
                    'beta': voxel_activations[voxel_idx]
                })
                
        df = pd.DataFrame(all_activations)

    return df

def get_all_roi_df(aggregated=False):
    """Get activation dataframe for all subjects, phases, ROIs and sides
    Args:
        aggregated (bool): Whether to aggregate across voxels within ROI
    Returns:
        pandas.DataFrame: Combined DataFrame containing all activations
    """
    # Get list of all ROIs from localizer dictionary (McMahon et al., 2023)
    localizer_dir = {
        'biomotion': ['biomotion', 'MT'],
        'EVC': ['EVC'],
        'FBOS': ['EBA','face-pSTS','FFA','LOC','PPA'],
        'SIpSTS': ['aSTS','pSTS'],
        'tom': ['TPJ']
    }
    all_rois = [roi for rois in localizer_dir.values() for roi in rois]
    
    all_dfs = []
    
    call_counts = {}
    
    # Iterate through all combinations
    for sub_id in ['01', '02', '03', '04']:
        call_counts[f'sub-{sub_id}'] = 0
        print(f'sub-{sub_id}')
        for roi in all_rois:
            for side in ['l', 'r']:
                total_len = 0
                for phase in ['train', 'test']:
                    index = ''
                    try:
                        df = get_single_roi_df(
                            sub_id=sub_id,
                            phase=phase,
                            RoI=roi,
                            side=side,
                            index=index,
                            display=False,
                            aggregated=aggregated
                        )
                        all_dfs.append(df)
                        call_counts[f'sub-{sub_id}'] += 1
                        total_len += len(df)
                    except Exception as e:
                        print(f"Error processing sub-{sub_id}, {phase}, {roi}, {side}: {str(e)}")
                        break
                print(f'{side} {roi}: {int(total_len/250)} x 250') #250 videos
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print("\nFunction call counts per subject:")
    for subj, count in call_counts.items():
        print(f"{subj}: {count} calls")
        
    print("\nCombined DataFrame shape:", combined_df.shape)
    print("\nFirst few rows:")
    print(combined_df.head())
    return combined_df

def process_neural_data(combined_df, aggregated=False, output_dir='../data/neural'):
    """
    Further process the activation dataframe by:
    1. Merging with video names
    2. Sorting and cleaning the data
    Args:
        combined_df (pd.DataFrame): DataFrame containing neural data with columns:
            subj, roi, side, phase, video, voxel_id, beta
        aggregated (bool): Whether to aggregate across voxels within ROI
    Returns:
        pd.DataFrame: Processed DataFrame with video names
    """
    neural_df = combined_df.copy()
    
    # Sort before merging with video names
    neural_df = neural_df.sort_values(['subj', 'roi', 'side', 'voxel_id', 'phase', 'video']) if not aggregated else neural_df.sort_values(['subj', 'roi', 'side', 'phase', 'video'])

    # Merge with video names
    test_df = pd.read_csv('../data/video_metadata/test_categories.csv')
    train_df = pd.read_csv('../data/video_metadata/train_categories.csv')
    name_vec = pd.concat([test_df['video_name'], train_df['video_name']]).values

    # Create video name column
    n_repeats = len(neural_df) // len(name_vec)
    video_names = np.tile(name_vec, n_repeats)
    neural_df.insert(0, 'video_name', video_names)

    # Clean up columns
    neural_df = neural_df.drop(['phase', 'video'], axis=1)
    print("\nFinal neural DataFrame with video names:")
    print(neural_df.head())
    print("\nShape of final DataFrame:", neural_df.shape)

    # Save the processed DataFrame to a CSV file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = f'{output_dir}/{"voxel" if not aggregated else "roi"}_df.csv'
    neural_df.to_csv(output_path, index=False)
    print(f"\nSaved processed DataFrame to: {output_path}")

    return neural_df

def organize_neural_data(neural_df, output_dir='../data/neural'):
    """Organize neural data into nested dictionaries by subject, ROI and side
    Args:
        neural_df (pd.DataFrame): DataFrame containing neural data with columns:
            video_name, subj, roi, side, voxel_id, beta
    Returns:
        dict: Nested dictionary organized as {subj: {roi: {side: neural_embedding_df}}}
            where neural_embedding_df has videos as rows and voxels as columns
    """
    # Initialize the nested dictionary
    organized_data = {}
    
    # Build the nested structure
    for subj in neural_df['subj'].unique():
        # Map subject ID (e.g. '01') to 'sub1' format
        subj_num = int(subj)
        subj_name = f'sub{subj_num}'
        organized_data[subj_name] = {}
        subj_data = neural_df[neural_df['subj'] == subj]
        
        for roi in subj_data['roi'].unique():
            organized_data[subj][roi] = {}
            roi_data = subj_data[subj_data['roi'] == roi]
            
            for side in roi_data['side'].unique():
                subset = roi_data[roi_data['side'] == side]
                
                # Pivot the data to get videos as rows and voxels as columns
                neural_embedding = subset.pivot(
                    index='video_name',
                    columns='voxel_id',
                    values='beta'
                )
                
                neural_embedding.columns = [f'voxel_{i+1}' for i in range(len(neural_embedding.columns))]
                
                neural_embedding.reset_index(inplace=True)
                
                organized_data[subj][roi][side] = neural_embedding
    # Save the organized data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = f'{output_dir}/neural_for_rsm.npy'
    np.save(output_path, organized_data)
    print(f"\nSaved organized neural data to: {output_path}")
    return organized_data


#%% combine activation data across ROIs
if __name__ == '__main__':
    #voxel summary
    combined_df_voxel = get_all_roi_df(aggregated=False)
    #roi summary
    #combined_df_roi = go_through_all_rois(aggregated=True)

    # process neural data and save to csv
    neural_df_voxel = process_neural_data(combined_df_voxel, aggregated=False)
    #neural_df_roi = process_neural_data(combined_df_roi, aggregated=True)

    # organize neural data: nested dictionary
    neural_embedding = organize_neural_data(neural_df_voxel)
