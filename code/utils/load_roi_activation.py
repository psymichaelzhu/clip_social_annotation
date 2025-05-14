#%% objective
# extract neural activation from a region of interest
#%% packages
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

#%% functions
def get_localizer(RoI):
    """
    Find the localizer task name based on RoI (mapping from McMahon et al., 2023 https://osf.io/4j29y/wiki/home/)
    Args:
        RoI (str): targeted ROI
    Returns:
        str: The localizer task name
    """
    localizer_dir = {
        'biomotion': ['biomotion', 'MT'],
        'EVC': ['EVC'],
        'FBOS': ['EBA','face-pSTS','FFA','LOC','PPA'],
        'SIpSTS': ['aSTS','pSTS'],
        'tom': ['TPJ']
    }
    for dir_name, roi_list in localizer_dir.items():
        if RoI in roi_list:
            return dir_name
    return None

def load_neural_data(sub_id, phase, index, RoI=None, side=None, display=False, data_dir='../data/neural'):
    """Load beta data and RoI mask
    Args:
        sub_id (str): subject index, e.g. "01"
        phase (str): train or test, e.g. "train"
        index (str): odd or even (only for test phase, in which videos are presented twice), e.g. "odd"
        RoI (str): targeted ROI, e.g. "EVC"
        side (str): left or right, e.g. "r"
        display (bool): whether to display the beta data
    Returns:
        tuple: beta data (x, y, z, video) and RoI mask (x, y, z)
    """
    # Load beta data
    beta_file = f'{data_dir}/betas/sub-{sub_id}/sub-{sub_id}_space-T1w_desc-{phase}-fracridge{index}_data.nii.gz'
    beta_img = nib.load(beta_file)
    beta_data = beta_img.get_fdata()
    if display:
        print(beta_data.shape) # x, y, z, video
    
    # Load RoI mask if RoI and side are both provided
    mask_data = None
    if RoI is not None and side is not None:
        localizer = get_localizer(RoI)
        mask_file = f'{data_dir}/localizers/sub-{sub_id}/sub-{sub_id}_task-{localizer}_space-T1w_roi-{RoI}_hemi-{side}h_roi-mask.nii.gz'
        mask_img = nib.load(mask_file)
        mask_data = mask_img.get_fdata()
        if display:
            print(mask_data.shape) # x, y, z
    
    return beta_data, mask_data

def visualize_slices(beta_data, mask_data, video_id, sub_id, RoI=None, side=None, z_slices=range(25, 65, 10)):
    """Visualize several slices (along z-axis) of the beta data with RoI mask
    Args:
        beta_data (numpy.ndarray): beta data (x, y, z, video)
        mask_data (numpy.ndarray): RoI mask (x, y, z)
        video_id (int): video index
        sub_id (str): subject index
        RoI (str): targeted ROI
        side (str): left or right
        z_slices (range): range of z-slices to visualize
    """
    _, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    video_data = beta_data[:,:,:,video_id]
    
    # Get min and max values for the entire video volume to standardize the color map
    vmin = np.nanmin(video_data)
    vmax = np.nanmax(video_data)
    
    for i, z in enumerate(z_slices):
        demo_data = video_data[:,:,z]
        if mask_data is not None:
            masked_demo = np.where(mask_data[:,:,z] > 0, demo_data, np.nan)
            im = axes[i].imshow(demo_data, cmap='gray', vmin=vmin, vmax=vmax)
            im = axes[i].imshow(masked_demo, cmap='viridis', vmin=vmin, vmax=vmax)
            mask_slice = mask_data[:,:,z]
            axes[i].contour(mask_slice, levels=[0], colors='red', linewidths=2)
            roi_text = f'{side}-{RoI}'
        # Show full brain if no mask is provided
        else:
            im = axes[i].imshow(demo_data, cmap='viridis', vmin=vmin, vmax=vmax)
            roi_text = 'Full Brain'
            
        cbar = plt.colorbar(im, ax=axes[i], shrink=0.6)
        cbar.set_label('Beta Value')
        axes[i].set_title(f'Z-Slice {z}th Visualization\nvideo {video_id}, {roi_text}, subject {sub_id}')
        axes[i].set_xlabel('X axis')
        axes[i].set_ylabel('Y axis')
    plt.tight_layout()
    plt.show()

#%% example
if __name__ == '__main__':
    sub_id = '01'
    video_id =10
    phase = 'train'
    index = ""
    side = 'r'


    # whole brain
    beta_data, mask_data = load_data(sub_id, phase, index )
    visualize_slices(beta_data, mask_data, video_id, sub_id)

    # RoI
    RoI = 'EVC'
    beta_data, mask_data = load_data(sub_id, phase, index, RoI, side)
    visualize_slices(beta_data, mask_data, video_id, sub_id, RoI, side)

    RoI = 'TPJ'
    beta_data, mask_data = load_data(sub_id, phase, index, RoI, side)
    visualize_slices(beta_data, mask_data, video_id, sub_id, RoI, side)

