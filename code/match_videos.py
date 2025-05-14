#%% objective
# match and copy videos used in the study from the original Moments in Time dataset

#%% packages
import os
import shutil
import pandas as pd


# %% function
def copy_videos_from_all_categories(path_MiT = '/Volumes/T9/Moments_in_Time_Raw_v2/Moments_in_Time_Raw', path_data = '../data'):
    """
    Match and copy videos from all categories in source directory
    """
    # Read train.csv and test.csv to get to-match video names
    train_df = pd.read_csv(f'{path_data}/annotations/train_categories.csv')
    test_df = pd.read_csv(f'{path_data}/annotations/test_categories.csv')
    train_names = train_df['video_name'].values
    test_names = test_df['video_name'].values
    match_names = list(train_names) + list(test_names)
    
    dest_dir = f'{path_data}/MiT_original_videos'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    category_counts = {}

    # Process training and validation folders
    for dataset in ['training', 'validation']:
        path_original_videos = f'{path_MiT}/{dataset}'
        categories = os.listdir(path_original_videos)

        # Process each category folder
        for category in categories:
            print(f'Processing category: {dataset}/{category}')
            source_dir = os.path.join(path_original_videos, category)
            if not os.path.isdir(source_dir):
                continue
                
            # Find matching videos in this category
            matching_videos = []
            for video in os.listdir(source_dir):
                if video in match_names:
                    matching_videos.append(video)
                    # Copy video to destination
                    src = os.path.join(source_dir, video)
                    dst = os.path.join(dest_dir, video)
                    shutil.copy2(src, dst)

            # Update category count by adding to existing count
            if category in category_counts:
                category_counts[category] += len(matching_videos)
            else:
                category_counts[category] = len(matching_videos)
                
            if len(matching_videos) > 0:
                print(f'found {len(matching_videos)} videos for {category}')
            else:
                pass

    # Print summary of matches found
    print("\nMatching videos found per category:")
    category_summary = []
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            category_summary.append(f"{category}: {count} videos")
    print("\n".join(category_summary))

    total = sum(category_counts.values())
    print(f"\nTotal videos copied: {total}")
    return category_summary


# %% match and copy videos from all categories
if __name__ == "__main__":
    category_summary = copy_videos_from_all_categories()
    with open('../data/video_metadata/category_summary.txt', 'w') as f:
        f.write("\n".join(category_summary))