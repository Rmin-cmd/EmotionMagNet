# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import scipy.io as sio
# # import os
# # from glob import glob
# # import re
# # from tqdm import tqdm
# #
# #
# # def read_mat_file(mat_path):
# #
# #     if mat_path.endswith('.mat'):
# #
# #         arr = sio.loadmat(mat_path)['DTF']['matrix'][0][0]
# #
# #         arr_norm = np.mean(arr, axis=2)
# #         #
# #         arr_norm /= np.max(arr_norm, axis=0)
# #
# #         np.fill_diagonal(arr_norm, 0)
# #
# #     return arr_norm
# #
# #
# # def str_num(str):
# #     number = re.search(r'\d+', str)
# #     return int(number.group())
# #
# #
# # def matfile_loader(data_path, vid_ord):
# #
# #     data_out = []
# #
# #     for vid in tqdm(vid_ord):
# #
# #         paths = sorted(glob(os.path.join(data_path, vid) + '\*'))
# #
# #         freq_sub_data = []
# #
# #         for path in paths:
# #
# #             path_ = sorted(glob(path + '\*'))
# #
# #             subs = np.unique([os.path.basename(path)[:6] for path in path_])
# #
# #             sub_data = []
# #
# #             for sub in subs:
# #
# #                 specified_ = [sub in path for path in path_]
# #
# #                 specified_paths = [item for item, flag in zip(path_, specified_) if flag]
# #
# #                 unique_vids = np.unique([str_num(os.path.basename(path_sp)[7:9]) for path_sp in specified_paths]).tolist()
# #
# #                 data_sub = np.array([read_mat_file(path_data) for path_data in specified_paths])
# #
# #                 sub_data.append(data_sub.reshape([len(unique_vids), data_sub.shape[0] // len(unique_vids),
# #                                                   data_sub.shape[1], data_sub.shape[2]]))
# #
# #             freq_sub_data.append(sub_data)
# #
# #         data_out.append(np.array(freq_sub_data).swapaxes(1, 0))
# #
# #     return np.concatenate(data_out, axis=2)
# #
# #
# # # path_root = 'D:\proposal and thesis\Emotion recognition dataset\FACED dataset\Clisa_data\PDC_connectivity_30'
# # path_root = 'D:\proposal and thesis\Emotion recognition dataset\FACED dataset\Clisa_data\PDC_connectivity_30_mod'
# #
# # out_path = 'preprocessed_data\preprocessed_connectivity\processed_conn_30_mod_3.mat'
# #
# # vid_order = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Neutral', 'Amusement', 'Inspiration', 'Joy', 'Tenderness']
# #
# # out = matfile_loader(data_path=path_root, vid_ord=vid_order)
# #
# # dic = {'data':out, 'label':'FACED_PDC'}
# #
# # sio.savemat(out_path, dic)
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import scipy.io as sio
# # import os
# # from glob import glob
# # import re
# # from tqdm import tqdm
# # import networkx as nx
# #
# #
# # def read_mat_file(mat_path):
# #     # Load and normalize connectivity
# #     arr = sio.loadmat(mat_path)['DTF']['matrix'][0][0]
# #     arr_norm = np.mean(arr, axis=2)
# #     max_vals = np.max(arr_norm, axis=0)
# #     max_vals[max_vals == 0] = 1e-12  # avoid division by zero
# #     arr_norm /= max_vals
# #     np.fill_diagonal(arr_norm, 0)
# #
# #     # Apply directed OMST thresholding
# #     edges = compute_directed_omst(arr_norm, max_rounds=3)
# #     arr_thresh = np.zeros_like(arr_norm)
# #     for (u, v) in edges:
# #         arr_thresh[u, v] = arr_norm[u, v]
# #     return arr_thresh
# #
# #
# # def str_num(str_input):
# #     number = re.search(r'\d+', str_input)
# #     return int(number.group())
# #
# #
# # def directed_global_efficiency(G):
# #     n = len(G)
# #     path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
# #     inv_sum = sum(1.0 / d for u in path_lengths for v, d in path_lengths[u].items() if u != v and d > 0)
# #     return inv_sum / (n * (n - 1)) if n > 1 else 0.0
# #
# #
# # def compute_directed_omst(adj, max_rounds=3):
# #     """
# #     Directed OMST extraction using Edmonds' algorithm, optimized by GCE and leaf fraction penalty.
# #     Returns a set of directed edges representing the thresholded backbone.
# #     """
# #     n = adj.shape[0]
# #     total_strength = adj.sum()
# #     best_edges = set()
# #     best_score = -np.inf
# #
# #     adj_copy = adj.copy()
# #     cum_edges = set()
# #
# #     for m in range(1, max_rounds + 1):
# #         G = nx.DiGraph()
# #         for i in range(n):
# #             for j in range(n):
# #                 if adj_copy[i, j] > 0:
# #                     G.add_edge(i, j, weight=1.0 / adj_copy[i, j])
# #         if G.number_of_edges() == 0:
# #             break
# #
# #         ed = nx.algorithms.tree.branchings.Edmonds(G)
# #         try:
# #             arb = ed.find_optimum(attr='weight', default=0.0, kind='min', style='arborescence')
# #         except Exception:
# #             break
# #
# #         edges = set(arb.edges())
# #         if not edges:
# #             break
# #         cum_edges |= edges
# #         for (u, v) in edges:
# #             adj_copy[u, v] = 0.0
# #
# #         H = nx.DiGraph()
# #         for (u, v) in cum_edges:
# #             H.add_edge(u, v, weight=adj[u, v])
# #
# #         try:
# #             ge = directed_global_efficiency(H)
# #         except Exception:
# #             ge = 0.0
# #
# #         cost = sum(adj[u, v] for u, v in cum_edges) / total_strength if total_strength > 0 else 0.0
# #         degrees = dict(H.degree())
# #         leaf_count = sum(1 for deg in degrees.values() if deg == 1)
# #         leaf_frac = leaf_count / float(n)
# #
# #         score = ge - cost - 0.5 * leaf_frac  # Weighted penalty on star structure
# #
# #         if score > best_score:
# #             best_score = score
# #             best_edges = cum_edges.copy()
# #
# #     if not best_edges:
# #         sym = (adj + adj.T) / 2.0
# #         U = nx.Graph()
# #         for i in range(n):
# #             for j in range(i + 1, n):
# #                 if sym[i, j] > 0:
# #                     U.add_edge(i, j, weight=1.0 / sym[i, j])
# #         T = nx.minimum_spanning_tree(U, weight='weight')
# #         for u, v in T.edges():
# #             if adj[u, v] >= adj[v, u]:
# #                 best_edges.add((u, v))
# #             else:
# #                 best_edges.add((v, u))
# #
# #     return best_edges
# #
# #
# # def matfile_loader(data_path, vid_ord):
# #     data_out = []
# #     for vid in tqdm(vid_ord):
# #         paths = sorted(glob(os.path.join(data_path, vid) + os.sep + '*'))
# #         freq_sub_data = []
# #
# #         for path in tqdm(paths):
# #             path_ = sorted(glob(path + os.sep + '*'))
# #             subs = np.unique([os.path.basename(p)[:6] for p in path_])
# #             sub_data = []
# #
# #             for sub in subs:
# #                 specified_paths = [p for p in path_ if sub in p]
# #                 data_sub = np.array([read_mat_file(p) for p in specified_paths])
# #                 vids = np.unique([str_num(os.path.basename(p)[7:9]) for p in specified_paths])
# #                 num_samples, C1, C2 = data_sub.shape
# #                 num_vids = len(vids)
# #                 num_windows = num_samples // num_vids
# #                 reshaped = data_sub.reshape(num_vids, num_windows, C1, C2)
# #                 sub_data.append(reshaped)
# #
# #             freq_sub_data.append(sub_data)
# #
# #         freq_sub_data = np.array(freq_sub_data, dtype=object)
# #         data_out.append(np.stack(freq_sub_data, axis=2))
# #
# #     return np.concatenate(data_out, axis=2)
# #
# #
# # if __name__ == '__main__':
# #     path_root = r'D:\proposal and thesis\Emotion recognition dataset\FACED dataset\Clisa_data\PDC_connectivity_30_mod'
# #     out_path = r'preprocessed_data\preprocessed_connectivity\processed_conn_30_mod_3_omst.mat'
# #     vid_order = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Neutral', 'Amusement', 'Inspiration', 'Joy', 'Tenderness']
# #
# #     out = matfile_loader(data_path=path_root, vid_ord=vid_order)
# #     sio.savemat(out_path, {'data': out, 'label': 'FACED_PDC_OMST'})
# #     print(f"Saved thresholded connectivity to {out_path}")
# import numpy as np
# import scipy.io as sio
# import os
# from glob import glob
# from tqdm import tqdm
# from concurrent.futures import ProcessPoolExecutor, as_completed
#
# # Import functions from the new processing_utils.py file
# from processing_utils import _process_subject_data, str_num  # Import any other functions directly used in main_processing.py if needed
#
#
# def matfile_loader(data_path, vid_ord, num_workers=None):
#     """
#     Loads and processes .mat files in parallel for different video orders and subjects.
#     """
#     data_out = []
#
#     if num_workers is None:
#         num_workers = os.cpu_count() or 1
#
#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         for vid in tqdm(vid_ord, desc="Processing Video Orders"):
#             paths = sorted(glob(os.path.join(data_path, vid) + os.sep + '*'))
#             freq_sub_data = []
#
#             for path_group_dir in tqdm(paths, desc=f"  Processing Freq/Sub Dirs for {vid}", leave=False):
#                 path_ = sorted(glob(path_group_dir + os.sep + '*'))
#
#                 sub_paths_map = {}
#                 for p in path_:
#                     sub = os.path.basename(p)[:6]
#                     if sub not in sub_paths_map:
#                         sub_paths_map[sub] = []
#                     sub_paths_map[sub].append(p)
#
#                 futures = {executor.submit(_process_subject_data, sub_paths): sub_id for sub_id, sub_paths in
#                            sub_paths_map.items()}
#
#                 # Get subjects sorted by their appearance in sub_paths_map for consistent ordering
#                 sorted_sub_ids = sorted(sub_paths_map.keys())
#                 # Initialize results list with Nones, maintaining order
#                 sub_data_results = [None] * len(sorted_sub_ids)
#
#                 for future in tqdm(as_completed(futures), total=len(futures),
#                                    desc=f"    Processing Subjects for {os.path.basename(path_group_dir)}", leave=False):
#                     sub_id = futures[future]
#                     try:
#                         result = future.result()
#                         if result is not None:
#                             original_index = sorted_sub_ids.index(sub_id)
#                             sub_data_results[original_index] = result
#                     except Exception as exc:
#                         print(f"Error processing subject {sub_id} from {path_group_dir}: {exc}")
#                 valid_sub_data = [res for res in sub_data_results if res is not None]
#
#                 if valid_sub_data:
#                     freq_sub_data.append(np.stack(valid_sub_data, axis=2))
#
#             if freq_sub_data:
#                 freq_sub_data_stacked = np.stack(freq_sub_data, axis=2)
#                 data_out.append(freq_sub_data_stacked)
#             else:
#                 print(f"Warning: No valid frequency/subject data found for video order: {vid}")
#
#     if data_out:
#         return np.concatenate(data_out, axis=2)
#     else:
#         print("Warning: No data generated for any video order.")
#         return np.array([])
#
#
# if __name__ == '__main__':
#     path_root = r'D:\proposal and thesis\Emotion recognition dataset\FACED dataset\Clisa_data\PDC_connectivity_30_mod'
#     out_path = r'preprocessed_data\preprocessed_connectivity\processed_conn_30_mod_3_omst.mat'
#     vid_order = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Neutral', 'Amusement', 'Inspiration', 'Joy', 'Tenderness']
#
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#
#     print(f"Starting data loading and processing with {os.cpu_count()} workers...")
#     out = matfile_loader(data_path=path_root, vid_ord=vid_order, num_workers=4)
#
#     if out.size > 0:
#         sio.savemat(out_path, {'data': out, 'label': 'FACED_PDC_OMST'})
#         print(f"Saved thresholded connectivity to {out_path}")
#     else:
#         print("No data was processed and saved.")


# main_processing.py

import numpy as np
import scipy.io as sio
import os
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import functions and global constants from the new processing_utils.py file
from processing_utils import _process_subject_data, str_num, \
    EXPECTED_C_DIM, EXPECTED_NUM_WINDOWS_PER_VIDEO, EXPECTED_NUM_FREQ_BANDS


def matfile_loader(data_path, vid_ord, expected_video_counts_by_emotion, num_workers=None):  # Added new parameter
    """
    Loads and processes .mat files in parallel for different video orders and subjects.
    """
    data_out = []

    if num_workers is None:
        num_workers = os.cpu_count() or 1

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for vid in tqdm(vid_ord, desc="Processing Video Orders"):

            # Get the expected number of videos for the current emotion
            expected_videos_for_current_emotion = expected_video_counts_by_emotion.get(vid)
            if expected_videos_for_current_emotion is None:
                raise ValueError(
                    f"Emotion '{vid}' not found in expected_video_counts_by_emotion map. Please define its expected video count.")

            paths = sorted(glob(os.path.join(data_path, vid) + os.sep + '*'))

            processed_data_for_all_freq_bands = []

            for path in tqdm(paths, desc=f"  Processing Freq/Sub Dirs for {vid}", leave=False):
                path_ = sorted(glob(path + os.sep + '*'))

                sub_paths_map = {}
                for p in path_:
                    sub_id = os.path.basename(p)[:6]
                    if sub_id not in sub_paths_map:
                        sub_paths_map[sub_id] = []
                    sub_paths_map[sub_id].append(p)
                sorted_subs = sorted(sub_paths_map.keys())

                # Submit each subject's data processing to the executor
                # Pass the expected_videos_for_current_emotion to the worker function
                futures = {
                    executor.submit(_process_subject_data, sub_paths, expected_videos_for_current_emotion): sub_id
                    for sub_id, sub_paths in sub_paths_map.items()}

                sub_data_results = [None] * len(sorted_subs)  # Pre-allocate list for ordered results

                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f"    Processing Subjects for {os.path.basename(path)}", leave=False):
                    sub_id = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            original_index = sorted_subs.index(sub_id)
                            sub_data_results[original_index] = result
                    except Exception as exc:
                        print(f"Error processing subject {sub_id} from {os.path.basename(path)}: {exc}")

                valid_sub_data = [res for res in sub_data_results if res is not None]

                if not valid_sub_data:
                    # print(f"Warning: No valid subject data collected for freq band: {path}. Skipping.")
                    continue

                stacked_subjects_for_this_freq = np.array(valid_sub_data)
                processed_data_for_all_freq_bands.append(stacked_subjects_for_this_freq)

            if not processed_data_for_all_freq_bands:
                print(f"Warning: No valid frequency band data processed for video order: {vid}. Skipping.")
                continue

            temp_stacked_by_freq = np.array(processed_data_for_all_freq_bands)

            # --- ASSERTION 3: Check number of frequency bands ---
            assert temp_stacked_by_freq.shape[0] == EXPECTED_NUM_FREQ_BANDS, \
                f"Assertion Error: Video order '{vid}' has {temp_stacked_by_freq.shape[0]} frequency bands, expected {EXPECTED_NUM_FREQ_BANDS}."

            final_stacked_for_vid = temp_stacked_by_freq.swapaxes(1, 0)

            data_out.append(final_stacked_for_vid)

    if not data_out:
        print("Warning: No data generated for any video order. Returning empty array.")
        return np.array([])

    # --- ASSERTION 4: Check total number of video orders (emotions) ---
    assert len(data_out) == len(vid_ord), \
        f"Assertion Error: Expected {len(vid_ord)} video orders, but collected {len(data_out)}."

    # Final concatenate: `data_out` contains elements of shape
    # (num_subjects, EXPECTED_NUM_FREQ_BANDS, EXPECTED_NUM_VIDEOS_PER_SUBJECT, EXPECTED_NUM_WINDOWS_PER_VIDEO, EXPECTED_C_DIM, EXPECTED_C_DIM)
    # The `EXPECTED_NUM_VIDEOS_PER_SUBJECT` here refers to the expected video count for the *specific emotion* for that item in `data_out`.
    # For `np.concatenate(data_out, axis=2)` to work, the shapes must match on all axes *except* axis 2.
    # This implies that `num_subjects`, `EXPECTED_NUM_FREQ_BANDS`, `EXPECTED_NUM_WINDOWS_PER_VIDEO`, `EXPECTED_C_DIM` must all be consistent across *all emotions*.
    # The axis=2 (number of videos) will sum up.

    if len(data_out) > 1:
        # Check consistency for the final concatenation.
        # This check is crucial because the video dimension is now variable per emotion.
        # The common dimensions must be consistent for np.concatenate(axis=2)
        first_item_shape = data_out[0].shape
        for i, item in enumerate(data_out):
            # Check all dimensions except the concatenation axis (axis=2)
            # Dims: (num_subjects, num_freq_bands, NUM_VIDEOS_VARIES, num_windows, C1, C2)
            if not (item.shape[0] == first_item_shape[0] and  # num_subjects
                    item.shape[1] == first_item_shape[1] and  # num_freq_bands
                    item.shape[3] == first_item_shape[3] and  # num_windows
                    item.shape[4] == first_item_shape[4] and  # C1
                    item.shape[5] == first_item_shape[5]):  # C2
                print(f"Error: Inconsistent shapes in data_out for final concatenation. "
                      f"Item {i} shape: {item.shape}, First item shape: {first_item_shape}. "
                      f"Cannot concatenate along axis=2. Returning empty array.")
                return np.array([])

    return np.concatenate(data_out, axis=2)


# --- Main execution block ---
if __name__ == '__main__':
    # path_root = r'D:\proposal and thesis\Emotion recognition dataset\FACED dataset\Clisa_data\PDC_connectivity_30_mod'
    path_root = os.path.join(os.getcwd(), 'PDC_connectivity_30_mod')
    out_path = os.path.join(os.getcwd(), 'PDC_connectivity_30_mod.mat')

    vid_order = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Neutral', 'Amusement', 'Inspiration', 'Joy', 'Tenderness']

    # Define the expected video counts per emotion
    EXPECTED_VIDEO_COUNTS_BY_EMOTION = {
        'Anger': 3,
        'Disgust': 3,
        'Fear': 3,
        'Sadness': 3,
        'Neutral': 4,
        'Amusement': 3,
        'Inspiration': 3,
        'Joy': 3,
        'Tenderness': 3
    }

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Starting data loading and processing with dynamic video count assertions...")
    try:
        out = matfile_loader(data_path=path_root, vid_ord=vid_order,
                             expected_video_counts_by_emotion=EXPECTED_VIDEO_COUNTS_BY_EMOTION, num_workers=4)

        if out.size > 0:  # Check if the array is not empty
            sio.savemat(out_path, {'data': out, 'label': 'FACED_PDC'})
            print(f"Saved thresholded connectivity to {out_path}")
        else:
            print("No data was processed and saved due to warnings/errors.")
    except AssertionError as e:
        print(f"Preprocessing failed due to an assertion error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")

