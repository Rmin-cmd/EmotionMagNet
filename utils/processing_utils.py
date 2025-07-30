# # # processing_utils.py
# #
# # import numpy as np
# # import scipy.io as sio
# # import os
# # import re
# # import networkx as nx
# #
# #
# # # Define all helper functions here
# # # --------------------------------------------------------------------------
# #
# # def read_mat_file(mat_path):
# #     """
# #     Loads a .mat file, normalizes the connectivity matrix,
# #     and applies directed OMST thresholding.
# #     """
# #     arr = sio.loadmat(mat_path)['DTF']['matrix'][0][0]
# #     arr_norm = np.mean(arr, axis=2)
# #     max_vals = np.max(arr_norm, axis=0)
# #     max_vals[max_vals == 0] = 1e-12
# #     arr_norm /= max_vals
# #     np.fill_diagonal(arr_norm, 0)
# #
# #     edges = compute_directed_omst(arr_norm, max_rounds=3)
# #     arr_thresh = np.zeros_like(arr_norm)
# #
# #     if edges:
# #         edges_array = np.array(list(edges))
# #         arr_thresh[edges_array[:, 0], edges_array[:, 1]] = arr_norm[edges_array[:, 0], edges_array[:, 1]]
# #     return arr_thresh
# #
# #
# # def str_num(str_input):
# #     """Extracts the first number from a string."""
# #     number = re.search(r'\d+', str_input)
# #     if number:
# #         return int(number.group())
# #     return None
# #
# #
# # def directed_global_efficiency(G):
# #     """Calculates the directed global efficiency of a graph."""
# #     n = len(G)
# #     if n <= 1:
# #         return 0.0
# #     path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
# #     inv_sum = sum(1.0 / d for u in path_lengths for v, d in path_lengths[u].items() if u != v and d > 0)
# #     return inv_sum / (n * (n - 1))
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
# #         rows, cols = np.where(adj_copy > 0)
# #         for i, j in zip(rows, cols):
# #             G.add_edge(i, j, weight=1.0 / adj_copy[i, j])
# #
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
# #
# #         if edges:
# #             edges_to_zero = np.array(list(edges))
# #             valid_rows = edges_to_zero[:, 0] < adj_copy.shape[0]
# #             valid_cols = edges_to_zero[:, 1] < adj_copy.shape[1]
# #             valid_indices = valid_rows & valid_cols
# #             if np.any(valid_indices):
# #                 adj_copy[edges_to_zero[valid_indices, 0], edges_to_zero[valid_indices, 1]] = 0.0
# #
# #         H = nx.DiGraph()
# #         if cum_edges:
# #             cum_edges_list = list(cum_edges)
# #             for u, v in cum_edges_list:
# #                 H.add_edge(u, v, weight=adj[u, v])
# #
# #         try:
# #             ge = directed_global_efficiency(H)
# #         except Exception:
# #             ge = 0.0
# #
# #         cost = sum(adj[u, v] for u, v in cum_edges) / total_strength if total_strength > 0 else 0.0
# #
# #         if H.number_of_nodes() > 0:
# #             degrees = dict(H.degree())
# #             leaf_count = sum(1 for deg in degrees.values() if deg == 1)
# #             leaf_frac = leaf_count / float(n)
# #         else:
# #             leaf_frac = 0.0
# #
# #         score = ge - cost - 0.5 * leaf_frac
# #
# #         if score > best_score:
# #             best_score = score
# #             best_edges = cum_edges.copy()
# #
# #     if not best_edges:
# #         sym = (adj + adj.T) / 2.0
# #         U = nx.Graph()
# #         rows, cols = np.where(sym > 0)
# #         for i, j in zip(rows, cols):
# #             if i < j:
# #                 U.add_edge(i, j, weight=1.0 / sym[i, j])
# #
# #         if U.number_of_edges() > 0:
# #             try:
# #                 T = nx.minimum_spanning_tree(U, weight='weight')
# #                 for u, v in T.edges():
# #                     if adj[u, v] >= adj[v, u]:
# #                         best_edges.add((u, v))
# #                     else:
# #                         best_edges.add((v, u))
# #             except nx.NetworkXNoCycle:
# #                 pass
# #             except Exception as e:
# #                 pass
# #
# #     return best_edges
# #
# #
# # def _process_subject_data(subject_paths):
# #     """
# #     Processes all mat files for a single subject and reshapes them.
# #     This function is designed to be run in parallel.
# #     """
# #     data_sub_list = []
# #     vids_in_sub = set()
# #
# #     for p in subject_paths:
# #         data = read_mat_file(p)  # Calls read_mat_file from this module
# #         data_sub_list.append(data)
# #
# #         vid_num = str_num(os.path.basename(p)[7:9])
# #         if vid_num is not None:
# #             vids_in_sub.add(vid_num)
# #
# #     data_sub = np.array(data_sub_list)
# #     vids_in_sub = sorted(list(vids_in_sub))
# #
# #     num_samples, C1, C2 = data_sub.shape
# #     num_vids = len(vids_in_sub)
# #
# #     if num_vids > 0 and num_samples % num_vids == 0:
# #         num_windows = num_samples // num_vids
# #         reshaped = data_sub.reshape(num_vids, num_windows, C1, C2)
# #         return reshaped
# #     else:
# #         return None
# # processing_utils.py
#
# import numpy as np
# import scipy.io as sio
# import os
# import re
# import networkx as nx
#
# # Define expected dimensions as constants for assertions
# # These constants are global to this module, so functions within it can access them.
# EXPECTED_C_DIM = 30 # For the adjacency matrix (30x30)
# EXPECTED_NUM_VIDEOS_PER_SUBJECT = 28
# EXPECTED_NUM_WINDOWS_PER_VIDEO = 11
# EXPECTED_NUM_FREQ_BANDS = 5 # This will be asserted in matfile_loader, not here
#
# # Define all helper functions here
# # --------------------------------------------------------------------------
#
# def read_mat_file(mat_path):
#     """
#     Loads a .mat file, normalizes the connectivity matrix,
#     and applies directed OMST thresholding.
#     Includes an assertion for the expected 30x30 matrix shape.
#     """
#     # Defensive check for file existence and type
#     if not os.path.exists(mat_path):
#         raise FileNotFoundError(f"Matrix file not found: {mat_path}")
#     if not mat_path.endswith('.mat'):
#         raise ValueError(f"File {mat_path} is not a .mat file.")
#
#     try:
#         arr = sio.loadmat(mat_path)['DTF']['matrix'][0][0]
#     except (KeyError, IndexError) as e:
#         raise ValueError(f"Error extracting 'DTF' matrix from {mat_path}: {e}")
#     except Exception as e:
#         raise ValueError(f"Error loading .mat file {mat_path}: {e}")
#
#     arr_norm = np.mean(arr, axis=2)
#
#     # --- ASSERTION 1: Check C1, C2 dimensions ---
#     assert arr_norm.shape == (EXPECTED_C_DIM, EXPECTED_C_DIM), \
#         f"Assertion Error: read_mat_file expected ({EXPECTED_C_DIM},{EXPECTED_C_DIM}) matrix, but got {arr_norm.shape} from {mat_path}"
#
#     max_vals = np.max(arr_norm, axis=0)
#     # Add the safety check for division by zero
#     max_vals[max_vals == 0] = 1e-12
#     arr_norm /= max_vals
#     np.fill_diagonal(arr_norm, 0)
#
#     edges = compute_directed_omst(arr_norm, max_rounds=3)
#     arr_thresh = np.zeros_like(arr_norm)
#
#     if edges:
#         edges_array = np.array(list(edges))
#         arr_thresh[edges_array[:, 0], edges_array[:, 1]] = arr_norm[edges_array[:, 0], edges_array[:, 1]]
#     return arr_thresh
#
#
# def str_num(str_input):
#     """Extracts the first number from a string."""
#     number = re.search(r'\d+', str_input)
#     if number:
#         return int(number.group())
#     return None
#
#
# def directed_global_efficiency(G):
#     """Calculates the directed global efficiency of a graph."""
#     n = len(G)
#     if n <= 1:
#         return 0.0
#     path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
#     inv_sum = sum(1.0 / d for u in path_lengths for v, d in path_lengths[u].items() if u != v and d > 0)
#     return inv_sum / (n * (n - 1))
#
#
# def compute_directed_omst(adj, max_rounds=3):
#     """
#     Directed OMST extraction using Edmonds' algorithm, optimized by GCE and leaf fraction penalty.
#     Returns a set of directed edges representing the thresholded backbone.
#     """
#     n = adj.shape[0]
#     total_strength = adj.sum()
#     best_edges = set()
#     best_score = -np.inf
#
#     adj_copy = adj.copy()
#     cum_edges = set()
#
#     for m in range(1, max_rounds + 1):
#         G = nx.DiGraph()
#         rows, cols = np.where(adj_copy > 0)
#         for i, j in zip(rows, cols):
#             G.add_edge(i, j, weight=1.0 / adj_copy[i, j])
#
#         if G.number_of_edges() == 0:
#             break
#
#         ed = nx.algorithms.tree.branchings.Edmonds(G)
#         try:
#             arb = ed.find_optimum(attr='weight', default=0.0, kind='min', style='arborescence')
#         except Exception:
#             # print(f"Warning: Edmonds' algorithm failed in round {m}. Breaking.")
#             break
#
#         edges = set(arb.edges())
#         if not edges:
#             break
#         cum_edges |= edges
#
#         if edges:
#             edges_to_zero = np.array(list(edges))
#             valid_rows = edges_to_zero[:, 0] < adj_copy.shape[0]
#             valid_cols = edges_to_zero[:, 1] < adj_copy.shape[1]
#             valid_indices = valid_rows & valid_cols
#             if np.any(valid_indices):
#                 adj_copy[edges_to_zero[valid_indices, 0], edges_to_zero[valid_indices, 1]] = 0.0
#
#         H = nx.DiGraph()
#         if cum_edges:
#             cum_edges_list = list(cum_edges)
#             for u, v in cum_edges_list:
#                 H.add_edge(u, v, weight=adj[u, v])
#
#         try:
#             ge = directed_global_efficiency(H)
#         except Exception:
#             ge = 0.0
#
#         cost = sum(adj[u, v] for u, v in cum_edges) / total_strength if total_strength > 0 else 0.0
#
#         if H.number_of_nodes() > 0:
#             degrees = dict(H.degree())
#             leaf_count = sum(1 for deg in degrees.values() if deg == 1)
#             leaf_frac = leaf_count / float(n)
#         else:
#             leaf_frac = 0.0
#
#         score = ge - cost - 0.5 * leaf_frac
#
#         if score > best_score:
#             best_score = score
#             best_edges = cum_edges.copy()
#
#     if not best_edges:
#         sym = (adj + adj.T) / 2.0
#         U = nx.Graph()
#         rows, cols = np.where(sym > 0)
#         for i, j in zip(rows, cols):
#             if i < j:
#                 U.add_edge(i, j, weight=1.0 / sym[i, j])
#
#         if U.number_of_edges() > 0:
#             try:
#                 T = nx.minimum_spanning_tree(U, weight='weight')
#                 for u, v in T.edges():
#                     if adj[u, v] >= adj[v, u]:
#                         best_edges.add((u, v))
#                     else:
#                         best_edges.add((v, u))
#             except nx.NetworkXNoCycle:
#                 pass
#             except Exception as e:
#                 pass
#
#     return best_edges
#
#
# def _process_subject_data(subject_paths):
#     """
#     Processes all mat files for a single subject and reshapes them.
#     This function is designed to be run in parallel.
#     Includes assertions for the expected number of videos and windows per video.
#     """
#     data_sub_list = []
#     vids_in_sub = set()
#
#     for p in subject_paths:
#         try:
#             data = read_mat_file(p)  # Calls read_mat_file from this module
#             data_sub_list.append(data)
#             vid_num = str_num(os.path.basename(p)[7:9])
#             if vid_num is not None:
#                 vids_in_sub.add(vid_num)
#         except (FileNotFoundError, ValueError) as e:
#             # print(f"Error processing file {p}: {e}. Skipping this file for subject.")
#             pass # Continue processing other files for this subject
#
#     if not data_sub_list: # If no valid data was collected for this subject
#         return None
#
#     data_sub = np.array(data_sub_list)
#     vids_in_sub = sorted(list(vids_in_sub))
#
#     num_samples, C1, C2 = data_sub.shape
#     num_vids = len(vids_in_sub)
#
#     # --- ASSERTION 2a: Check number of videos ---
#     assert num_vids == EXPECTED_NUM_VIDEOS_PER_SUBJECT, \
#         f"Assertion Error: Subject data from '{subject_paths[0]}' has {num_vids} videos, expected {EXPECTED_NUM_VIDEOS_PER_SUBJECT}."
#
#     # Validate divisibility before calculating num_windows
#     if num_samples % num_vids != 0:
#         # This will be caught by the next assertion too, but good to have explicit handling
#         # print(f"Warning: Number of samples ({num_samples}) not divisible by number of unique videos ({num_vids}). Returning None.")
#         return None # Indicate invalid data for this subject
#
#     num_windows = num_samples // num_vids
#
#     # --- ASSERTION 2b: Check number of windows per video ---
#     assert num_windows == EXPECTED_NUM_WINDOWS_PER_VIDEO, \
#         f"Assertion Error: Subject data from '{subject_paths[0]}' has {num_windows} windows per video, expected {EXPECTED_NUM_WINDOWS_PER_VIDEO}."
#
#     # Validate C1, C2, though already checked in read_mat_file
#     assert C1 == EXPECTED_C_DIM and C2 == EXPECTED_C_DIM, \
#         f"Assertion Error: Subject data from '{subject_paths[0]}' has matrix dimensions ({C1},{C2}), expected ({EXPECTED_C_DIM},{EXPECTED_C_DIM})."
#
#     reshaped = data_sub.reshape(num_vids, num_windows, C1, C2)
#     return reshaped


# processing_utils.py

import numpy as np
import scipy.io as sio
import os
import re
import networkx as nx

# Define expected dimensions as constants for assertions
EXPECTED_C_DIM = 30 # For the adjacency matrix (30x30)
# EXPECTED_NUM_VIDEOS_PER_SUBJECT is now dynamic, removed from global constants
EXPECTED_NUM_WINDOWS_PER_VIDEO = 11
EXPECTED_NUM_FREQ_BANDS = 5 # Still global as it applies to all subjects/emotions in the context of one processing run


# Define all helper functions here
# --------------------------------------------------------------------------

def read_mat_file(mat_path):
    """
    Loads a .mat file, normalizes the connectivity matrix,
    and applies directed OMST thresholding.
    Includes an assertion for the expected 30x30 matrix shape.
    """
    # Defensive check for file existence and type
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Matrix file not found: {mat_path}")
    if not mat_path.endswith('.mat'):
        raise ValueError(f"File {mat_path} is not a .mat file.")

    try:
        arr = sio.loadmat(mat_path)['DTF']['matrix'][0][0]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Error extracting 'DTF' matrix from {mat_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading .mat file {mat_path}: {e}")

    arr_norm = np.mean(arr, axis=2)

    # --- ASSERTION 1: Check C1, C2 dimensions ---
    assert arr_norm.shape == (EXPECTED_C_DIM, EXPECTED_C_DIM), \
        f"Assertion Error: read_mat_file expected ({EXPECTED_C_DIM},{EXPECTED_C_DIM}) matrix, but got {arr_norm.shape} from {mat_path}"

    max_vals = np.max(arr_norm, axis=0)
    # Add the safety check for division by zero
    max_vals[max_vals == 0] = 1e-12
    arr_norm /= max_vals
    np.fill_diagonal(arr_norm, 0)

    edges = compute_directed_omst(arr_norm, max_rounds=3)
    arr_thresh = np.zeros_like(arr_norm)

    if edges:
        edges_array = np.array(list(edges))
        arr_thresh[edges_array[:, 0], edges_array[:, 1]] = arr_norm[edges_array[:, 0], edges_array[:, 1]]
    return arr_thresh


def str_num(str_input):
    """Extracts the first number from a string."""
    number = re.search(r'\d+', str_input)
    if number:
        return int(number.group())
    return None


def directed_global_efficiency(G):
    """Calculates the directed global efficiency of a graph."""
    n = len(G)
    if n <= 1:
        return 0.0
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    inv_sum = sum(1.0 / d for u in path_lengths for v, d in path_lengths[u].items() if u != v and d > 0)
    return inv_sum / (n * (n - 1))


def compute_directed_omst(adj, max_rounds=3):
    """
    Directed OMST extraction using Edmonds' algorithm, optimized by GCE and leaf fraction penalty.
    Returns a set of directed edges representing the thresholded backbone.
    """
    n = adj.shape[0]
    total_strength = adj.sum()
    best_edges = set()
    best_score = -np.inf

    adj_copy = adj.copy()
    cum_edges = set()

    for m in range(1, max_rounds + 1):
        G = nx.DiGraph()
        rows, cols = np.where(adj_copy > 0)
        for i, j in zip(rows, cols):
            G.add_edge(i, j, weight=1.0 / adj_copy[i, j])

        if G.number_of_edges() == 0:
            break

        ed = nx.algorithms.tree.branchings.Edmonds(G)
        try:
            arb = ed.find_optimum(attr='weight', default=0.0, kind='min', style='arborescence')
        except Exception:
            break

        edges = set(arb.edges())
        if not edges:
            break
        cum_edges |= edges

        if edges:
            edges_to_zero = np.array(list(edges))
            valid_rows = edges_to_zero[:, 0] < adj_copy.shape[0]
            valid_cols = edges_to_zero[:, 1] < adj_copy.shape[1]
            valid_indices = valid_rows & valid_cols
            if np.any(valid_indices):
                adj_copy[edges_to_zero[valid_indices, 0], edges_to_zero[valid_indices, 1]] = 0.0

        H = nx.DiGraph()
        if cum_edges:
            cum_edges_list = list(cum_edges)
            for u, v in cum_edges_list:
                H.add_edge(u, v, weight=adj[u, v])

        try:
            ge = directed_global_efficiency(H)
        except Exception:
            ge = 0.0

        cost = sum(adj[u, v] for u, v in cum_edges) / total_strength if total_strength > 0 else 0.0

        if H.number_of_nodes() > 0:
            degrees = dict(H.degree())
            leaf_count = sum(1 for deg in degrees.values() if deg == 1)
            leaf_frac = leaf_count / float(n)
        else:
            leaf_frac = 0.0

        score = ge - cost - 0.5 * leaf_frac

        if score > best_score:
            best_score = score
            best_edges = cum_edges.copy()

    if not best_edges:
        sym = (adj + adj.T) / 2.0
        U = nx.Graph()
        rows, cols = np.where(sym > 0)
        for i, j in zip(rows, cols):
            if i < j:
                U.add_edge(i, j, weight=1.0 / sym[i, j])

        if U.number_of_edges() > 0:
            try:
                T = nx.minimum_spanning_tree(U, weight='weight')
                for u, v in T.edges():
                    if adj[u, v] >= adj[v, u]:
                        best_edges.add((u, v))
                    else:
                        best_edges.add((v, u))
            except nx.NetworkXNoCycle:
                pass
            except Exception as e:
                pass

    return best_edges


def _process_subject_data(subject_paths, expected_num_videos_for_emotion): # Added new parameter
    """
    Processes all mat files for a single subject and reshapes them.
    This function is designed to be run in parallel.
    Includes assertions for the expected number of videos and windows per video.
    """
    data_sub_list = []
    vids_in_sub = set()

    for p in subject_paths:
        try:
            data = read_mat_file(p)  # Calls read_mat_file from this module
            data_sub_list.append(data)
            vid_num = str_num(os.path.basename(p)[7:9])
            if vid_num is not None:
                vids_in_sub.add(vid_num)
        except (FileNotFoundError, ValueError) as e:
            # print(f"Error processing file {p}: {e}. Skipping this file for subject.")
            pass # Continue processing other files for this subject

    if not data_sub_list: # If no valid data was collected for this subject
        return None

    data_sub = np.array(data_sub_list)
    vids_in_sub = sorted(list(vids_in_sub))

    num_samples, C1, C2 = data_sub.shape
    num_vids = len(vids_in_sub)

    # --- ASSERTION 2a: Check number of videos (dynamic based on emotion) ---
    assert num_vids == expected_num_videos_for_emotion, \
        f"Assertion Error: Subject data from '{subject_paths[0]}' has {num_vids} videos, expected {expected_num_videos_for_emotion} for this emotion."

    # Validate divisibility before calculating num_windows
    if num_samples % num_vids != 0:
        return None # Indicate invalid data for this subject

    num_windows = num_samples // num_vids

    # --- ASSERTION 2b: Check number of windows per video ---
    assert num_windows == EXPECTED_NUM_WINDOWS_PER_VIDEO, \
        f"Assertion Error: Subject data from '{subject_paths[0]}' has {num_windows} windows per video, expected {EXPECTED_NUM_WINDOWS_PER_VIDEO}."

    # Validate C1, C2, though already checked in read_mat_file
    assert C1 == EXPECTED_C_DIM and C2 == EXPECTED_C_DIM, \
        f"Assertion Error: Subject data from '{subject_paths[0]}' has matrix dimensions ({C1},{C2}), expected ({EXPECTED_C_DIM},{EXPECTED_C_DIM})."

    reshaped = data_sub.reshape(num_vids, num_windows, C1, C2)
    return reshaped

# --------------------------------------------------------------------------
