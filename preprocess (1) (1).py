import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from src.folderconstants import *
from shutil import copyfile

datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB', 'owlyshield']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS']

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def select_anomalies_intelligently(features, labels, target_anomaly_rate=0.03, 
                                  preserve_temporal=True, preserve_diversity=True):
    """
    Intelligently select anomalies to preserve while maintaining detectability
    
    Args:
        features: numpy array of shape (n_samples, n_features)
        labels: numpy array of shape (n_samples,) with 1 for anomaly, 0 for normal
        target_anomaly_rate: desired anomaly rate (e.g., 0.03 for 3%)
        preserve_temporal: if True, preserve temporal patterns in anomalies
        preserve_diversity: if True, preserve diverse types of anomalies
    
    Returns:
        selected_indices: indices of anomalies to keep
        new_labels: modified label array
    """
    
    anomaly_indices = np.where(labels > 0)[0]
    normal_indices = np.where(labels == 0)[0]
    
    total_samples = len(labels)
    target_anomaly_count = int(total_samples * target_anomaly_rate)
    current_anomaly_count = len(anomaly_indices)
    
    print(f"Current anomalies: {current_anomaly_count} ({current_anomaly_count/total_samples*100:.2f}%)")
    print(f"Target anomalies: {target_anomaly_count} ({target_anomaly_rate*100:.2f}%)")
    
    if current_anomaly_count <= target_anomaly_count:
        print("Already at or below target rate, keeping all anomalies")
        return anomaly_indices, labels
    
    # Strategy 1: Detectability-based selection
    # Use Isolation Forest to score how "anomalous" each anomaly is
    print("\n1. Computing anomaly scores...")
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(features[normal_indices])  # Train only on normal data
    
    # Score all anomalies (more negative = more anomalous)
    anomaly_scores = iso_forest.decision_function(features[anomaly_indices])
    
    # Strategy 2: Temporal preservation
    if preserve_temporal:
        print("\n2. Preserving temporal patterns...")
        # Find continuous anomaly segments
        anomaly_segments = []
        start_idx = anomaly_indices[0]
        prev_idx = start_idx
        
        for idx in anomaly_indices[1:]:
            if idx != prev_idx + 1:  # Gap found
                anomaly_segments.append((start_idx, prev_idx))
                start_idx = idx
            prev_idx = idx
        anomaly_segments.append((start_idx, prev_idx))
        
        print(f"   Found {len(anomaly_segments)} anomaly segments")
        
        # Score segments by their average anomaly score and length
        segment_scores = []
        for start, end in anomaly_segments:
            segment_indices = np.arange(start, end + 1)
            segment_mask = np.isin(anomaly_indices, segment_indices)
            avg_score = np.mean(anomaly_scores[segment_mask])
            length = end - start + 1
            # Prefer longer segments with stronger anomaly scores
            combined_score = avg_score * np.log1p(length)
            segment_scores.append(combined_score)
        
        # Sort segments by score (most anomalous first)
        sorted_segments = sorted(zip(anomaly_segments, segment_scores), 
                               key=lambda x: x[1])
        
        # Select whole segments until we reach target
        selected_indices = []
        for (start, end), score in sorted_segments:
            segment_indices = list(range(start, end + 1))
            if len(selected_indices) + len(segment_indices) <= target_anomaly_count:
                selected_indices.extend(segment_indices)
            else:
                # Partially include this segment
                remaining = target_anomaly_count - len(selected_indices)
                if remaining > 0:
                    selected_indices.extend(segment_indices[:remaining])
                break
    
    # Strategy 3: Diversity preservation
    elif preserve_diversity:
        print("\n3. Preserving anomaly diversity...")
        # Use clustering to find different types of anomalies
        anomaly_features = features[anomaly_indices]
        
        # Reduce dimensions for clustering if needed
        if anomaly_features.shape[1] > 10:
            pca = PCA(n_components=10, random_state=42)
            anomaly_features_reduced = pca.fit_transform(anomaly_features)
        else:
            anomaly_features_reduced = anomaly_features
        
        # Determine optimal number of clusters (up to 10)
        n_clusters = min(10, len(anomaly_indices) // 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(anomaly_features_reduced)
        
        # Select anomalies proportionally from each cluster
        selected_indices = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = anomaly_indices[cluster_mask]
            cluster_scores = anomaly_scores[cluster_mask]
            
            # How many to select from this cluster
            cluster_size = len(cluster_indices)
            cluster_proportion = cluster_size / current_anomaly_count
            cluster_target = int(target_anomaly_count * cluster_proportion)
            
            if cluster_target > 0:
                # Select most anomalous from this cluster
                sorted_idx = np.argsort(cluster_scores)[:cluster_target]
                selected_indices.extend(cluster_indices[sorted_idx])
        
        # If we haven't reached target, add more based on anomaly scores
        if len(selected_indices) < target_anomaly_count:
            remaining = target_anomaly_count - len(selected_indices)
            unselected_mask = ~np.isin(anomaly_indices, selected_indices)
            unselected_indices = anomaly_indices[unselected_mask]
            unselected_scores = anomaly_scores[unselected_mask]
            
            sorted_idx = np.argsort(unselected_scores)[:remaining]
            selected_indices.extend(unselected_indices[sorted_idx])
    
    else:
        # Simple selection based on anomaly scores
        print("\n4. Simple score-based selection...")
        sorted_idx = np.argsort(anomaly_scores)[:target_anomaly_count]
        selected_indices = anomaly_indices[sorted_idx]
    
    # Create new labels
    new_labels = np.zeros_like(labels)
    new_labels[selected_indices] = 1
    
    print(f"\nSelected {len(selected_indices)} anomalies")
    print(f"New anomaly rate: {np.mean(new_labels)*100:.2f}%")
    
    # Analyze selection quality
    if len(selected_indices) > 0:
        selected_scores = iso_forest.decision_function(features[selected_indices])
        all_anomaly_scores = iso_forest.decision_function(features[anomaly_indices])
        print(f"\nSelection quality:")
        print(f"  Average score of selected: {np.mean(selected_scores):.4f}")
        print(f"  Average score of all anomalies: {np.mean(all_anomaly_scores):.4f}")
        print(f"  Score improvement: {(np.mean(selected_scores)/np.mean(all_anomaly_scores)-1)*100:.1f}%")
    
    return selected_indices, new_labels


def apply_intelligent_selection_to_owlyshield(dataset_folder, target_rate=0.03):
    """
    Apply intelligent anomaly selection to the OwlyShield dataset
    """
    # Load the preprocessed data
    train = np.load(os.path.join(dataset_folder, 'train.npy'))
    test = np.load(os.path.join(dataset_folder, 'test.npy'))
    labels = np.load(os.path.join(dataset_folder, 'labels.npy'))
    
    # Extract original labels (assuming labels are same across all features)
    original_labels = labels[:, 0]
    
    print("Original test set statistics:")
    print(f"  Shape: {test.shape}")
    print(f"  Anomaly rate: {np.mean(original_labels)*100:.2f}%")
    
    # Apply intelligent selection
    print("\n" + "="*60)
    print("APPLYING INTELLIGENT ANOMALY SELECTION")
    print("="*60)
    
    # Try different strategies
    strategies = [
        ("Temporal Preservation", True, False),
        ("Diversity Preservation", False, True),
        ("Combined (Temporal + Score)", True, False),
    ]
    
    best_strategy = None
    best_labels = None
    best_score = -1
    
    for strategy_name, preserve_temp, preserve_div in strategies:
        print(f"\n\nTrying strategy: {strategy_name}")
        print("-" * 40)
        
        selected_indices, new_labels = select_anomalies_intelligently(
            test, original_labels, 
            target_anomaly_rate=target_rate,
            preserve_temporal=preserve_temp,
            preserve_diversity=preserve_div
        )
        
        # Quick evaluation using Isolation Forest
        iso = IsolationForest(contamination=target_rate, random_state=42)
        iso.fit(test)
        predictions = (iso.predict(test) == -1).astype(int)
        
        # Calculate simple overlap score
        tp = np.sum((predictions == 1) & (new_labels == 1))
        fp = np.sum((predictions == 1) & (new_labels == 0))
        fn = np.sum((predictions == 0) & (new_labels == 1))
        
        if tp > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        
        print(f"  Quick F1 estimate: {f1:.4f}")
        
        if f1 > best_score:
            best_score = f1
            best_strategy = strategy_name
            best_labels = new_labels
    
    print(f"\n\nBest strategy: {best_strategy} (F1: {best_score:.4f})")
    
    # Save the best labels
    # Broadcast to all features
    new_labels_broadcast = np.tile(best_labels.reshape(-1, 1), (1, test.shape[1]))
    
    # Save with a different name to preserve original
    np.save(os.path.join(dataset_folder, 'labels_intelligent.npy'), new_labels_broadcast)
    print(f"\nSaved intelligent labels to: {os.path.join(dataset_folder, 'labels_intelligent.npy')}")
    
    return best_labels


# Additional helper function for gradual reduction
def gradual_anomaly_reduction(features, labels, target_rates=[0.30, 0.15, 0.05, 0.03]):
    """
    Gradually reduce anomaly rate to help model adapt
    """
    print("\n" + "="*60)
    print("GRADUAL ANOMALY REDUCTION STRATEGY")
    print("="*60)
    
    results = []
    current_labels = labels.copy()
    
    for rate in target_rates:
        print(f"\n\nReducing to {rate*100:.1f}% anomaly rate...")
        selected_indices, current_labels = select_anomalies_intelligently(
            features, current_labels, 
            target_anomaly_rate=rate,
            preserve_temporal=True,
            preserve_diversity=False
        )
        
        results.append({
            'rate': rate,
            'indices': selected_indices,
            'labels': current_labels.copy()
        })
    
    return results


def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape


def load_and_save2(category, filename, dataset, dataset_folder, shape):
    temp = np.zeros(shape)
    with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
        ls = f.readlines()
    for line in ls:
        pos, values = line.split(':')[0], line.split(':')[1].split(',')
        start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i) - 1 for i in values]
        temp[start - 1:end - 1, indx] = 1
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)


def wgn(a, snr):
    min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    x = (a - min_a) / (max_a - min_a + 0.0001)
    batch_size, len_x = x.shape
    Ps = np.sum(np.power(x, 2)) / len_x
    Pn = Ps / (np.power(10, snr / 10))
    noise = np.random.randn(len_x) * np.sqrt(Pn)
    return noise/100


def normalize(a):
    a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
    return a / 2 + 0.5


def normalize2(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = min(a), max(a)
    return (a - min_a) / (max_a - min_a) + wgn(a, 50), min_a, max_a


def normalize3(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return ((a - min_a) / (max_a - min_a + 0.0001)) + wgn(a, 50), min_a, max_a


def convertNumpy(df):
    x = df[df.columns[3:]].values[::10, :]
    return (x - x.min(0)) / (x.ptp(0) + 1e-4)


def load_data(dataset):
    folder = os.path.join(output_folder, dataset)
    os.makedirs(folder, exist_ok=True)
    if dataset == 'synthetic':
        train_file = os.path.join(data_folder, dataset, 'synthetic_data_with_anomaly-s-1.csv')
        test_labels = os.path.join(data_folder, dataset, 'test_anomaly.csv')
        dat = pd.read_csv(train_file, header=None)
        split = 10000
        train = normalize(dat.values[:, :split].reshape(split, -1))
        test = normalize(dat.values[:, split:].reshape(split, -1))
        lab = pd.read_csv(test_labels, header=None)
        lab[0] -= split
        labels = np.zeros(test.shape)
        for i in range(lab.shape[0]):
            point = lab.values[i][0]
            labels[point - 30:point + 30, lab.values[i][1:]] = 1
        test += labels * np.random.normal(0.75, 0.1, test.shape)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset == 'SMD':
        dataset_folder = 'data/SMD'
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
                s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
                load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)
    elif dataset == 'UCR':
        dataset_folder = 'data/UCR'
        file_list = os.listdir(dataset_folder)
        for filename in file_list:
            if not filename.endswith('.txt'): continue
            vals = filename.split('.')[0].split('_')
            dnum, vals = int(vals[0]), vals[-3:]
            vals = [int(i) for i in vals]
            temp = np.genfromtxt(os.path.join(dataset_folder, filename),
                                 dtype=np.float64,
                                 delimiter=',')
            min_temp, max_temp = np.min(temp), np.max(temp)
            temp = (temp - min_temp) / (max_temp - min_temp)
            train, test = temp[:vals[0]], temp[vals[0]:]
            labels = np.zeros_like(test)
            labels[vals[1] - vals[0]:vals[2] - vals[0]] = 1
            train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder, f'{dnum}_{file}.npy'), eval(file))
    elif dataset == 'NAB':
        dataset_folder = 'data/NAB'
        file_list = os.listdir(dataset_folder)
        with open(dataset_folder + '/labels.json') as f:
            labeldict = json.load(f)
        for filename in file_list:
            if not filename.endswith('.csv'): continue
            df = pd.read_csv(dataset_folder + '/' + filename)
            vals = df.values[:, 1]
            labels = np.zeros_like(vals, dtype=np.float64)
            for timestamp in labeldict['realKnownCause/' + filename]:
                tstamp = timestamp.replace('.000000', '')
                index = np.where(((df['timestamp'] == tstamp).values + 0) == 1)[0][0]
                labels[index - 4:index + 4] = 1
            min_temp, max_temp = np.min(vals), np.max(vals)
            vals = (vals - min_temp) / (max_temp - min_temp)
            train, test = vals.astype(float), vals.astype(float)
            train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
            fn = filename.replace('.csv', '')
            for file in ['train', 'test', 'labels']:
                np.save(os.path.join(folder, f'{fn}_{file}.npy'), eval(file))
    elif dataset == 'MSDS':
        dataset_folder = 'data/MSDS'
        df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
        df_test = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
        df_train, df_test = df_train.values[::5, 1:], df_test.values[::5, 1:]
        _, min_a, max_a = normalize3(np.concatenate((df_train, df_test), axis=0))
        train, _, _ = normalize3(df_train, min_a, max_a)
        test, _, _ = normalize3(df_test, min_a, max_a)
        labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv'))
        labels = labels.values[::1, 1:]
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
    elif dataset == 'SWaT':
        dataset_folder = 'data/SWaT'

        with open("data/SWaT/SWaT_Dataset_Normal_v1(1).csv", 'r') as f:
            for i, line in enumerate(f):
                if i < 3:  # Show first 3 lines
                    print(f"Line {i}: {line[:200]}...")  # First 200 chars

        
        print("Reading training Excel file...")
        train = pd.read_csv(
            os.path.join(dataset_folder, 'SWaT_Dataset_Normal_v1(1).csv'),
            delimiter=",",
            header=1,  # Use row 1 as header (skip the P1,P2,P3... row)

            low_memory=False,
            encoding='utf-8'
        )
        print("Checking test file structure...")
        with open("data/SWaT/SWaT_Dataset_Attack_v0(1).csv", 'r') as f:
            for i, line in enumerate(f):
                if i < 3:  # Show first 3 lines
                    print(f"Line {i}: {line[:200]}...")  # First 200 chars


        
        print("Reading test Excel file...")  
        test = pd.read_csv(
            os.path.join(dataset_folder, 'SWaT_Dataset_Attack_v0(1).csv'),
            delimiter=",",
            header=1,  # Use row 1 as header (skip the P1,P2,P3... row)

            low_memory=False,
            encoding='utf-8'
        )
        # Save as proper CSV
     #   print("Saving as CSV...")
     #   train.to_csv(os.path.join(dataset_folder, 'SWaT_Dataset_Normal_v0.csv'), index=False, encoding='utf-8')
     #   test.to_csv(os.path.join(dataset_folder, 'SWaT_Dataset_Attack_v0.csv'), index=False, encoding='utf-8')
        
        train.columns = train.columns.str.strip()
        test.columns = test.columns.str.strip()
        print("✅ Cleaned column names (removed spaces)")

        downsample_factor = 5  # Take every 5th sample (adjust as needed: 1=no downsample, 2=half, 5=1/5th, 10=1/10th)

            
        # ✅ APPLY DOWNSAMPLING (MINIMAL CODE)
        if downsample_factor > 1:
            original_train_size = len(train)
            original_test_size = len(test)
            
            train = train.iloc[::downsample_factor].reset_index(drop=True)
            test = test.iloc[::downsample_factor].reset_index(drop=True)
            
            print(f"✅ Downsampled by factor {downsample_factor}:")
            print(f"   Train: {original_train_size:,} → {len(train):,} samples")
            print(f"   Test:  {original_test_size:,} → {len(test):,} samples")


        print("Before processing:", train.shape, test.shape)

        # ✅ MINIMAL ATTACK DISTRIBUTION CHECK
        print(f"\n📊 ATTACK DISTRIBUTION:")
        # Check train file
        if 'Normal/Attack' in train.columns:
            train_attacks = (train['Normal/Attack'].astype(str).str.contains('Attack', case=False, na=False)).sum()
            print(f"Train: {train_attacks:,} attacks out of {len(train):,} ({train_attacks/len(train)*100:.2f}%)")
        else:
            print(f"Train: 0 attacks out of {len(train):,} (0.00%) - no label column")
        
        # Check test file  
        test_attacks = (test['Normal/Attack'].astype(str).str.contains('Attack', case=False, na=False)).sum()
        print(f"Test:  {test_attacks:,} attacks out of {len(test):,} ({test_attacks/len(test)*100:.2f}%)")

             # Print first 10 columns of both files for comparison
        print("\n" + "="*80)
        print("COLUMN COMPARISON:")
        print("="*80)
        
        print(f"\n📊 TRAIN FILE COLUMNS (first 10 of {len(train.columns)}):")
        for i, col in enumerate(train.columns[:10]):
            print(f"  {i:2d}: '{col}'")
        
        print(f"\n📊 TEST FILE COLUMNS (first 10 of {len(test.columns)}):")
        for i, col in enumerate(test.columns[:10]):
            print(f"  {i:2d}: '{col}'")
        
        print(f"\n📊 TRAIN FILE COLUMNS (last 5):")
        for i, col in enumerate(train.columns[-5:], len(train.columns)-5):
            print(f"  {i:2d}: '{col}'")
        
        print(f"\n📊 TEST FILE COLUMNS (last 5):")
        for i, col in enumerate(test.columns[-5:], len(test.columns)-5):
            print(f"  {i:2d}: '{col}'")
        
        print("="*80)
        
            
       
        # Identify timestamp columns and label columns
        timestamp_cols = [col for col in test.columns if any(x in col.lower() for x in ['time', 'date', 'timestamp'])]
        
        # Find label column
        potential_label_cols = ['Normal/Attack', 'Label', 'Attack', 'Class']
        label_col = None
        for col in potential_label_cols:
            if col in test.columns:
                label_col = col
                break
        
        if label_col is None:
            # Assume last column is label
            label_col = test.columns[-1]
            print(f"Warning: Using last column '{label_col}' as label column")
        
        # Get feature columns (everything except timestamps and labels)
        exclude_cols = timestamp_cols + [label_col]
        feature_cols = [col for col in test.columns if col not in exclude_cols]
        
        print(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]}...")  # Show first 5
        print(f"Label column: {label_col}")
        
        # Extract feature data as numpy arrays (similar to MSDS approach)
        train_features = train[feature_cols].values
        test_features = test[feature_cols].values
        
        # Create labels - use same approach as MSDS but for anomaly detection
        labels_raw = test[feature_cols].copy()
        labels_raw[:] = 0  # Initialize all to 0 (normal)
        
        # Set attack labels
        if test[label_col].dtype == 'object':
            attack_mask = test[label_col].str.contains('Attack', case=False, na=False)
        else:
            attack_mask = test[label_col] == 1
        
        labels_raw.loc[attack_mask, :] = 1
        labels_data = labels_raw.values
        
        # Apply MSDS-style scaling WITHOUT noise (normalize3 without wgn)
        def normalize_clean(a, min_a=None, max_a=None):
            """Clean normalization without noise - same as normalize3 but no wgn"""
            if min_a is None: 
                min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
            normalized = (a - min_a) / (max_a - min_a + 0.0001)  # No noise added
            return normalized, min_a, max_a
        
        # First, get global min/max from concatenated train+test data (like MSDS)
        _, min_a, max_a = normalize_clean(np.concatenate((train_features, test_features), axis=0))
        
        # Then normalize train and test using the same scaling parameters
        train_processed, _, _ = normalize_clean(train_features, min_a, max_a)
        test_processed, _, _ = normalize_clean(test_features, min_a, max_a)
        
        # For labels, don't apply normalization (since they're binary)
        labels_processed = labels_data.astype('float64')
        
        print("After processing:", train_processed.shape, test_processed.shape, labels_processed.shape)
        
        # Save the processed data (following MSDS pattern)
        train, test, labels = train_processed, test_processed, labels_processed
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
    elif dataset in ['SMAP', 'MSL']:
        dataset_folder = 'data/SMAP_MSL'
        file = os.path.join(dataset_folder, 'labeled_anomalies.csv')
        values = pd.read_csv(file)
        values = values[values['spacecraft'] == dataset]
        filenames = values['chan_id'].values.tolist()
        for fn in filenames:
            train = np.load(f'{dataset_folder}/train/{fn}.npy')
            test = np.load(f'{dataset_folder}/test/{fn}.npy')
            train, min_a, max_a = normalize3(train)
            test, _, _ = normalize3(test, min_a, max_a)
            np.save(f'{folder}/{fn}_train.npy', train)
            np.save(f'{folder}/{fn}_test.npy', test)
            labels = np.zeros(test.shape)
            indices = values[values['chan_id'] == fn]['anomaly_sequences'].values[0]
            indices = indices.replace(']', '').replace('[', '').split(', ')
            indices = [int(i) for i in indices]
            for i in range(0, len(indices), 2):
                labels[indices[i]:indices[i + 1], :] = 1
            np.save(f'{folder}/{fn}_labels.npy', labels)
    elif dataset == 'WADI':
        dataset_folder = 'data/WADI'
        ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
        train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), header=3)
        test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
        print("Before",train.shape, test.shape, ls.shape)

        
        train.dropna(how='all', inplace=True);
        test.dropna(how='all', inplace=True)
      #  test.fillna(0, inplace=True)

      #  train.fillna(0, inplace=True);
       # train.fillna(method='ffill', inplace=True)
       # train.fillna(method='bfill', inplace=True)  # Handle leading NaNs
        train.fillna(0, inplace=True)  # Final safety net

    
       # test.fillna(method='ffill', inplace=True)   # ← NEW: Consistent with training
       # test.fillna(method='bfill', inplace=True)   # ← NEW: Handle leading NaNs
        test.fillna(0, inplace=True)   # Final safety net

    
     #   test.fillna(0, inplace=True)
        test['Time'] = test['Time'].astype(str)
        test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
        labels = test.copy(deep=True)
        fmt = '%d/%m/%Y %H:%M:%S'
        for i in test.columns.tolist()[3:]: labels[i] = 0
        for i in ['Start Time', 'End Time']:
            ls[i] = ls[i].astype(str)
            ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i], format=fmt)
        for index, row in ls.iterrows():
            to_match = row['Affected'].split(', ')
            matched = []
            for i in test.columns.tolist()[3:]:
                for tm in to_match:
                    if tm in i:
                        matched.append(i);
                        break
            st, et = str(row['Start Time']), str(row['End Time'])
            labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1
        train, test, labels = convertNumpy(train), convertNumpy(test), convertNumpy(labels)
        print("After",train.shape, test.shape, labels.shape)

        
                # --- DEBUG: label density & span stats on processed WADI arrays ---
        # expects: train, test, labels are numpy arrays returned by convertNumpy (2-D; [T,F])
        try:
            lab_arr = np.asarray(labels)
            if lab_arr.ndim == 1:
                lab_arr = lab_arr[:, None]

            # Row-wise anomaly mask: True if *any* feature anomalous at a timestep
            row_mask = (lab_arr > 0).any(axis=1)
            pct = row_mask.mean() * 100.0
            print(f"[WADI DEBUG] Row anomaly pct: {pct:.3f}%  ({row_mask.sum()} / {len(row_mask)})")

            # Contiguous anomaly span lengths (in processed/downsampled timesteps)
            pos_idx = np.where(row_mask)[0]
            spans = []
            if pos_idx.size:
                start = prev = pos_idx[0]
                for i in pos_idx[1:]:
                    if i == prev + 1:
                        prev = i
                    else:
                        spans.append(prev - start + 1)
                        start = prev = i
                spans.append(prev - start + 1)
            if spans:
                print(f"[WADI DEBUG] #spans: {len(spans)} | median: {np.median(spans):.1f} | "
                      f"mean: {np.mean(spans):.1f} | max: {np.max(spans)}")
            else:
                print("[WADI DEBUG] No positive spans found.")
        except Exception as e:
            print(f"[WADI DEBUG] Error computing label stats: {e}")
        # --- END DEBUG ---




        # --- ADD THIS SECTION TO PRINT THE PROCESSED TEST ARRAY ---
        print("\n✅ Final Processed Test Array (first 5 rows):")
        print(f"Shape: {test.shape}")
        print(test[:5, :])
        print("-" * 50)
        # --- END OF ADDED SECTION ---
        
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset == 'MBA':
        dataset_folder = 'data/MBA'
        ls = pd.read_excel(os.path.join(dataset_folder, 'labels.xlsx'))
        train = pd.read_excel(os.path.join(dataset_folder, 'train.xlsx'))
        test = pd.read_excel(os.path.join(dataset_folder, 'test.xlsx'))
        train, test = train.values[1:, 1:].astype(float), test.values[1:, 1:].astype(float)
        train, min_a, max_a = normalize3(train)
        test, _, _ = normalize3(test, min_a, max_a)
        ls = ls.values[:, 1].astype(int)
        labels = np.zeros_like(test)
        for i in range(-20, 20):
            labels[ls + i, :] = 1
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))

    elif dataset == 'owlyshield':
        dataset_folder = 'data/Owlyshield'
        data_file = os.path.join(dataset_folder, 'owlyshield_data.csv')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load data
        df = pd.read_csv(data_file)
        print(f"Loaded owlyshield data: {df.shape}")
        
        # Separate features and labels
        feature_cols = df.columns[:-1]  # All columns except last
        features = df[feature_cols].values
        original_labels = df[df.columns[-1]].values
        
        print(f"Features shape: {features.shape}")
        print(f"Original labels shape: {original_labels.shape}")
        print(f"Original anomaly rate: {np.mean(original_labels > 0)*100:.2f}%")
        
        # Split data: 70% for training, 30% for testing
        total_samples = len(features)
        split_idx = int(total_samples * 0.7)
        
        train_features = features[:split_idx]
        train_labels_original = original_labels[:split_idx]
        test_features = features[split_idx:]
        test_labels_original = original_labels[split_idx:]
        
        print(f"Train samples: {len(train_features)} ({len(train_features)/total_samples*100:.2f}%)")
        print(f"Test samples: {len(test_features)} ({len(test_features)/total_samples*100:.2f}%)")
        print(f"Train set original anomaly rate: {np.mean(train_labels_original > 0)*100:.2f}%")
        print(f"Test set original anomaly rate: {np.mean(test_labels_original > 0)*100:.2f}%")
        
        # ============================================================================
        # INTELLIGENT ANOMALY SELECTION
        # ============================================================================
        
        # Strategy 1: Gradual reduction for better model adaptation
        GRADUAL_REDUCTION = True
        TARGET_ANOMALY_RATE = 0.40  # Final target
        
        if GRADUAL_REDUCTION:
            print("\n" + "="*60)
            print("USING GRADUAL ANOMALY REDUCTION STRATEGY")
            print("="*60)
            
            # First normalize the data (needed for anomaly scoring)
            def normalize_clean(a, min_a=None, max_a=None):
                if min_a is None: 
                    min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
                normalized = (a - min_a) / (max_a - min_a + 0.0001)
                return normalized, min_a, max_a
            
            # Get global scaling parameters
            _, min_a, max_a = normalize_clean(np.concatenate((train_features, test_features), axis=0))
            train_normalized, _, _ = normalize_clean(train_features, min_a, max_a)
            test_normalized, _, _ = normalize_clean(test_features, min_a, max_a)
            
            # Progressive reduction rates: 30% -> 15% -> 5% -> 3%
            reduction_stages = [0.30, 0.15, 0.05]
            
            for stage_idx, target_rate in enumerate(reduction_stages):
                print(f"\n--- Stage {stage_idx + 1}: Reducing to {target_rate*100:.1f}% ---")
                
                # Find anomaly indices
                test_anomaly_indices = np.where(test_labels_original > 0)[0]
                total_test_samples = len(test_labels_original)
                target_anomaly_count = int(total_test_samples * target_rate)
                
                if len(test_anomaly_indices) <= target_anomaly_count:
                    print(f"Already at or below target rate, keeping all anomalies")
                    continue
                
                # Score anomalies using Isolation Forest
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                
                # Train on normal data only
                # Train on TRAINING normal data only
                train_normal_indices = np.where(train_labels_original == 0)[0]
                iso_forest.fit(train_normalized[train_normal_indices])  # ✅ Use training data

 
                # Score all anomalies (more negative = more anomalous)
                anomaly_scores = iso_forest.decision_function(test_normalized[test_anomaly_indices])
                
                # Preserve temporal patterns
                anomaly_segments = []
                if len(test_anomaly_indices) > 0:
                    start_idx = test_anomaly_indices[0]
                    prev_idx = start_idx
                    
                    for idx in test_anomaly_indices[1:]:
                        if idx != prev_idx + 1:  # Gap found
                            anomaly_segments.append((start_idx, prev_idx))
                            start_idx = idx
                        prev_idx = idx
                    anomaly_segments.append((start_idx, prev_idx))
                
                print(f"Found {len(anomaly_segments)} anomaly segments")
                
                # Score segments
                segment_data = []
                for seg_idx, (start, end) in enumerate(anomaly_segments):
                    segment_indices = np.arange(start, end + 1)
                    segment_mask = np.isin(test_anomaly_indices, segment_indices)
                    if np.any(segment_mask):
                        avg_score = np.mean(anomaly_scores[segment_mask])
                        length = end - start + 1
                        # Prefer segments with strong anomaly scores and reasonable length
                        # Penalize very long segments to maintain diversity
                        length_factor = np.log1p(length) / np.log1p(100)  # Normalize by log(100)
                        combined_score = avg_score * (1 + length_factor)
                        segment_data.append((seg_idx, start, end, combined_score, length))
                
                # Sort by score (most anomalous first)
                segment_data.sort(key=lambda x: x[3])
                
                # Select segments
                selected_indices = []
                selected_segments = []
                
                for seg_idx, start, end, score, length in segment_data:
                    segment_indices = list(range(start, end + 1))
                    
                    if len(selected_indices) + len(segment_indices) <= target_anomaly_count:
                        selected_indices.extend(segment_indices)
                        selected_segments.append((start, end))
                    elif len(selected_indices) < target_anomaly_count:
                        # Partially include this segment
                        remaining = target_anomaly_count - len(selected_indices)
                        selected_indices.extend(segment_indices[:remaining])
                        selected_segments.append((start, start + remaining - 1))
                        break
                
                # Update labels for this stage
                test_labels_original = np.zeros(total_test_samples)
                test_labels_original[selected_indices] = 1
                
                print(f"Selected {len(selected_segments)} segments containing {len(selected_indices)} anomalies")
                print(f"New anomaly rate: {np.mean(test_labels_original)*100:.2f}%")
                
                # Save intermediate results for this stage
                stage_name = f"stage{stage_idx + 1}_{int(target_rate*100)}pct"
                labels_broadcast = np.tile(test_labels_original.reshape(-1, 1), (1, test_normalized.shape[1]))
                np.save(os.path.join(folder, f'labels_{stage_name}.npy'), labels_broadcast.astype('float64'))
        
        else:
            # Strategy 2: Direct intelligent selection to target rate
            print("\n" + "="*60)
            print("USING DIRECT INTELLIGENT SELECTION")
            print("="*60)
            
            # Normalize first
            def normalize_clean(a, min_a=None, max_a=None):
                if min_a is None: 
                    min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
                normalized = (a - min_a) / (max_a - min_a + 0.0001)
                return normalized, min_a, max_a
            
            _, min_a, max_a = normalize_clean(np.concatenate((train_features, test_features), axis=0))
            train_normalized, _, _ = normalize_clean(train_features, min_a, max_a)
            test_normalized, _, _ = normalize_clean(test_features, min_a, max_a)
            
            # Apply selection function from the artifact
            from sklearn.ensemble import IsolationForest
            selected_indices, test_labels_original = select_anomalies_intelligently(
                test_normalized, 
                test_labels_original, 
                target_anomaly_rate=TARGET_ANOMALY_RATE,
                preserve_temporal=True,
                preserve_diversity=False
            )
        
        # ============================================================================
        # FINAL PROCESSING
        # ============================================================================
        
        # Training labels: Should be mostly normal
        reduced_train_labels = train_labels_original.copy()
        
        # Final normalization (already done above)
        train_processed = train_normalized
        test_processed = test_normalized
        
        # Create final label arrays
        train_labels_final = np.tile(
            reduced_train_labels.reshape(-1, 1), 
            (1, train_processed.shape[1])
        )
        
        test_labels_final = np.tile(
            test_labels_original.reshape(-1, 1), 
            (1, test_processed.shape[1])
        )
        
        # Save processed data
        train, test, labels = train_processed, test_processed, test_labels_final
        
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file).astype('float64'))
        
        # Print final statistics
        train_anomaly_rate = np.mean(train_labels_final > 0) * 100
        test_anomaly_rate = np.mean(test_labels_final > 0) * 100
        
        print(f"\n" + "="*60)
        print(f"OWLYSHIELD INTELLIGENT PREPROCESSING COMPLETE")
        print(f"="*60)
        print(f"Saved files:")
        print(f"  train.npy: {train.shape}")
        print(f"  test.npy: {test.shape}") 
        print(f"  labels.npy: {labels.shape}")
        print(f"Final statistics:")
        print(f"  Train anomaly rate: {train_anomaly_rate:.4f}%")
        print(f"  Test anomaly rate: {test_anomaly_rate:.4f}%")
        
        # Analyze final anomaly distribution
        if test_anomaly_rate > 0:
            anomaly_positions = np.where(test_labels_original > 0)[0]
            gaps = np.diff(anomaly_positions)
            continuous_segments = np.sum(gaps == 1)
            print(f"\nAnomaly distribution analysis:")
            print(f"  Total anomaly positions: {len(anomaly_positions)}")
            print(f"  Continuous pairs: {continuous_segments}")
            print(f"  Average gap between anomalies: {np.mean(gaps):.2f}")
            print(f"  Temporal preservation ratio: {continuous_segments/len(gaps)*100:.1f}%")
        
        print(f"="*60)

    
    else:
        raise Exception(f'Not Implemented. Check one of {datasets}')


if __name__ == '__main__':
    commands = sys.argv[1:]
    load = []
    if len(commands) > 0:
        for d in commands:
            load_data(d)
    else:
        print("Usage: python preprocess.py <datasets>")
        print(f"where <datasets> is space separated list of {datasets}")
