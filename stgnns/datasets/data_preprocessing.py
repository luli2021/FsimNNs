import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA



def filter_edges(original_file, output_file, distance):
    """
    Reads the original edges.csv, filters and shortens edge features based on parameter distance,
    then saves the filtered edges to output_file.
    """
    df = pd.read_csv(original_file)
    df['feat'] = df['feat'].apply(lambda x: [float(v) for v in x.split(',')])

    # Step 1: Filter and shorten features
    filtered_rows = []
    shortened_feats = []

    for _, row in df.iterrows():
        feat = row['feat']
        removed = feat[distance:]
        if any(v != 0 for v in removed):
            continue  # Skip this edge
        shortened_feat = feat[:distance]
        filtered_rows.append((row['graph_id'], row['src_id'], row['dst_id']))
        shortened_feats.append(shortened_feat)

    if not shortened_feats:
        raise ValueError("No valid edges left after filtering.")

    # Step 2: Apply PCA
    pca = PCA(n_components=distance-2)
    pca_features = pca.fit_transform(shortened_feats)

    # Step 3: Write filtered + PCA-transformed data to tmp file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    rows_to_save = []
    for i, (graph_id, src_id, dst_id) in enumerate(filtered_rows):
        pca_feat_str = ','.join(map(str, pca_features[i]))
        rows_to_save.append([graph_id, src_id, dst_id, pca_feat_str])

    out_df = pd.DataFrame(rows_to_save, columns=['graph_id', 'src_id', 'dst_id', 'feat'])
    out_df.to_csv(output_file, index=False)


