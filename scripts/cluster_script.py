import pandas as pd

from readmitrx.cluster.cluster import (
    train_select_cluster_model,
    generate_umap,
    plot_and_save_umap,
)

df = pd.read_csv("data/processed_visits_cleaned.csv")
df_clustered = train_select_cluster_model(df)

df_clustered.to_csv("data/clustered_visits.csv")

plot_and_save_umap(generate_umap(df_clustered))
