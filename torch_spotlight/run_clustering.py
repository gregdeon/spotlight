import argparse
import pickle
import numpy as np
import torch

from sklearn.mixture import GaussianMixture

from torch_spotlight.utils import ClusteringResults, loadResults, saveResults

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run clustering algorithm.')
    parser.add_argument('inference_path', type=str, help='Path to inference results file')
    parser.add_argument('output_path',    type=str, help='Path to save spotlight results')
    parser.add_argument('--num_clusters', type=int, default=50, help='Number of clusters to create')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output while fitting clusters')
    # TODO: possible arguments:
    # - clustering algorithm?
    # - type of covariance matrix to use? (full, diagonal, etc...)
    # - counts? (for "fractional datapoints" in SQuAD results...)

    args = parser.parse_args()

    # Load inference results
    print('Loading inference results from %s...' % args.inference_path)
    inference_results = loadResults(args.inference_path)
    embeddings = inference_results.embeddings.numpy()

    # Run clustering
    print('Running clustering...')
    gm = GaussianMixture(n_components = args.num_clusters, random_state=args.seed, verbose=args.verbose, verbose_interval=1)
    clusters = gm.fit_predict(embeddings)

    # Save results
    print('Saving results to %s...' % args.output_path)
    results = ClusteringResults(
        num_clusters=args.num_clusters,
        clusters=torch.from_numpy(clusters),
        cluster_pickle=pickle.dumps(gm), # TODO: is this annoyingly large to save?
    )
    saveResults(args.output_path, results)
