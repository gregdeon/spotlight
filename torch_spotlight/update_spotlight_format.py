import argparse
from torch_spotlight.utils import loadResults, saveResults, SpotlightResults

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update spotlight results format.')
    parser.add_argument('fname_old', type=str, help='Spotlight results in old format')
    parser.add_argument('fname_new', type=str, help='Path to store new spotlight results')
    
    args = parser.parse_args()
    print(args)
    
    old_results = loadResults(args.fname_old)
    new_results = SpotlightResults(
        weights = old_results['weights'],
        unnormalized_weights = old_results['weights_unnorm'],
        spotlight_mean = None,
        spotlight_precision = None,
        training_history = {
            'loss_history': old_results['loss_history'],
            'weight_history': old_results['weight_history'],
            'lr_history': old_results['lr_history'],
        }
    )
    saveResults(args.fname_new, new_results)
    