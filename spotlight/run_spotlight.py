import argparse
import numpy as np
from sklearn.decomposition import PCA

import spotlight
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run spotlight.')
    parser.add_argument('min_weight',     type=int, help='Number of points required in spotlight')
    parser.add_argument('inference_path', type=str, help='Path to inference results file')
    parser.add_argument('output_path',    type=str, help='Path to save spotlight results')
    parser.add_argument('--spherical',     action='store_true',       help='Use single-parameter precision matrix')
    parser.add_argument('--barrier_min_x', type=int,                  help='Barrier x scale at end of optimization')
    parser.add_argument('--barrier_scale', type=float, default=1,     help='Barrier y scale')
    parser.add_argument('--num_steps',     type=int,   default=1000,  help='Number of optimizer steps')
    parser.add_argument('--learning_rate', type=float, default=1e-3,  help='Initial learning rate')
    parser.add_argument('--lr_patience',   type=int,   default=20,    help='LR scheduler: number stagnated steps before lowering learning rate')
    parser.add_argument('--lr_factor',     type=float, default=0.5,   help='LR scheduler: factor to apply when lowering learning rate')
    parser.add_argument('--print_every',   type=int,   default=20,    help='Frequency of optimizer status updates')
    parser.add_argument('--device',        type=str,   default='cpu', help='Device to run optimization on')
    parser.add_argument('--past_weights', nargs='*',                  help='List of paths to previous weights')
    parser.add_argument('--top_components', type=int,                 help='Number of PCA components to give to the adversary; default is unmodified embedding')
    parser.add_argument('--flip_objective', action='store_true',      help='Use spotlight to search for low-loss points')
    
    args = parser.parse_args()
#     print(args)
    
    print('Loading inference results from %s' % args.inference_path)
    embeddings, _, losses = utils.loadInferenceResults(args.inference_path)
    num_points = len(embeddings)
    
    barrier_min_x = args.barrier_min_x if args.barrier_min_x else 0.05 * args.min_weight
    barrier_x_schedule = np.geomspace(num_points - args.min_weight, barrier_min_x, args.num_steps)
    
    if args.past_weights is not None:
        print('Reducing losses based on past weights...')
        for weight_path in args.past_weights:
            print('- %s' % weight_path)
            weights_unnorm = utils.loadSpotlightResults(weight_path)[1]
            weights_unnorm /= max(weights_unnorm)
            losses = losses * (1 - weights_unnorm)
            
    if args.top_components is not None:
        print('Computing PCA with %d components...' % args.top_components)
        pca = PCA(n_components=args.top_components)
        pca.fit(embeddings)
        embeddings = pca.transform(embeddings)
     
    weights, weights_unnorm, objective_history, total_weight_history, lr_history = spotlight.run_spotlight(
        embeddings, 
        losses,
        args.min_weight,
        args.spherical,
        barrier_x_schedule,
        args.barrier_scale,
        args.learning_rate,
        args.lr_patience,
        args.lr_factor,
        args.print_every,
        args.device,
        args.flip_objective,
    )
    
    print('Saving spotlight results to %s' % args.output_path)
    utils.saveSpotlightResults(args.output_path, weights, weights_unnorm, objective_history, total_weight_history, lr_history)
    
    