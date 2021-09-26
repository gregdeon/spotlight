import pickle
from dataclasses import dataclass
import torch

@dataclass
class InferenceResults:
    """
    Class for storing embeddings and losses from running inference on a model.
    
    Fields:
    - embeddings: (num_examples x num_dimensions) tensor of last-layer embeddings
    - losses: (num_examples x 1) tensor of losses
    - outputs: optional (num_examples x num_classes) tensor of output logits
    - labels: optional (num_examples x 1) tensor of labels
    - aux: optional field for additional information
    """
    
    embeddings: torch.Tensor
    losses: torch.Tensor
    outputs: torch.Tensor = None
    labels: torch.Tensor = None 
    aux: 'typing.Any' = None
        
@dataclass
class SpotlightResults:
    """
    Class for storing results of a single spotlight run.
    
    Fields:
    - weights: (num_examples) tensor of probabilities (summing to 1)
    - unnormalized_weights: (num_examples) tensor of weights (summing to spotlight size)
    - spotlight_mean: (num_dimensions) tensor: location of spotlight's center
    - spotlight_precision: (num_dimensions, num_dimensions) tensor: spotlight's precision matrix
    - training_history: dictionary with training results for debugging
    """
    
    weights: torch.Tensor
    unnormalized_weights: torch.Tensor
    spotlight_mean: torch.Tensor
    spotlight_precision: torch.Tensor
    training_history: dict = None

@dataclass
class ClusteringResults:
    """
    Class for storing results of a clustering run.
    
    Fields:
    - num_clusters: number of clusters 
    - clusters: (num_examples) list of assignments of each point to a cluster
    - cluster_pickle: TODO
    """
    num_clusters: int
    clusters: torch.Tensor
    cluster_pickle: str = None
        
def saveResults(fname, results):
    """
    Save inference or spotlight results to file
    
    Arguments:
    - fname: path to save file to
    - results: an InferenceResults or SpotlightResults object
    """
    
    with open(fname, 'wb+') as f:
        pickle.dump(results, f)
        
def loadResults(fname):
    """
    Load inference or spotlight results from file
    
    Arguments:
    - fname: path with saved results
    """
    
    with open(fname, 'rb') as f:
        results = pickle.load(f)
    
    return results

def loadLegacyInferenceResults(fname):
    """
    Load inference results in old format
    """
    
    with open(fname, 'rb') as f: 
        results = pickle.load(f)
        
    return InferenceResults(
        embeddings = results['embeddings'],
        losses = results['losses'],
        outputs = results['outputs'],
        labels = results['labels'] if 'labels' in results else None,
        aux = results['aux'] if 'aux' in results else None,
    )

def loadLegacySpotlightResults(fname):
    """
    Load spotlight results in old format
    """
    
    with open(fname, 'rb') as f: 
        results = pickle.load(f)
        
    return SpotlightResults(
        weights = results['weights'],
        unnormalized_weights = results['weights_unnorm'],
        spotlight_mean = None,
        spotlight_precision = None,
        training_history = {
            'loss': results['loss_history'],
            'weight': results['weight_history'],
            'lr': results['lr_history'],
        },
    )

# Data processing
def normalizeEmbeddings(embeddings):
    embeddings_std = embeddings.std(axis=0)
    embeddings_std[embeddings_std == 0] = 1
    embeddings_norm = (embeddings - embeddings.mean(axis=0)) / embeddings_std
    return embeddings_norm
