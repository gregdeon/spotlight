import pickle

# Saving/loading for inference and spotlight
def saveInferenceResults(fname, embeddings, outputs, labels, losses):
    with open(fname, 'wb+') as f:
        # Don't bother compressing - model outputs don't tend to have a ton of structure
        pickle.dump({
            'embeddings': embeddings,
            'outputs': outputs,
            'labels': labels,
            'losses': losses,
        }, f)
    
def loadInferenceResults(fname):
    with open(fname, 'rb') as f: 
        results = pickle.load(f)
        
    embeddings = results['embeddings'].numpy()
    losses = results['losses'].numpy()
    
    outputs = results['outputs']
    if outputs is not None:
        outputs = outputs.numpy()
    
    if 'labels' in results:
        labels = results['labels'].numpy()
    else:
        labels = None
    
    return (embeddings, outputs, labels, losses)

def saveSpotlightResults(fname, weights, weights_unnorm=None, loss_history=None, weight_history=None, lr_history=None):
    with open(fname, 'wb+') as f:
        pickle.dump({
            'weights': weights,
            'weights_unnorm': weights_unnorm,
            'loss_history': loss_history,
            'weight_history': weight_history,
            'lr_history': lr_history,
        }, f)
        
def loadSpotlightResults(fname):
    with open(fname, 'rb') as f: 
        results = pickle.load(f)
    weights = results['weights']
    weights_unnorm = results['weights_unnorm']
    loss_history = results['loss_history']
    weight_history = results['weight_history']
    lr_history = results['lr_history']
    return weights, weights_unnorm, loss_history, weight_history, lr_history

# Data processing
def normalizeEmbeddings(embeddings):
    embeddings_std = embeddings.std(axis=0)
    embeddings_std[embeddings_std == 0] = 1
    embeddings_norm = (embeddings - embeddings.mean(axis=0)) / embeddings_std
    return embeddings_norm
