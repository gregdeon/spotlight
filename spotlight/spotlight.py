import torch
import torch.optim as optim
import datetime
import numpy as np

max_svd_attempts = 10

def gaussian_probs(mean, precision, x):
    # Similarity kernel: describe how similar each point in x is to mean as number in [0, 1] 
    # - mean: (dims) vector
    # - precision: (dims, dims) precision matrix; must be PSD
    # - x: (num_points, dims) set of points
    dists = torch.sum(((x - mean) @ precision) * (x - mean), axis=1)
    return torch.exp(-dists / 2)

def md_adversary_weights(mean, precision, x, losses):
    # Calculate normalized weights, average loss, and spotlight size for current mean and precision settings
    # - mean, precision, x: as in gaussian_probs
    # - losses: (num_points) vector of losses
    weights_unnorm = gaussian_probs(mean, precision, x)
    total_weight = torch.sum(weights_unnorm)
    weights = weights_unnorm / torch.sum(weights_unnorm)
    weighted_loss = weights @ losses
    
    return (weights, weights_unnorm, weighted_loss, total_weight)

def md_objective_spherical(mean, precision, x, losses, min_weight, barrier_x, barrier_scale, std_coeff=0.0, flip_objective=False):
    # Spotlight objective, parameterized by a single precision shared across all dimensions
    # Objective to maximize is (weighted loss) - (barrier penalty) - (regularization penalties for small precision)
    precision_matrix = torch.eye(x.shape[1], device=precision.device) * precision
    _, _, weighted_loss, total_weight = md_adversary_weights(mean, precision_matrix, x, losses)
    if flip_objective:
        weighted_loss = -weighted_loss
    if total_weight < (min_weight + barrier_x):
        barrier_penalty = barrier_scale * (total_weight - (min_weight + barrier_x))**2 / barrier_x**2
        weighted_loss -= barrier_penalty    
    weighted_loss -= std_coeff / precision
    return (weighted_loss, total_weight)

def md_objective(mean, V, d, x, losses, min_weight, barrier_x, barrier_scale, std_coeff=0.0, flip_objective=False):
    # Spotlight objective, parameterized by eigendecomposition of precision matrix
    # - V: orthogonal set of eigenvectors
    # - d: precision in each dimension
    precision = V @ torch.diag(d) @ torch.inverse(V)
    _, _, weighted_loss, total_weight = md_adversary_weights(mean, precision, x, losses)
    if flip_objective:
        weighted_loss = -weighted_loss
    if total_weight < (min_weight + barrier_x):
        barrier_penalty = barrier_scale * (total_weight - (min_weight + barrier_x))**2 / barrier_x**2
        weighted_loss -= barrier_penalty    
    weighted_loss -= torch.sum(std_coeff / d)
    return (weighted_loss, total_weight)

class ResetOnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def _reduce_lr(self, epoch):
        super(ResetOnPlateau, self)._reduce_lr(epoch)
        self._reset()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def run_spotlight(
    embeddings, 
    losses, 
    min_weight, 
    spherical_spotlight, 
    barrier_x_schedule, 
    barrier_scale, 
    learning_rate, 
    scheduler_patience=20,
    scheduler_decay=0.5,
    print_every=20, 
    device='cpu',
    flip_objective=False
):
    x = torch.tensor(embeddings, device=device)
    y = torch.tensor(losses, device=device)
    dimensions = x.shape[1]

    mean = torch.zeros((dimensions,), requires_grad=True, device=device)
    
    if spherical_spotlight:
        log_precision = torch.tensor(np.log(0.0001), requires_grad=True, device=device)
        optimizer = optim.Adam([mean, log_precision], lr=learning_rate)
    else:
        V = torch.eye(dimensions, requires_grad=True, device=device)
        log_d = torch.full((dimensions,), np.log(0.001), requires_grad=True, device=device)
        optimizer = optim.Adam([mean, V, log_d], lr=learning_rate)
    
    scheduler = ResetOnPlateau(optimizer, patience=scheduler_patience, factor=scheduler_decay)
    
    num_steps = len(barrier_x_schedule)
    
    objective_history = []
    total_weight_history = []
    lr_history = []
    
    start_time = datetime.datetime.now()

    for t in range(num_steps):
        optimizer.zero_grad()
        if spherical_spotlight:
            precision = torch.exp(log_precision)
            objective, total_weight = md_objective_spherical(mean, precision, x, y, min_weight, barrier_x_schedule[t], barrier_scale, flip_objective=flip_objective)
        else:
            d = torch.exp(log_d)
            objective, total_weight = md_objective(mean, V, d, x, y, min_weight, barrier_x_schedule[t], barrier_scale, flip_objective=flip_objective)
        neg_objective = -objective
        neg_objective.backward()    
        optimizer.step()
        scheduler.step(neg_objective)
        
        objective_history.append(objective.detach().cpu().item())
        total_weight_history.append(total_weight.detach().cpu().item())
        lr_history.append(get_lr(optimizer))
        
        if not spherical_spotlight:
            # Project eigenvectors to closest orthogonal matrix
            # Note that svd.V in output is already transposed from typical notation
            # Also note that SVD has issues when singular values are similar
            # Add random noise to V before computing SVD to avoid issues
            with torch.no_grad():
                for attempt in range(max_svd_attempts):
                    try:
                        svd = torch.linalg.svd(V + 1e-4*V.abs().mean()*torch.rand(*V.shape, device=device))
                        break
                    except:
                        print('SVD failed (attempt %d/%d)' % (attempt+1, max_svd_attempts))
                        if attempt+1 == max_svd_attempts:
                            raise
#                         np.save('failed_matrix.npy', V.detach().cpu().numpy())
                        
                V.data = (svd.U @ svd.V).detach().clone().to(device)
            
        if (t+1) % print_every == 0:
            if spherical_spotlight:
                precision_matrix = torch.eye(dimensions, device=precision.device) * torch.exp(log_precision)
            else:
                d = torch.exp(log_d)
                precision_matrix = V @ torch.diag(d) @ torch.inverse(V)
                
            weights, weights_unnorm, weighted_loss, total_weight = md_adversary_weights(mean, precision_matrix, x, y)
            ms_spent = (datetime.datetime.now() - start_time).total_seconds() * 1000
            
            if spherical_spotlight:
                precision_print = torch.exp(log_precision)
            else:
                precision_print = torch.max(d)
            print('steps = %5d | ms = %5d | mean = %5.2f | precision = %5.6f | loss = %.3f | total weight = %7.1f | barrier = %6.1f | lr = %.5f' % (
                t+1, ms_spent, (mean**2).sum(), precision_print, weighted_loss, total_weight, barrier_x_schedule[t], get_lr(optimizer)
            ))
    
    final_weights = weights.detach().cpu().numpy()
    final_weights_unnorm = weights_unnorm.detach().cpu().numpy()
    
    return (final_weights, final_weights_unnorm, objective_history, total_weight_history, lr_history)
