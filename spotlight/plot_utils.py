import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Matplotlib helpers
def setupPlots():
    matplotlib.rcParams['figure.dpi'] = 150
    matplotlib.rcParams['font.size'] = 8
    matplotlib.rcParams['axes.titlepad'] = 1
    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

blues_trunc = truncate_colormap(plt.get_cmap('Blues'), 0.1, 1.0)

def makeGrid(width, height, **kwargs):
    fig, axes = plt.subplots(height, width, squeeze=False, subplot_kw={'xticks': [], 'yticks': []}, **kwargs)
    return fig, axes.flat

def plotImage(ax, image, **plot_kwargs):
    ax.imshow(image.permute(1, 2, 0), **plot_kwargs)
    ax.axis('off')

def plotImageGrid(fig_size, array_size, image_list, title_list, padding=0.6, fontsize=6, **plot_kwargs):
    num_x, num_y = (array_size)
    fig, axes = makeGrid(num_x, num_y, figsize=fig_size)
    for i, ax in enumerate(axes):
        plotImage(ax, image_list[i], **plot_kwargs)
        ax.set_title(title_list[i], fontsize=6)
    fig.tight_layout(pad=padding)
    
def plotEmbeddingLosses(embeddings_pca, losses, pca_components, plot_first=None, sort_by_z=True, **plot_kwargs):
    component_1, component_2 = (pca_components)

    plt_x = embeddings_pca[:, component_1]
    plt_y = embeddings_pca[:, component_2]
    plt_z = losses[:]
    
    if plot_first is not None:
        plt_x = plt_x[:plot_first]
        plt_y = plt_y[:plot_first]
        plt_z = plt_z[:plot_first]
        
    if sort_by_z:
        idx = np.argsort(plt_z)
        plt_x = plt_x[idx]
        plt_y = plt_y[idx]
        plt_z = plt_z[idx]

    plt.figure(figsize=(4,3))
    plt.scatter(plt_x, plt_y, s=4, c=plt_z, alpha=1, zorder=2, cmap=blues_trunc, **plot_kwargs)
    cbar = plt.colorbar(extend='max')
    plt.xlabel('PCA component %d' % (component_1 + 1))
    plt.ylabel('PCA component %d' % (component_2 + 1))
    cbar.set_label('Loss')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

def plotEmbeddingOutputs(embeddings_pca, outputs, pca_components, plot_first=None, sort_by_z=True, **plot_kwargs):
    component_1, component_2 = (pca_components)

    plt_x = embeddings_pca[:, component_1]
    plt_y = embeddings_pca[:, component_2]
    plt_z = outputs[:]
    
    if plot_first is not None:
        plt_x = plt_x[:plot_first]
        plt_y = plt_y[:plot_first]
        plt_z = plt_z[:plot_first]
        
    if sort_by_z:
        idx = np.argsort(plt_z)
        plt_x = plt_x[idx]
        plt_y = plt_y[idx]
        plt_z = plt_z[idx]

    plt.figure(figsize=(4,3))
    plt.scatter(plt_x, plt_y, s=4, c=plt_z, alpha=1, zorder=2, cmap='RdBu', vmin=0, vmax=1, **plot_kwargs)
    cbar = plt.colorbar()
    plt.xlabel('PCA component %d' % (component_1 + 1))
    plt.ylabel('PCA component %d' % (component_2 + 1))
    cbar.set_label('Prediction')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

