import setuptools

setuptools.setup(
    name="spotlight",
    description="PyTorch implementation of https://arxiv.org/abs/2107.00758",
    version=0.1,
    author="Greg d'Eon",
    author_email="gregdeon@cs.ubc.ca",
    # install_requires = [ # TODO: is this right?
    #    'numpy>=1.19.1',
    #    'outdated>=0.2.0',
    #    'pandas>=1.1.0',
    #    'pillow>=7.2.0',
    #    'pytz>=2020.4',
    #    'torch>=1.7.0',
    #    'torchvision>=0.8.2',
    #    'tqdm>=4.53.0',
    #    'scikit-learn>=0.20.0',
    #    'scipy>=1.5.4'
    #],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)

