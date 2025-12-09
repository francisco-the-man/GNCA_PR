from setuptools import setup, find_packages

setup(
    name='gnca',
    version='0.1.0',
    description='Implementation of "Learning Graph Cellular Automata" in PyTorch Geometric',
    author='Your Name',
    author_email='your.email@stanford.edu',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torch-geometric',
        'numpy',
        'scipy',
        'matplotlib',
        'pygsp',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)