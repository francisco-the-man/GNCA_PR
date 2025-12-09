graph-cellular-automata/
├── README.md               # Explanation, installation, citations
├── setup.py                # For `pip install -e .`
├── requirements.txt
├── gnca/                   # THE LIBRARY (Source Code)
│   ├── __init__.py
│   ├── conv.py             # The GNCAConv layer (PR Candidate)
│   ├── model.py            # The full Wrapper (Pre-MLP -> Conv -> Post-MLP)
│   └── buffer.py           # The SamplePool/Cache (Critical for stability)
└── examples/               # THE EXPERIMENTS (Project 3)
    ├── voronoi.py          # Experiment 1: Voronoi Tessellation
    └── growing_shapes.py   # Experiment 3: Target State Convergence (Bunny/Logo)