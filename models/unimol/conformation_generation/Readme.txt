Conformation Generation

1. We put each downstream task data in a folder.
These tasks include:
GEOM-Drugs, GEOM-QM9.

PKL files are the raw data. The numbers, e.g., 200, represent the number of molecules in it.
The splitting and raw data are from the previous work ConfGF.
After pre-processing the PKL file we generated the LMDB data used by Uni-Mol.
Each folder contains the train/valid LMDB data.

schema: {idx: {atoms:[], coordinates:[], smi: string, target: []}}


2. Details
atoms:atoms type
coordinates: 3D coordinates 
smi: SMILES
target:  3D coordinates from reference set


3. How to load data
you can read this data use lmdb.
```python
# pip install lmdb
import lmdb
import numpy as np
import os
import pickle
env = lmdb.open(
    lmdb_path,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=256,
)
txn = env.begin()
keys = list(txn.cursor().iternext(values=False))
for idx in keys:
    datapoint_pickled = txn.get(idx)
    data = pickle.loads(datapoint_pickled)
```

4.Others
A detailed description of these task can be found in our paper: 
https://chemrxiv.org/engage/chemrxiv/article-details/628e5b4d5d948517f5ce6d72

Please refer to our code repository for more information on how to load lmdb data: 
https://github.com/dptech-corp/Uni-Mol