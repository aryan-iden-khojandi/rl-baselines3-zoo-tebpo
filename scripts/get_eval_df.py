import numpy as np
import pandas as pd
import sys


results = np.load(sys.argv[1])

df = pd.DataFrame.from_dict(
    dict(t=results['timesteps'],
         mean=results['results'].mean(axis=1),
         std=results['results'].std(axis=1)))

df.to_csv(sys.stdout, index=False)
