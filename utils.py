import glob 
import pandas as pd
import os

BASE_PATH = os.environ.get('GEMINI_BASE_PATH')

# read all platform files 
def load_platform_conversations_df(base_path=BASE_PATH, platform="gemini"):
    files = sorted(glob.glob(os.path.join(base_path, f"*_{platform}.json")))
    print(f"Found {len(files)} files for platform '{platform}'.")
    dfs = []

    for fp in files:
        filename = os.path.basename(fp)
        user_id = filename[:-(len(f"_{platform}.json"))]  
        
        df_one = pd.read_json(fp)

        # Add user_id
        df_one.insert(0, "user_id", user_id)
        dfs.append(df_one)

    if not dfs:
        print("No files found.")
        return False 
    total_rows = sum(len(df) for df in dfs)
    return pd.concat(dfs, ignore_index=True)


# helper function 
from bs4 import BeautifulSoup

def html_to_text(html_content):
    if not html_content or not isinstance(html_content, str):
        return ''
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator=' ', strip=True)




# HELPERS FOR STATS TESTS 

# Helpers
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

RAREFY_N = 15       # turns sampled per user for breadth comparisons or number of conversations when relevant 
RAREFY_REPS = 25
N_BOOT = 1000
N_PERM = 1000
TOP_N = 12

def norm_entropy(p, k_total):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size < 2 or k_total < 2:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(k_total))

def js_distance(p, q, eps=1e-12):
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))
    return float(np.sqrt(kl))

def gini(x):
    x = np.sort(np.asarray(x, float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return np.nan
    idx = np.arange(1, n + 1)
    return float(((2 * idx - n - 1) * x).sum() / (n * x.sum()))

def bh(p):
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    adj = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out

def boot_mean_ci(mat, n_boot=N_BOOT, alpha=0.05):
    """Cluster bootstrap over users. Returns mean, lower and upper bounds."""
    a = np.asarray(mat, float)
    n = a.shape[0]
    draws = np.empty((n_boot, a.shape[1]))
    for b in range(n_boot):
        draws[b] = a[RNG.integers(0, n, n)].mean(axis=0)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return a.mean(axis=0), lo, hi

def boot_ci(x, stat=np.mean, n=N_BOOT, alpha=0.05):
    """For trouble data. bootstrap cis"""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, x.size, size=(n, x.size))
    reps = stat(x[idx], axis=1)
    return stat(x), np.quantile(reps, alpha / 2), np.quantile(reps, 1 - alpha / 2)


def cliffs_delta(a, b):
    a = np.asarray(pd.Series(a).dropna(), dtype=float)
    b = np.asarray(pd.Series(b).dropna(), dtype=float)
    if a.size == 0 or b.size == 0:
        return np.nan
    diff = np.sign(a[:, None] - b[None, :])
    return diff.mean()

