import subprocess, pandas as pd, numpy as np

def run_nprint(pcap_path, out_csv, headers="-4 -t -u -i", count=20, exclude_regex=None):
    cmd = ["./nprint_bin/nprint","-P",pcap_path] + headers.split() + ["-c",str(count),"-W",out_csv]
    if exclude_regex: cmd += ["-x", exclude_regex]
    subprocess.run(cmd, check=True)

def csv_to_numpy(csv_path):
    df = pd.read_csv(csv_path, dtype=np.int8)
    X = df.select_dtypes(include=["int8","int16","int32"]).to_numpy(dtype=np.int8)
    return X