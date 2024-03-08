"""
Read data, split data into training and test sets, save data.
""" 

import argparse 
import os, glob
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None, help="data directory")
    ap.add_argument("--train_fraction", default=0.8, type=float, help="fraction of training data")
    ap.add_argument("--random_seed", default=42, type=int, help="random seed")
    args = ap.parse_args()
    
    print("data_dir       =", args.data_dir)
    print("train_fraction =", args.train_fraction)
    print("random_seed    =", args.random_seed)
    
    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    filenames = [os.path.basename(fpath) for fpath in glob.glob(os.path.join(data_dir, "*.csv"))]
    data = [pd.read_csv(os.path.join(data_dir, fname)) for fname in filenames]
    print(f"\nFound {len(filenames)} files in data_dir")
    for fname in filenames:
        print("\t", fname)
    
    print("\nSplit data ...")
    splitted_data = train_test_split(*data, train_size=args.train_fraction, random_state=args.random_seed)
    print("Done.")
    
    print("\nSave data ...")
    for d, fname in zip(splitted_data[::2], filenames):
        d.to_csv(os.path.join(train_dir, fname), index=False)
    for d, fname in zip(splitted_data[1::2], filenames):
        d.to_csv(os.path.join(test_dir, fname), index=False)
    
    print(f"Saved training data to {train_dir}")
    print(f"Saved test data to {test_dir}")