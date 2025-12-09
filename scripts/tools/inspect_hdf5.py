#!/usr/bin/env python3
"""Inspect HDF5 demonstration files."""

import argparse
import h5py
import numpy as np

def inspect(path, show_samples=3):
    with h5py.File(path, "r") as f:
        print(f"\n{'='*60}")
        print(f"FILE: {path}")
        print(f"{'='*60}")
        
        # Top-level attributes
        if f.attrs:
            print("\n[Attributes]")
            for k, v in f.attrs.items():
                print(f"  {k}: {v}")
        
        # Data group
        if "data" in f:
            demos = list(f["data"].keys())
            print(f"\n[Demos] {len(demos)} total")
            
            for i, demo_name in enumerate(demos[:show_samples]):
                demo = f["data"][demo_name]
                print(f"\n  {demo_name}:")
                
                # Attributes
                for k, v in demo.attrs.items():
                    print(f"    {k}: {v}")
                
                # Datasets
                for key in demo.keys():
                    ds = demo[key]
                    if isinstance(ds, h5py.Dataset):
                        print(f"    {key}: shape={ds.shape}, dtype={ds.dtype}")
                        # Show sample values
                        data = ds[:]
                        if len(data) > 0:
                            print(f"      first: {data[0][:5] if data[0].ndim > 0 else data[0]}...")
                            print(f"      last:  {data[-1][:5] if data[-1].ndim > 0 else data[-1]}...")
                    elif isinstance(ds, h5py.Group):
                        print(f"    {key}/ (group with {len(ds.keys())} items)")
            
            if len(demos) > show_samples:
                print(f"\n  ... and {len(demos) - show_samples} more demos")
        
        print(f"\n{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect HDF5 demo files")
    parser.add_argument("file", help="Path to HDF5 file")
    parser.add_argument("--samples", type=int, default=3, help="Number of demos to show")
    args = parser.parse_args()
    inspect(args.file, args.samples)

