import h5py, numpy as np
path="/home/radu/IsaacLab-Tactile/datasets/pick_place_basket_tacex_100.hdf5"
with h5py.File(path,"r") as f:
    demo=sorted(f["data"].keys())[0]
    g=f["data"][demo]["actions"][:,6]
    print("gripper min/max:", float(g.min()), float(g.max()))
    print("unique (first 200):", np.unique(g[:200])[:20])