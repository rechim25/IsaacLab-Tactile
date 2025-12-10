#!/usr/bin/env python3
"""Inspect and visualize HDF5 demonstration files."""

import argparse
import h5py
import numpy as np

def inspect(path, show_samples=3, plot=False):
    with h5py.File(path, "r") as f:
        print(f"\n{'='*60}")
        print(f"FILE: {path}")
        print(f"{'='*60}")
        
        # Attributes
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
                
                for k, v in demo.attrs.items():
                    print(f"    {k}: {v}")
                
                for key in demo.keys():
                    ds = demo[key]
                    if isinstance(ds, h5py.Dataset):
                        print(f"    {key}: shape={ds.shape}, dtype={ds.dtype}")
            
            if len(demos) > show_samples:
                print(f"\n  ... and {len(demos) - show_samples} more demos")
            
            # Plot if requested
            if plot and len(demos) > 0:
                plot_demo(f["data"][demos[0]])
        
        print(f"\n{'='*60}\n")


def plot_demo(demo_group):
    """Plot tactile and camera data from a demo."""
    import matplotlib.pyplot as plt
    
    # Find available image data
    image_keys = [k for k in demo_group.keys() if any(x in k for x in ["tactile", "rgb"])]
    
    if not image_keys:
        print("\n[No image data to plot]")
        return
    
    print(f"\n[Plotting: {image_keys}]")
    
    # Get middle frame
    num_steps = demo_group[image_keys[0]].shape[0]
    frames_to_show = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]
    frames_to_show = sorted(set(f for f in frames_to_show if f < num_steps))
    
    fig, axes = plt.subplots(len(image_keys), len(frames_to_show), figsize=(3 * len(frames_to_show), 3 * len(image_keys)))
    if len(image_keys) == 1:
        axes = axes.reshape(1, -1)
    if len(frames_to_show) == 1:
        axes = axes.reshape(-1, 1)
    
    for row, key in enumerate(image_keys):
        data = demo_group[key][:]
        for col, frame_idx in enumerate(frames_to_show):
            img = data[frame_idx]
            # Handle different formats
            if img.ndim == 2:  # Grayscale
                axes[row, col].imshow(img, cmap="gray")
            elif img.shape[-1] == 4:  # RGBA
                axes[row, col].imshow(img[..., :3])
            else:  # RGB
                axes[row, col].imshow(img)
            
            axes[row, col].set_title(f"{key}\nt={frame_idx}")
            axes[row, col].axis("off")
    
    plt.tight_layout()
    plt.savefig("hdf5_preview.png", dpi=150)
    print(f"[Saved plot to hdf5_preview.png]")
    plt.show()


def plot_trajectory(path):
    """Plot action/joint trajectories."""
    import matplotlib.pyplot as plt
    
    with h5py.File(path, "r") as f:
        if "data" not in f or len(f["data"].keys()) == 0:
            print("No data to plot")
            return
        
        demo = f["data"][list(f["data"].keys())[0]]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Actions
        if "actions" in demo:
            actions = demo["actions"][:]
            axes[0, 0].plot(actions[:, :3], label=["dx", "dy", "dz"])
            axes[0, 0].set_title("Actions (position)")
            axes[0, 0].legend()
            axes[0, 0].set_xlabel("Step")
            
            axes[0, 1].plot(actions[:, -1], label="gripper")
            axes[0, 1].set_title("Gripper command")
            axes[0, 1].legend()
        
        # Joint positions
        if "joint_pos" in demo:
            jp = demo["joint_pos"][:]
            axes[1, 0].plot(jp[:, :7])
            axes[1, 0].set_title("Joint positions (arm)")
            axes[1, 0].set_xlabel("Step")
        
        # EE position
        if "ee_pos" in demo:
            ee = demo["ee_pos"][:]
            axes[1, 1].plot(ee, label=["x", "y", "z"])
            axes[1, 1].set_title("End-effector position")
            axes[1, 1].legend()
            axes[1, 1].set_xlabel("Step")
        
        plt.tight_layout()
        plt.savefig("hdf5_trajectory.png", dpi=150)
        print(f"[Saved trajectory plot to hdf5_trajectory.png]")
        plt.show()


def plot_forces(path):
    """Plot pseudo-force data from tactile sensors."""
    import matplotlib.pyplot as plt
    
    with h5py.File(path, "r") as f:
        if "data" not in f or len(f["data"].keys()) == 0:
            print("No data to plot")
            return
        
        demo = f["data"][list(f["data"].keys())[0]]
        
        # Check for force data
        force_keys = [k for k in demo.keys() if "force" in k.lower()]
        if not force_keys:
            print("[No force data to plot]")
            return
        
        print(f"[Plotting forces: {force_keys}]")
        
        # Determine layout
        n_plots = len(force_keys)
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
        
        colors = {'Fx': 'r', 'Fy': 'g', 'Fz': 'b'}
        
        for idx, key in enumerate(force_keys):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            force_data = demo[key][:]
            steps = np.arange(len(force_data))
            
            # Plot each force component
            if force_data.ndim == 2 and force_data.shape[1] >= 3:
                ax.plot(steps, force_data[:, 0], colors['Fx'], label='Fx (shear X)', alpha=0.8)
                ax.plot(steps, force_data[:, 1], colors['Fy'], label='Fy (shear Y)', alpha=0.8)
                ax.plot(steps, force_data[:, 2], colors['Fz'], label='Fz (normal)', alpha=0.8)
                
                # Add magnitude
                magnitude = np.linalg.norm(force_data, axis=1)
                ax.plot(steps, magnitude, 'k--', label='|F| (magnitude)', alpha=0.5)
            else:
                ax.plot(steps, force_data, label=key)
            
            ax.set_title(key.replace('_', ' ').title())
            ax.set_xlabel("Step")
            ax.set_ylabel("Pseudo-Force (a.u.)")
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_plots, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig("hdf5_forces.png", dpi=150)
        print(f"[Saved force plot to hdf5_forces.png]")
        plt.show()


def _render_force_plot_single(ax, force_data, current_frame, num_frames, title, colors):
    """Render a single force plot on given axes."""
    if force_data is None:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=10)
        ax.set_title(title, fontsize=9)
        return
    
    steps = np.arange(num_frames)
    labels = ['Fx (shear)', 'Fy (shear)', 'Fz (normal)']
    
    # Plot full trajectory faded
    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.plot(steps, force_data[:, i], color=color, alpha=0.2, linewidth=1)
    
    # Plot up to current frame
    for i, (color, label) in enumerate(zip(colors, labels)):
        ax.plot(steps[:current_frame+1], force_data[:current_frame+1, i], 
                color=color, linewidth=1.5, label=label)
        ax.scatter([current_frame], [force_data[current_frame, i]], 
                   c=color, s=30, zorder=5)
    
    ax.axvline(x=current_frame, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0, num_frames)
    ax.set_xlabel('Step', fontsize=7)
    ax.set_ylabel('Force', fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.legend(loc='upper right', fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=6)


def _render_force_plots(force_left, force_right, current_frame, num_frames, size=(640, 240)):
    """Render two force plots side-by-side (left and right finger)."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(size[0]/100, size[1]/100), dpi=100)
    
    colors_left = ['#e74c3c', '#2ecc71', '#3498db']   # Red, Green, Blue
    colors_right = ['#c0392b', '#27ae60', '#2980b9']  # Darker variants
    
    _render_force_plot_single(ax1, force_left, current_frame, num_frames, 
                              'Left Finger Force', colors_left)
    _render_force_plot_single(ax2, force_right, current_frame, num_frames, 
                              'Right Finger Force', colors_right)
    
    fig.tight_layout(pad=0.5)
    
    # Convert to image
    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    
    return img


def create_video(path, output_video="demo_video.mp4", demo_idx=0, fps=30):
    """Create split-screen video: table RGB, wrist RGB, tactile left, tactile right, force plot."""
    import cv2
    
    with h5py.File(path, "r") as f:
        if "data" not in f:
            print("No data found")
            return
        
        demos = list(f["data"].keys())
        if demo_idx >= len(demos):
            print(f"Demo {demo_idx} not found (only {len(demos)} demos)")
            return
        
        demo = f["data"][demos[demo_idx]]
        print(f"[Creating video from {demos[demo_idx]}]")
        
        # Get available streams
        streams = {}
        stream_keys = ["rgb_table", "rgb_wrist", "tactile_left", "tactile_right"]
        labels = ["Table Camera", "Wrist Camera", "Tactile Left", "Tactile Right"]
        
        for key in stream_keys:
            if key in demo:
                streams[key] = demo[key][:]
                print(f"  Found {key}: {streams[key].shape}")
        
        # Check for force data
        force_left = demo["force_geometric_left"][:] if "force_geometric_left" in demo else None
        force_right = demo["force_geometric_right"][:] if "force_geometric_right" in demo else None
        has_force = force_left is not None or force_right is not None
        
        if has_force:
            print(f"  Found force data: left={force_left is not None}, right={force_right is not None}")
        
        if not streams:
            print("No video streams found")
            return
        
        # Get frame count and determine layout
        num_frames = min(s.shape[0] for s in streams.values())
        num_streams = len(streams)
        
        # Calculate grid layout for video streams only
        if num_streams <= 2:
            grid = (1, 2)
        elif num_streams <= 4:
            grid = (2, 2)
        elif num_streams <= 6:
            grid = (2, 3)
        else:
            grid = (3, 3)
        
        # Target size per panel
        panel_size = (320, 240)
        # Add extra row for force plots if present
        num_rows = grid[0] + (1 if has_force else 0)
        frame_size = (panel_size[0] * grid[1], panel_size[1] * num_rows)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)
        
        stream_list = list(streams.items())
        label_list = [labels[stream_keys.index(k)] for k, _ in stream_list]
        
        print(f"  Creating {num_frames} frames at {fps} fps...")
        
        for frame_idx in range(num_frames):
            # Create composite frame
            composite = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
            
            # Draw image streams
            for i, ((key, data), label) in enumerate(zip(stream_list, label_list)):
                row, col = i // grid[1], i % grid[1]
                y_start, x_start = row * panel_size[1], col * panel_size[0]
                
                # Get and process frame
                img = data[frame_idx]
                if img.ndim == 2:  # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[-1] == 4:  # RGBA
                    img = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)
                else:  # RGB
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Resize
                img = cv2.resize(img, panel_size)
                
                # Add label
                cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img, f"t={frame_idx}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Place in composite
                composite[y_start:y_start + panel_size[1], x_start:x_start + panel_size[0]] = img
            
            # Draw force plots (spans full bottom row)
            if has_force:
                row = (len(stream_list) + grid[1] - 1) // grid[1]  # Next row
                y_start = row * panel_size[1]
                
                # Render two plots side-by-side spanning full width
                full_width = panel_size[0] * grid[1]
                force_img = _render_force_plots(force_left, force_right, frame_idx, num_frames, 
                                                (full_width, panel_size[1]))
                force_img = cv2.cvtColor(force_img, cv2.COLOR_RGB2BGR)
                force_img = cv2.resize(force_img, (full_width, panel_size[1]))
                
                composite[y_start:y_start + panel_size[1], 0:full_width] = force_img
            
            out.write(composite)
        
        out.release()
        print(f"[Saved video to {output_video}]")


def create_all_videos(path, output_dir="demo_videos", fps=30):
    """Create videos for all demos in the dataset."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(path, "r") as f:
        if "data" not in f:
            return
        demos = list(f["data"].keys())
    
    for i, demo_name in enumerate(demos):
        output_video = os.path.join(output_dir, f"{demo_name}.mp4")
        create_video(path, output_video, demo_idx=i, fps=fps)
    
    print(f"[Created {len(demos)} videos in {output_dir}/]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect HDF5 demo files")
    parser.add_argument("file", help="Path to HDF5 file")
    parser.add_argument("--samples", type=int, default=3, help="Number of demos to show")
    parser.add_argument("--plot", action="store_true", help="Plot tactile/camera images")
    parser.add_argument("--trajectory", action="store_true", help="Plot action/joint trajectories")
    parser.add_argument("--forces", action="store_true", help="Plot pseudo-force data")
    parser.add_argument("--video", action="store_true", help="Create split-screen video (first demo)")
    parser.add_argument("--video-all", action="store_true", help="Create videos for all demos")
    parser.add_argument("--video-output", type=str, default="demo_video.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--demo-idx", type=int, default=0, help="Demo index for single video")
    args = parser.parse_args()
    
    inspect(args.file, args.samples, args.plot)
    
    if args.trajectory:
        plot_trajectory(args.file)
    
    if args.forces:
        plot_forces(args.file)
    
    if args.video:
        create_video(args.file, args.video_output, args.demo_idx, args.fps)
    
    if args.video_all:
        create_all_videos(args.file, "demo_videos", args.fps)
