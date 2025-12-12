import numpy as np
import matplotlib.pyplot as plt

def apply_transform(pts, R, t):
    return (R.dot(pts.T)).T + t

def visualize_matching(slam_pts, real_pts, slam_colors, real_colors, result):
    """
    Run match_and_estimate and visualize the results.
    """
    # Run matching
    R = result['R']
    t = result['t']
    matches = result['matches']
    rms = result['rms']

    # Transform SLAM points
    slam_transformed = apply_transform(slam_pts, R, t)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Separate by color
    slam_yellow = slam_transformed[np.array(slam_colors) == True]
    slam_blue = slam_transformed[np.array(slam_colors) == False]
    real_yellow = real_pts[np.array(real_colors) == True]
    real_blue = real_pts[np.array(real_colors) == False]

    # Plot real cones (filled, larger)
    ax.scatter(real_yellow[:, 0], real_yellow[:, 1],
               c='orange', marker='s', s=200, edgecolors='darkorange',
               linewidths=2, label='Real Yellow', zorder=3, alpha=0.9)
    ax.scatter(real_blue[:, 0], real_blue[:, 1],
               c='blue', marker='s', s=200, edgecolors='darkblue',
               linewidths=2, label='Real Blue', zorder=3, alpha=0.9)

    # Plot SLAM cones (hollow, smaller)
    ax.scatter(slam_yellow[:, 0], slam_yellow[:, 1],
               c='none', marker='o', s=120, edgecolors='goldenrod',
               linewidths=2.5, label='SLAM Yellow', zorder=4, alpha=0.8)
    ax.scatter(slam_blue[:, 0], slam_blue[:, 1],
               c='none', marker='s', s=120, edgecolors='cornflowerblue',
               linewidths=2.5, label='SLAM Blue', zorder=4, alpha=0.8)

    # Draw matching lines
    for slam_idx, real_idx, dist in matches:
        pt_slam = slam_transformed[slam_idx]
        pt_real = real_pts[real_idx]

        # Color based on distance
        if dist < 0.1:
            color = 'green'
            alpha = 0.4
        elif dist < 0.3:
            color = 'yellow'
            alpha = 0.5
        else:
            color = 'red'
            alpha = 0.6

        ax.plot([pt_slam[0], pt_real[0]], [pt_slam[1], pt_real[1]],
                color=color, linestyle='--', linewidth=1.5, alpha=alpha, zorder=2)

        # Distance label
        mid_x = (pt_slam[0] + pt_real[0]) / 2
        mid_y = (pt_slam[1] + pt_real[1]) / 2
        ax.text(mid_x, mid_y, f'{dist:.2f}', fontsize=7,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.7))

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X coordinate (cm)', fontsize=12)
    ax.set_ylabel('Y coordinate (cm)', fontsize=12)
    ax.set_title(f'Cone Matching (RMS Error: {rms:.3f}cm)',
                 fontsize=14, fontweight='bold')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)

    # Statistics
    stats_text = f'Matches: {len(matches)}\n'
    stats_text += f'SLAM cones: {len(slam_pts)}\n'
    stats_text += f'Real cones: {len(real_pts)}\n'
    stats_text += f'RMS: {rms:.4f}cm'

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.axis('equal')
    return fig, ax, result, plt
