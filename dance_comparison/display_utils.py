def draw_frame(ax, pose_data, joints_names, animation_parents, frame_idx):
    ax.clear()

    # Get the data for the current frame
    frame_data = pose_data[frame_idx]

    # Plot the skeleton connections
    for child_idx, parent_idx in enumerate(animation_parents):
        if parent_idx == -1:
            continue  # Skip the root (no parent)

        # Get coordinates for parent and child
        parent_coord = frame_data[parent_idx, :2]  # X, Y of the parent
        child_coord = frame_data[child_idx, :2]  # X, Y of the child

        # Draw a line between the parent and child
        ax.plot(
            [parent_coord[0], child_coord[0]],
            [parent_coord[1], child_coord[1]],
            "k-",
            lw=2,
        )

    # Plot each joint and annotate it with its name
    for idx, (x, y, _) in enumerate(frame_data):
        ax.scatter(x, y, color="red", s=40, zorder=5)  # Plot the joint as a red dot
        ax.text(x, y, f"{joints_names[idx]}", fontsize=9, color="blue", zorder=10)

    # Configure the plot
    ax.set_title(
        f"2D Skeleton Visualization - Frame {frame_idx + 1}/{pose_data.shape[0]}"
    )
    ax.axis("equal")
    # ax.invert_yaxis()  # Invert Y-axis to match most 2D coordinate systems for poses
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
