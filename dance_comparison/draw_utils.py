import matplotlib.pyplot as plt


def draw_skeleton(pose_data, parents, joints_names=None, axes=[0, 1]):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Ensure pose_data has the correct shape
    assert (
        pose_data.shape[1] >= 2
    ), "pose_data must have at least two columns for X and Y coordinates"

    # Plot the skeleton connections based on the parent-child relationships
    for child_idx, parent_idx in enumerate(parents):
        if parent_idx == -1:
            continue  # Skip the root (no parent)

        # Get coordinates for parent and child
        child_coord = pose_data[child_idx, axes]  # X, Y of the child
        parent_coord = pose_data[parent_idx, axes]  # X, Y of the parent

        # Draw a line between the parent and child
        plt.plot(
            [parent_coord[0], child_coord[0]],
            [parent_coord[1], child_coord[1]],
            "k-",
            lw=2,
        )

    # Plot each joint and annotate it with its name
    for idx, (x, y, _) in enumerate(pose_data):
        plt.scatter(x, y, color="red", s=40, zorder=5)  # Plot the joint as a red dot
        if joints_names:
            plt.text(x, y, f"{joints_names[idx]}", fontsize=9, color="blue", zorder=10)

    # Configure the plot
    plt.title("2D Skeleton Visualization")
    plt.axis("equal")
    # plt.gca().invert_yaxis()  # Invert Y-axis to match most 2D coordinate systems for poses
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()
