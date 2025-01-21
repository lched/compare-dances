import matplotlib.pyplot as plt


def draw_skeleton(pose_data, parents, joints_names=None):
    # Ensure pose_data has the correct shape
    assert (
        pose_data.shape[1] >= 2
    ), "pose_data must have at least two columns for X and Y coordinates"

    # Plot the skeleton connections based on the parent-child relationships
    for child_idx, parent_idx in enumerate(parents):
        if parent_idx == -1:
            continue  # Skip the root (no parent)

        # Get coordinates for parent and child
        child_coord = pose_data[child_idx]  # X, Y of the child
        parent_coord = pose_data[parent_idx]  # X, Y of the parent

        # Draw a line between the parent and child
        plt.plot(
            [parent_coord[0], child_coord[0]],
            [parent_coord[1], child_coord[1]],
            "k-",
            lw=2,
        )

    # Plot each joint and annotate it with its name
    for idx, (x, y) in enumerate(pose_data):
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
    # plt.show()


# # Dynamic version
# def draw_skeleton(pose_data, parents, joints_names=None, color="red", label=None):
#     # Ensure pose_data has the correct shape
#     assert pose_data.shape[1] >= 2, "pose_data must have at least two columns for X and Y coordinates"

#     # Plot the skeleton connections based on the parent-child relationships
#     lines_drawn = False  # Track if any lines are drawn to apply label
#     for child_idx, parent_idx in enumerate(parents):
#         if parent_idx == -1:
#             continue  # Skip the root (no parent)

#         # Get coordinates for parent and child
#         child_coord = pose_data[child_idx]  # X, Y of the child
#         parent_coord = pose_data[parent_idx]  # X, Y of the parent

#         # Draw a line between the parent and child
#         plt.plot(
#             [parent_coord[0], child_coord[0]],
#             [parent_coord[1], child_coord[1]],
#             color=color,
#             lw=2,
#             label=label if not lines_drawn else None,  # Apply label only once
#         )
#         lines_drawn = True

#     # Plot each joint
#     plt.scatter(
#         pose_data[:, 0],  # X-coordinates
#         pose_data[:, 1],  # Y-coordinates
#         color=color,
#         s=40,
#         zorder=5,
#         label=label if not lines_drawn else None,  # Apply label here if no lines
#     )

#     # Annotate joints with their names
#     if joints_names:
#         for idx, (x, y) in enumerate(pose_data):
#             plt.text(x, y, f"{joints_names[idx]}", fontsize=9, color=color, zorder=10)
