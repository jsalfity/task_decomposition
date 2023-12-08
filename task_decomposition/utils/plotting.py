import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import base64

# def visualize_trajectory_decompositions(actual, predicted, title):
#     """
#     Visualize two trajectory decompositions in a single plot.

#     Args:
#     actual (list of dictionaries): The first trajectory decomposition.
#     predicted (list of dictionaries): The second trajectory decomposition.
#     title (str): The title for the plot.
#     """
#     plt.figure(figsize=(12, 6))

#     for i, entry in enumerate(actual):
#         start_step = entry[0]
#         end_step = entry[1]
#         subtask = entry[2]

#         # Plot a box for each subtask in decomposition actual
#         plt.plot(
#             [start_step, end_step],
#             [i, i],
#             marker="o",
#             markersize=10,
#             label=f"actual - Subtask {subtask}",
#             color="blue",
#         )

#     for i, entry in enumerate(predicted):
#         start_step = entry[0]
#         end_step = entry[1]
#         subtask = entry[2]

#         # Plot a box for each subtask in decomposition predicted
#         plt.plot(
#             [start_step, end_step],
#             [i, i],
#             marker="x",
#             markersize=10,
#             label=f"predicted - Subtask {subtask}",
#             color="red",
#         )

#     plt.xlabel("Step Number")
#     plt.ylabel("Subtask Number")
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

ACTUAL_COLOR = "blue"
PREDICTED_COLOR = "red"


def format_string(s: list) -> str:
    """
    Add a newline after every 3 words in the list
    """
    x = s.split(" ")
    s_format = []
    for i in range(len(x)):
        s_format.append(x[i])
        s_format.append("\n") if i % 3 == 0 else None
    return " ".join(s_format)


def visualize_trajectory_decompositions(actual, predicted, title):
    """
    Visualize two trajectory decompositions in a single plot with labeled subtasks on both sides.

    Args:
    actual (list of dictionaries): The first trajectory decomposition.
    predicted (list of dictionaries): The second trajectory decomposition.
    title (str): The title for the plot.
    """
    plt.figure(figsize=(12, 6))

    for i, entry in enumerate(actual):
        start_step_a = entry[0]
        end_step_a = entry[1]
        subtask_a = entry[2]

        # Plot a box for each subtask in decomposition actual with labels on the left
        plt.plot(
            [start_step_a, end_step_a],
            [i, i],
            marker="o",
            markersize=5,
            label=f"actual - Subtask {subtask_a}",
            color=ACTUAL_COLOR,
        )
        plt.text(
            end_step_a + 1,
            i + 0.1,
            format_string(subtask_a),
            va="center",
            ha="left",
            color=ACTUAL_COLOR,
        )

    for i, entry in enumerate(predicted):
        start_step_b = entry[0]
        end_step_b = entry[1]
        subtask_b = entry[2]

        # Plot a box for each subtask in decomposition predicted with labels on the right
        plt.plot(
            [start_step_b, end_step_b],
            [i, i],
            marker="x",
            markersize=5,
            label=f"predicted - Subtask {subtask_b}",
            color=PREDICTED_COLOR,
        )
        plt.text(
            start_step_b - 1,
            i + 0.1,
            format_string(subtask_b),
            va="center",
            ha="right",
            color=PREDICTED_COLOR,
        )

    # Create a custom legend with two markers, blue for A and red for B
    legend_labels = [
        mpatches.Patch(color=ACTUAL_COLOR, label="actual"),
        mpatches.Patch(color=PREDICTED_COLOR, label="predicted"),
    ]
    plt.tight_layout()
    plt.legend(handles=legend_labels, loc="upper left")
    plt.xlabel("Step Number")
    plt.ylabel("Subtask Number")
    plt.title(title)
    plt.grid(False)
    plt.show()


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Redefine the add_intervals_with_colors function to position the text at the edges
def add_intervals_with_colors(intervals, row, color_map, ax, angle):
    for start, end, description, hierarchy in intervals:
        color = color_map[int(hierarchy) - 1]
        rect_width = end - start if end - start > 0 else 1
        rect = mpatches.Rectangle(
            (start, row), rect_width, 0.8, edgecolor="black", facecolor=color
        )
        ax.add_patch(rect)

        # Position the text at the top and bottom edges of the bars
        if row == 1:  # For the top bar, start the text at the top edge
            text_y = row + 0.8
            va = "bottom"
        else:  # For the bottom bar, start the text at the bottom edge
            text_y = row
            va = "top"

        # Place the text at a 45 degree angle for actual and -45 for predicted
        ax.text(
            (start + end) / 2,
            text_y,
            format_string(description),
            ha="left",
            va=va,
            fontsize="large",
            color="black",
            rotation=angle,
            rotation_mode="anchor",
        )


def box_plots(actual, predicted, env):
    # Set up the plot again
    fig, ax = plt.subplots(figsize=(14, 4))
    color_map = ["green", "blue", "purple", "orange", "yellow"]
    max_hierarchy = max(
        max([int(a[3]) for a in actual]), max([int(p[3]) for p in predicted])
    )
    color_map = color_map[:max_hierarchy]

    # Add actual intervals with color map and angle the text at 45 degrees
    add_intervals_with_colors(actual, 1, color_map, ax, angle=45)

    # Add predicted intervals with color map and angle the text at -45 degrees
    add_intervals_with_colors(predicted, 0, color_map, ax, angle=-45)

    # Set the limits, labels, and title
    ax.set_xlim(-5, max(actual[-1][1], predicted[-1][1]) + 5)
    ax.set_ylim(-8, 10)
    ax.set_yticks([0.4, 1.4])
    ax.set_yticklabels(["Predicted", "Actual"], fontsize=12)
    ax.set_xlabel("Step Number")
    ax.set_title(f"Actual vs Predicted Intervals for {env}", fontsize=12)

    # Place the legend in the upper right corner
    legend_elements = [
        Line2D([0], [0], color=c, lw=4, label=f"Hierarchy Level: {n}")
        for n, c in enumerate(color_map)
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 1), ncol=1)

    plt.tight_layout()
    plt.show()
