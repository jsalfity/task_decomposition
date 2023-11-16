import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
#         start_step = entry["start_step"]
#         end_step = entry["end_step"]
#         subtask = entry["subtask"]

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
#         start_step = entry["start_step"]
#         end_step = entry["end_step"]
#         subtask = entry["subtask"]

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
        start_step_a = entry["start_step"]
        end_step_a = entry["end_step"]
        subtask_a = entry["subtask"]

        # Plot a box for each subtask in decomposition actual with labels on the left
        plt.plot(
            [start_step_a, end_step_a],
            [i, i],
            marker="o",
            markersize=10,
            label=f"actual - Subtask {subtask_a}",
            color=ACTUAL_COLOR,
        )
        plt.text(
            end_step_a + 1,
            i + 0.1,
            f"{subtask_a}",
            va="center",
            ha="left",
            color=ACTUAL_COLOR,
        )

    for i, entry in enumerate(predicted):
        start_step_b = entry["start_step"]
        end_step_b = entry["end_step"]
        subtask_b = entry["subtask"]

        # Plot a box for each subtask in decomposition predicted with labels on the right
        plt.plot(
            [start_step_b, end_step_b],
            [i, i],
            marker="x",
            markersize=10,
            label=f"predicted - Subtask {subtask_b}",
            color=PREDICTED_COLOR,
        )
        plt.text(
            start_step_b - 1,
            i + 0.1,
            f"{subtask_b}",
            va="center",
            ha="right",
            color=PREDICTED_COLOR,
        )

    # Create a custom legend with two markers, blue for A and red for B
    legend_labels = [
        mpatches.Patch(color=ACTUAL_COLOR, label="actual"),
        mpatches.Patch(color=PREDICTED_COLOR, label="predicted"),
    ]
    plt.legend(handles=legend_labels, loc="upper left")
    plt.xlabel("Number of Steps")
    plt.ylabel("Subtask Number")
    plt.title(title)
    plt.grid(True)
    plt.show()


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
