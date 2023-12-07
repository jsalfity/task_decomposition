GROUND_TRUTH = {
    "lift": [
        (0, 12, "Move to above cube", 1.1),
        (13, 14, "Grasp Cube", 1.2),
        (15, 39, "Lift Cube", 2.1),
    ],
    "stack": [
        (0, 7, "Move to above Cube A", 1.1),
        (8, 19, "Moving directly down to Cube A", 1.2),
        (20, 21, "Grasping Cube A", 1.3),
        (22, 35, "Vertically picking up Cube A", 2.1),
        (36, 43, "Aligning Cube A with Cube B", 3.1),
        (44, 49, "Moving Cube A vertically down to Cube B", 3.2),
        (50, 53, "Releasing Cube A onto Cube B", 3.2),
        (54, 79, "Returning Home", 4.1),
    ],
    "open_door": [
        (0, 6, "Align manipulator height with Door", 1.1),
        (7, 19, "Get closer to Door", 1.2),
        (20, 41, "Turn Door handle", 2.1),
        (42, 74, "Open Door", 2.2),
    ],
    "picknplace": [
        (0, 9, "Move to above can", 1.1),
        (10, 17, "Move down to can", 1.2),
        (18, 19, "Grasp can", 1.3),
        (20, 26, "Pick up can", 2.1),
        (27, 119, "Move to goal position", 2.2),
    ],
    "tool_hang": [
        (0, 60, "Move to first tool", 1.1),
        (61, 75, "Fine tune position for grasping", 1.2),
        (76, 80, "Grasp tool", 1.3),
        (78, 150, "Move around the general region of the fixture", 2.1),
        (151, 195, "Fine tune to get ready for insertion", 2.2),
        (196, 235, "Attempt to insert tool", 2.3),
        (236, 290, "Insert tip of tool into fixture", 2.4),
        (291, 400, "Insert full length of tool into fixture", 2.5),
        (401, 470, "Move to second tool", 3.1),
        (471, 500, "Fine tune position for grasping", 3.2),
        (500, 505, "Grasp second tool", 3.3),
        (506, 615, "Move to fixture", 4.1),
        (615, 670, "Fine tune position for hanging", 4.2),
        (671, 680, "Release tool", 4.3),
    ],
}
