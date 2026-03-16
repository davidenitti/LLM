import json
import matplotlib.pyplot as plt
import numpy as np

# Load a sample ARC-AGI task
with open("arc_agi/arc-agi_concept-challenges.json") as f:
    data = json.load(f)

# Pick the first task and the first train example
first_task = next(iter(data.values()))
example = first_task["train"][0]
input_grid = np.array(example["input"])
output_grid = np.array(example["output"])

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(input_grid, cmap="tab20", interpolation="nearest")
axs[0].set_title("Input")
axs[0].axis("off")
axs[1].imshow(output_grid, cmap="tab20", interpolation="nearest")
axs[1].set_title("Output")
axs[1].axis("off")
plt.tight_layout()
plt.show()
