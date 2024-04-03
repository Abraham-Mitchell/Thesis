import numpy as np
import matplotlib.pyplot as plt
import math

steps = []
error_scale = 0.3  # Scale for movement error
localization_error_mean = 3.0  # Mean of the localization error in feet
localization_error_std = 1.0  # Standard deviation to approximate a 2-6 feet range
number_of_steps = 50
actual_trials = []
num_trials = 1

steps.append([0, 0])  # Initial step

# Generate planned path
for itr, step in enumerate(range(number_of_steps)):
    # Alternating steps between right and up
    if step % 2 == 0:
        steps.append([steps[itr][0] + 1, steps[itr][1]])
    else:
        steps.append([steps[itr][0], steps[itr][1] + 1])

steps = np.array(steps).T  # Transpose for plotting

for i in range(num_trials):
    actual_location = [(0, 0)]  # Start at the origin
    localized_position = [(0, 0)]  # Initial localization also at the origin

    for itr, step in enumerate(range(number_of_steps)):
        # Determine next target position from steps
        target_pos = steps[:, step + 1]  # +1 as steps includes the initial step

        # Movement error
        error = abs(np.random.normal() * error_scale)
        angle = np.random.uniform() * 360
        error_x = error * math.cos(math.radians(angle))
        error_y = error * math.sin(math.radians(angle))

        # Apply correction based on last localized position vs. target position
        correction = target_pos - np.array(localized_position[-1])
        correction_norm = correction / np.linalg.norm(correction) if np.linalg.norm(correction) > 0 else correction

        # Actual movement including error and correction
        next_pos = np.array(actual_location[-1]) + correction_norm + np.array([error_x, error_y])

        actual_location.append(next_pos.tolist())

        # Localization with error
        loc_error = np.random.normal(loc=localization_error_mean, scale=localization_error_std, size=2)
        localized_x = next_pos[0] + loc_error[0] * math.cos(math.radians(angle))
        localized_y = next_pos[1] + loc_error[1] * math.sin(math.radians(angle))
        localized_position.append([localized_x, localized_y])

    actual_location = np.array(actual_location).T
    localized_position = np.array(localized_position).T
    actual_trials.append(actual_location)

    # Plot actual path taken with errors
    plt.plot(actual_location[0], actual_location[1], label='Actual Path', marker='o')
    # Plot localized positions
    plt.plot(localized_position[0], localized_position[1], 'x', label='Localized Positions', markersize=5)

# Plot planned steps
plt.plot(steps[0], steps[1], 'k--', label='Planned Path')
plt.legend()
plt.show()
