
plant_densities = {
    'Asparagus': 1, 'Basil': 4, 'Lima Beans': 16, 'Snap Beans': 16, 'Beats': 9,
    'Brussel Sprouts': 1, 'Carrots': 16, 'Chinese Cabbage': 4, 'Collard': 1,
    'Corn': 2, 'Cucumber': 1, 'Dill': 4, 'Eggplant': 1, 'Endive': 2, 'Garlic': 9,
    'Kale': 1, 'Kholrabi': 4, 'Leeks': 16, 'Head Lettuce': 1, 'Leaf Lettuce': 4,
    'Mustard': 16, 'New Zealand Spinach': 1, 'Okra': 1, 'Onion Bulb': 9,
    'Onion Bunching': 16, 'Parsley': 9, 'Parsnips': 16, 'Peppers': 1, 'Potatoes': 1,
    'radish': 16,  # Example: 16 radishes per square foot
    'tomato': 1,   # 1 tomato per square foot
    # Add more plant types as needed
}

#weed biomass 55% lower in high crop density uniform pattern
#densites with 1 plant have 8 weeds
#55% drop from 5 gives 4.4 which will be rounded to 4
# 1 to 8... 2 to 7... 4 to 6... 9 to 5... 16 to 4 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_maze(size, plant_types, plant_ratio = 0.3):
    maze = np.ones((size, size), dtype=np.int8)
    density_layer = np.zeros((size, size), dtype=np.int8)
    
    # Count the total number of obstacle cells
    total_obstacles = np.sum(maze == 1)
    
    # Determine the number of obstacles to be converted into plant areas based on the plant_ratio
    num_plant_cells = int(total_obstacles * plant_ratio)
    

    for x in range(1, size, 2):
        for y in range(1, size, 2):
            maze[x, y] = 0
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            np.random.shuffle(directions)
            for dx, dy in directions[:2]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < size and 0 <= ny < size:
                    maze[nx, ny] = 0
                    maze[x + dx//2, y + dy//2] = 0
                
                # else:
                #     plant_type = np.random.choice(list(plant_types.keys()))
                #     density_layer[i, j] = plant_densities[plant_type]
                    
    # Randomly select obstacles to convert into plant areas
    plant_positions = []
    while len(plant_positions) < num_plant_cells:
        x, y = np.random.randint(0, size, size=2)
        if maze[x, y] == 1 and (x, y) not in plant_positions:  # If it's an obstacle and not already selected
            maze[x, y] = 2  # Convert to a plant area
            plant_positions.append((x, y))
            plant_type = np.random.choice(list(plant_types.keys()))
            density_layer[x, y] = plant_densities[plant_type]                 

    return maze, density_layer

def find_start_end_points(maze, number_of_points=2):
    points = []
    size = maze.shape[0]
    while len(points) < number_of_points:
        x, y = np.random.randint(1, size-1, size=2)
        if maze[x, y] == 0 and (x, y) not in points:
            points.append((x, y))
    return points



def get_neighbors(maze, node):
    """Get all possible neighbors for a given node in the maze."""
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, Right, Down, Left
    neighbors = []
    for dx, dy in directions:
        nx, ny = node[0] + dx, node[1] + dy
        if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx][ny] == 0:
            neighbors.append((nx, ny))
    return neighbors


def simulate_gps_error(actual_position, error_range):
    error_x = np.random.randint(-error_range, error_range + 1)
    error_y = np.random.randint(-error_range, error_range + 1)
    perceived_position = (actual_position[0] + error_x, actual_position[1] + error_y)
    return perceived_position




def round_robin_dfs(maze, start_points, weed_map, sigma = 0.95, num_runs=30, gps_error_range=1):
    rows, cols = len(maze), len(maze[0])
    checked_plants = np.full((rows, cols), False, dtype=bool)  # Tracks if a plant has been checked for weeds

    def grow_weeds():
        for x in range(rows):
            for y in range(cols):
                if weed_map[x][y] != -1:  # Only grow weeds in valid spots
                    weed_map[x][y] += 4  # Adjust growth logic as needed

    def check_and_eliminate_weeds(x, y):
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if weed_map[nx][ny] > 0 and not checked_plants[nx][ny] and np.random.rand() < sigma:
                    weed_map[nx][ny] = 0  # Weed elimination based on sigma
                    checked_plants[nx][ny] = True  # Mark as checked

    while num_runs > 0:
        stacks = [[start] for start in start_points]
        shared_visited = set()
        steps_taken = [0] * len(start_points)  # Initialize steps counter for each DFS

        while any(stacks):
            for i, stack in enumerate(stacks):
                if stack:
                    current = stack.pop()
                    # Apply GPS error here
                    perceived_current = simulate_gps_error(current, gps_error_range)
                    if perceived_current not in shared_visited:
                        shared_visited.add(perceived_current)
                        steps_taken[i] += 1  # Increment steps for this DFS
                        check_and_eliminate_weeds(perceived_current[0], perceived_current[1])  # Adjusted for GPS error
                        
                        for nx, ny in get_neighbors(maze, perceived_current):
                            if (nx, ny) not in shared_visited:
                                stack.append((nx, ny))

        # Reporting the total step cost for each DFS
        for i, steps in enumerate(steps_taken):
            print(f"DFS from start point {start_points[i]} took {steps} steps.")

        # Check if all reachable nodes have been visited
        all_nodes_visited = all(cell == 1 or cell == 2 or (x, y) in shared_visited
                                for x, row in enumerate(maze)
                                for y, cell in enumerate(row))
        print(f"All nodes covered by DFS: {all_nodes_visited}")
        print(num_runs)
        num_runs -=1

        if num_runs > 0:
            grow_weeds()  # Grow weeds after each complete round-robin DFS iteration
        checked_plants.fill(False)  # Optionally reset checked_plants if needed


# Example usage remains the same


# Define the size of the maze and generate it
maze_size = 15
maze, density_layer = generate_maze(maze_size, plant_densities)


# Find start/end points in the maze
#change the int parameter for number of random start points
start_end_points = find_start_end_points(maze, 3)  # Generating 3 start/end points

# start_end_points = [(5, 13), (1, 5), (9, 11)]
## uncomment this to manual start point
# start_end_points = [(1,1),(10,10)]

def weed_growth(density_layer):
    # Create a deep copy of the density_layer to modify
    modified_layer = [row[:] for row in density_layer]
    
    # Iterate over each row and column in the copied layer
    for row_index in range(len(modified_layer)):
        for col_index in range(len(modified_layer[row_index])):
            if modified_layer[row_index][col_index] == 1:
                modified_layer[row_index][col_index] = 8
            elif modified_layer[row_index][col_index] == 2:
                modified_layer[row_index][col_index] = 7
            elif modified_layer[row_index][col_index] == 4:
                modified_layer[row_index][col_index] = 6
            elif modified_layer[row_index][col_index] == 9:
                modified_layer[row_index][col_index] = 5
            elif modified_layer[row_index][col_index] == 16:
                modified_layer[row_index][col_index] = 4
            else:
                modified_layer[row_index][col_index] = -1
    
    # Return the modified copy without changing the original density_layer
    return modified_layer


            
# 1 to 8... 2 to 7... 4 to 6... 9 to 5... 16 to 4 
################################
maze = [
 [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
 [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
 [2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 2, 2, 1],
 [2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2],
 [1, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 2, 2, 1],
 [2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2],
 [2, 0, 2, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 2, 1],
 [2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
 [1, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2],
 [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2],
 [2, 0, 2, 0, 2, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2],
 [2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2],
 [1, 2, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 1],
 [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2],
 [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1]
 ]

density_layer = [
[0,  1, 16,  4,  1, 16,  0,  1,  9,  1, 16, 16,  9,  1,  0],
 [ 9,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  1,  0,  1],
 [ 0, 16,  4,  0,  9,  0,  1,  0,  1,  0,  9,  0,  1,  0,  0],
 [16,  0,  0,  0,  9,  0,  0,  0,  0,  0,  1,  0,  0,  0,  4],
 [ 0,  0,  9, 16,  1,  0,  9,  0,  9,  0,  0,  9,  1,  1,  0],
 [16,  0,  0,  0,  0,  0,  4,  0,  0,  0,  0,  0,  0,  0, 16],
 [ 1,  0,  1,  0,  1, 16,  1,  4,  1,  0,  1,  0,  1,  0, 0],
 [ 4,  0,  0,  0,  0,  0,  0,  0,  4,  0,  0,  0,  1,  0,  1],
 [ 0,  0,  1,  0, 16, 16, 16,  0,  1,  9,  1,  9,  9,  0, 16],
 [ 1,  0, 16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16],
 [ 1,  0, 16,  0,  4,  0, 16,  0,  1,  0,  1, 16, 16,  0,  1],
 [ 1,  0,  0,  0, 16,  0,  0,  0,  0,  0, 16,  0,  0,  0,  4],
 [0,  0, 16,  0,  1,  0,  9, 16,  9,  9, 16,  0,  9,  0,  0],
 [ 4,  0,  1,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  9],
 [ 0,  4,  0,  0,  4,  9, 16,  4, 16, 16, 0, 16,  0, 16, 0]]


weed_map = weed_growth(density_layer)

#show maze
print("Maze:")
print(maze)
print("Density Layer:")
print(density_layer)
print("Weed Map:")
print(weed_map)
print("Start/End Points:", start_end_points)    



#NO ERROR

fig, axs = plt.subplots(1, 3, figsize=(12, 5))  # Adjust figsize to your needs
# Plotting the first heatmap with annotations
sns.heatmap(weed_map, ax=axs[0], cmap='crest', annot=True, fmt="d")
axs[0].set_title('Weed Density-Pre')

# round_robin_dfs(maze, start_end_points)
round_robin_dfs(maze, start_end_points, weed_map, 0.5, 30, 0)


# Plotting the first heatmap with annotations
sns.heatmap(weed_map, ax=axs[1], cmap='crest', annot=True, fmt="d")
axs[1].set_title('Weed Density-Post')

sns.heatmap(density_layer, ax=axs[2], cmap='crest', annot=True, fmt="d")
axs[2].set_title('Crop Density')

plt.tight_layout()  # Adjust layout so everything fits without overlapping
plt.show()





weed_map = weed_growth(density_layer)
fig2, axs2 = plt.subplots(1, 3, figsize=(12, 5))  # Adjust figsize to your needs

# Plotting the first heatmap with annotations
sns.heatmap(weed_map, ax=axs2[0], cmap='crest', annot=True, fmt="d")
axs2[0].set_title('Weed Density-Pre')

#ERROR MAX 1
round_robin_dfs(maze, start_end_points, weed_map, 0.5, 30, 5)

# Plotting the first heatmap with annotations
sns.heatmap(weed_map, ax=axs2[1], cmap='crest', annot=True, fmt="d")
axs2[1].set_title('Weed Density-Post')

sns.heatmap(density_layer, ax=axs2[2], cmap='crest', annot=True, fmt="d")
axs2[2].set_title('Crop Density')

plt.tight_layout()  # Adjust layout so everything fits without overlapping
plt.show()



print("Weed Map:")
print(weed_map)

mytuple = start_end_points

mytuple.pop()

