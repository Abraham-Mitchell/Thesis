"""

Created on Tue Apr  2 10:31:23 2024

 

@author: alexandernelson

"""

 

import numpy as np

import matplotlib.pyplot as plt

import math

 

 

steps = []

actual_location = []

error_scale = 0.3

# number_of_steps = 500
number_of_steps = 50

actual_trials = []

# num_trials = 500
num_trials = 10

 

 

steps.append([0,0])

for itr,step in enumerate(range(number_of_steps)):

    #make go steps up then left until num steps reached 
    if step%2==0:

        steps.append([steps[itr][0]+1,steps[itr][1]])

    else:

        steps.append([steps[itr][0],steps[itr][1]+1])

 

steps = np.array(steps).T

 

for i in range(num_trials):

    actual_location = []

    actual_location.append((0,0))

    for itr,step in enumerate(range(number_of_steps)):

        error = abs(np.random.normal()*error_scale)

        angle = np.random.uniform()*360

        error_x = error*math.cos(angle)

        error_y = error*math.sin(angle)

        if step%2==0:

            actual_location.append([actual_location[itr][0]+1+error_x,actual_location[itr][1]+error_y])

        else:

            actual_location.append([actual_location[itr][0]+error_x,actual_location[itr][1]+1+error_y])

    actual_location = np.array(actual_location).T

    actual_trials.append(actual_location)

    plt.plot(actual_location[0],actual_location[1])

 

plt.plot(steps[0],steps[1])


plt.show()
