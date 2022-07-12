# MADRL-for-Taxi-Fleet-Management

This is the code repository for [A Scalable Multi-agent Reinforcement Learning Approach for the Dynamic Taxi Dispatch Problem](https://link.springer.com/chapter/10.1007/978-981-19-2259-6_101)

## Code Structure 
1. Simulation.py : containes that code that implements the Taxi simulation environment in the header of the file you can change the simulation options 

	-Training = True #specify the mode of operation True for training and False for inference
	-disp = False #display option, you can also use UP key to enable the display and Down key for disabling it. 
	-M = grid_dim #height
	-N = grid_dim #width
	-num_taxis = 10 # The number of taxis in the city
	-max_num_requests = 15 # the peak number of requests
	-setup = 4 # reward setup selection - details in the paper
	-batch_size = 24 # batch size for the DQN network

2. DQN.py : contains the code for Deep Q learning MODEL. 


## Requirements 
The code the written in python and  following Libraries are required. 
	-tensorflow
	-matplotlib
	-pygame
	-gym


## How to run 
run the following command in your terminal 
```
python Simulation.py 
```
A blank window will popup, to enable display press and hold UP key.

### Episode No. 0
![Episode #0](https://myoctocat.com/assets/images/base-octocat.svg)



### Episode No. 100
![Episode #100](https://myoctocat.com/assets/images/base-octocat.svg)


## Plots  
when you exits the pygame display window a plot for the learning curves will pop up. 


### Learning Curves 
![](https://myoctocat.com/assets/images/base-octocat.svg)


## Cite Our Paper
Alhusin, M., Pasquier, M., Barlas, G. (2022). A Scalable Multi-agent Reinforcement Learning Approach for the Dynamic Taxi Dispatch Problem. In: Zhang, Z. (eds) 2021 6th International Conference on Intelligent Transportation Engineering (ICITE 2021). ICITE 2021. Lecture Notes in Electrical Engineering, vol 901. Springer, Singapore. https://doi.org/10.1007/978-981-19-2259-6_101

