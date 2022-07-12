
import random
from random import randint
import numpy as np
from DQN import DQNAgent
from queue import Queue
import pygame
from collections import deque
import matplotlib.pyplot as plt
import itertools


class IterableQueue(Queue):

    def to_list(self):
        """
        Returns a copy of all items in the queue without removing them.
        """
        with self.mutex:
            return list(self.queue)



Training = True #specify the mode of operation True for training and False for inference
disp = False #display option, you can also use UP key to enable the display and Down key for disabling it. 

# defining variables for plotting learning curves graphs
avgLen = 100
numofpoints = 2000
numofTrials = 1





# Define  variables for GUI dispaly
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

grid_dim = 20 # grid size = grid_dim x grid_dim
resolution = 1024

width = 0.75*(resolution//grid_dim)
height = 0.75*(resolution//grid_dim)
margin = 0.25*(resolution//grid_dim)
size = (1024, 1024)
screen = pygame.display.set_mode(size)

pygame.display.set_caption("My Game")
clock = pygame.time.Clock()

# defining the simulation parameters
M = grid_dim #height
N = grid_dim #width
num_taxis = 10 # The number of taxis in the city
max_num_requests = 15 # the peak number of requests
setup = 4 # reward setup selection - details in the paper
batch_size = 24 # batch size for the DQN network
action_list = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)] # actions STAY,LEFT,RIGHT,DOWN,UP
action_size = len(action_list) #number of possible actions









# Temp variables for calculating the running average for (waiting time, traveled distance,reward)
datapoints = deque(maxlen=avgLen)
psgwait = deque(maxlen=avgLen)
IdelDist = deque(maxlen=avgLen)
ravggraph = np.zeros((numofTrials, numofpoints))
IdleDistgraph = np.zeros((numofTrials, numofpoints))
waitgraph = np.zeros((numofTrials, numofpoints))
psgwaitavg = deque(maxlen=numofpoints+1)

IdleDistavg = deque(maxlen=numofpoints+1)
runningAvg = deque(maxlen=numofpoints+1)

exploration = []


Total_waiting = 0 #Tracking the total waiting time for all passengers to calculate the average
Req_count = 0 # Tracking the current number of requests
num_req_ser = 0 #tracking the  number of served requests
num_req = 0 # tracking the total number generated requests 


class Request:
    def __init__(self, ORGx, ORGy, DESTx, DESTy, Time):
        self.ORGx = ORGx #Origin X coordinate
        self.ORGy = ORGy #Origin Y coordinate
        self.DESTx = DESTx #Desintation X coordinate
        self.DESTy = DESTy #Desintation Y coordinate
        self.Time = Time # Time stamp created when the request emerges


class Taxi:
    def __init__(self, id, state, posx, posy, req, xlim, ylim, idle_dist, TimetoDest):
        self.id = id
        self.state = state # ( 0 for IDLE (i.e vacant), 1 for Driving to pickup, 2 for Driving to drop )
        self.posx = posx # X coordinate
        self.posy = posy #Y coordinate
        self.request = req # the assigned request if any
        self.xlim = xlim # Taxi zone Limit in the X dim
        self.ylim = ylim # Taxi zone Limit in the Y dim
        self.idle_dist = idle_dist # Distance Traveled while IDLE
        self.TimetoDest = TimetoDest # Time required to reach the next destination
        self.prevStep = 0 # The previous action

    def RandomAction(self, taxi_grid, request_grid, DQN):
        reward = 0
        done = False
        if(self.state == 0): #state = 0 means taxi is vacant and looking for passenger
            (M, N) = taxi_grid.shape
            state = np.zeros((M, N, 3))
            state[:, :, 0] = request_grid
            state[self.posx, self.posy, 1] = 1
            state[:, :, 2] = taxi_grid
            state = np.expand_dims(state, axis=0)
            # state =state.reshape(1,2*M*N)
            #for i in range(M):
            #    for j in range(N):
            #        if(request_grid[i, j] > 0):
            #            reqx = i
            #            reqy = j

            # state=np.array([[self.posx,self.posy,reqx,reqy]])
            actlist = []
            prev_loc = (self.posx, self.posy)
            action = DQN.act(state)
            curr_act = action_list[action]
            prev_act = action_list[self.prevStep]
            resultant_act = tuple(map(sum, zip(curr_act, prev_act)))
            if(resultant_act == (0, 0)):
                actlist.append(action)
                action = DQN.actmod(state, actlist)

            (xstep, ystep) = action_list[action]
            taxi_grid[self.posx][self.posy] -= 1
            if(0 <= (self.posx+xstep) < self.xlim):
                self.posx += xstep
            else:
                if (setup == 4):
                    reward -= 100
            if(0 <= self.posy+ystep < self.ylim):
                self.posy += ystep
            else:
                if(setup == 4):
                    reward -= 100

            self.prevStep = action
            if prev_loc != (self.posx, self.posy):
                self.idle_dist += 1
            # reward += 1 * request_grid[self.posx][self.posy]
            if(setup == 1):
                reward += 10000*np.sign(request_grid[self.posx][self.posy])
            elif(setup == 2):
                reward += 10000*np.sign(request_grid[self.posx][self.posy])-100
            elif(setup >= 3):
                reward += 10000*np.sign(request_grid[self.posx][self.posy]) - \
                    100*np.sign(taxi_grid[self.posx][self.posy])-100

            if(action == 0 and setup == 5):
                reward += 1000

            taxi_grid[self.posx][self.posy] += 1
            next_state = np.zeros((M, N, 3))
            next_state[:, :, 0] = request_grid
            next_state[self.posx, self.posy, 1] = 1
            next_state[:, :, 2] = taxi_grid
            # next_state=next_state.reshape(1,2*M*N)
            next_state = np.expand_dims(next_state, axis=0)
            # next_state =np.array([[self.posx,self.posy,reqx,reqy]])
            done = False
            if(request_grid[self.posx][self.posy] > 0):
                done = True
            if not done:
                DQN.remember(state, action, reward, next_state, done)
            else:
                for i in range(10):
                    DQN.remember(state, action, reward, next_state, done)

            state = next_state
            # print("Action:{} dx={} dy={} reward={} exploration:{}".format(action, xstep,ystep,reward,DQN.epsilon))

        return (taxi_grid, request_grid, DQN, reward)

    def pickup(self, requestList, request_grid, taxi_grid, timestep):
        global Total_waiting
        global num_req_ser
        global Req_count
        if(self.state == 0): #state=0 IDLE state for the taxi
            if(not requestList[self.posx][self.posy].empty()):
                Req = requestList[self.posx][self.posy].get()
                self.request = Req
                Total_waiting += timestep-Req.Time
                psgwait.append(timestep-Req.Time)
                IdelDist.append(self.idle_dist)
                self.idle_dist = 0
                num_req_ser += 1
                request_grid[self.posx][self.posy] -= 1
                taxi_grid[self.posx][self.posy] -= 1
                self.state = 2
                self.TimetoDest = abs(Req.DESTx-Req.ORGx) + \
                    abs(Req.DESTy-Req.ORGy)
                Req_count -= 1
        return (requestList, request_grid, taxi_grid)

    def Drive(self, taxi_grid):
        if(self.state == 2):
            if(self.TimetoDest > 0):
                self.TimetoDest -= 1
            else:
                self.posx = self.request.DESTx
                self.posy = self.request.DESTy
                self.state = 0
                taxi_grid[self.posx][self.posy] += 1
        return taxi_grid


Taxi_tracking = [[IterableQueue() for _ in range(M)] for _ in range(N)]
# initialze taxis


def NearstTaxi(Taxi_grid, posx, posy, M, N):
    if(Taxi_grid[posx][posy] > 0):
        return (posx, posy)
    for iter in range(1, max(M, N)):
        y = posy-iter
        if(y >= 0):
            for x in range(max(0, posx-iter), min(posx+iter, M-1)):
                if(Taxi_grid[x][y] > 0):
                    return (x, y)
        x = posx-iter
        if(x >= 0):
            for y in range(max(0, posy-iter), min(posx+iter, N-1)):
                if(Taxi_grid[x][y] > 0):
                    return (x, y)
        y = posy+iter
        if(y < N):
            for x in range(max(0, posx-iter), min(posx+iter, M-1)):
                if(Taxi_grid[x][y] > 0):
                    return (x, y)
        x = x+iter
        if(x < M):
            for y in range(max(0, posy-iter), min(posx+iter, N-1)):
                if(Taxi_grid[x][y] > 0):
                    return (x, y)
    return (-1, -1)


class Enviroment:
    def __init__(self, numTaxis, Xdim, Ydim, QN):
        self.taxi_list = []
        self.taxi_grid = np.zeros((M, N))
        self.Taxi_tracking = [[IterableQueue() for _ in range(M)]
                              for _ in range(N)]
        for i in range(numTaxis):
            x = randint(0, Xdim-1)
            y = randint(0, Ydim-1)
            self.taxi_list.append(Taxi(id=i, state=0, posx=x, posy=y,
                                       req=None, xlim=M, ylim=N, idle_dist=0, TimetoDest=0))
            self.taxi_grid[x][y] += 1
            self.Taxi_tracking[x][y].put(i)
        self.request_list = [[IterableQueue() for _ in range(M)]
                             for _ in range(N)]
        self.request_grid = np.zeros((M, N))
        self.Xdim = Xdim
        self.Ydim = Ydim
        self.Req_count = 0
        self.Total_waiting = 0
        self.num_req_ser = 0
        self.num_req = 0
        self.QN = QN
        self.steps = 0
        # generating  a request
        global num_req, Req_count
        i = randint(0, M-1)
        j = randint(0, N-1)
        xdest = randint(0, M-1)
        ydest = randint(0, N-1)
        num_req += 1
        self.request_list[i][j].put(
            Request(ORGx=i, ORGy=j, DESTx=xdest, DESTy=ydest, Time=0))
        self.request_grid[i][j] += 1
        Req_count = 1

    def step(self):
        global Req_count, num_req
        comm_reward = 0
        requests_rate = 0.1
        for i in range(len(self.taxi_list)):
            T = self.taxi_list[i]
            (self.taxi_grid, self.request_grid, self.QN, reward) = T.RandomAction(
                self.taxi_grid, self.request_grid, self.QN)
            (self.request_list, self.request_grid, self.taxi_grid) = T.pickup(
                self.request_list, self.request_grid, self.taxi_grid, self.steps)
            self.taxi_grid = T.Drive(self.taxi_grid)
            if(Req_count < max_num_requests):
                if True:  # np.random.rand() <= requests_rate:  # if acting randomly take random action
                    # generating  a request
                    i = randint(0, M-1)
                    j = randint(0, N-1)
                    xdest = randint(0, M-1)
                    ydest = randint(0, N-1)
                    num_req += 1
                    self.request_list[i][j].put(
                        Request(ORGx=i, ORGy=j, DESTx=xdest, DESTy=ydest, Time=self.steps))
                    self.request_grid[i][j] += 1
                    Req_count += 1
                    comm_reward += reward
        comm_reward /= len(self.taxi_list)
        datapoints.append(comm_reward)
        if(self.steps % avgLen == 0):
            runningAvg.append(np.mean(datapoints))
            psgwaitavg.append(np.mean(psgwait))
            IdleDistavg.append(np.mean(IdelDist))
            exploration.append(Q.epsilon)
            print("Episode: "+str(self.steps/avgLen)+" avg waiting= " +
                  str(np.mean(psgwait))+" avg Reward="+str(np.mean(datapoints))+"Idle Distance:"+str(np.mean(IdelDist)) + " Exploration:"+str(Q.epsilon))

        self.steps += 1

        # print("Step:{} reward={}".format(self.steps, reward))



numofTrials = 1
for trial in range(numofTrials):
    state = np.zeros((M, N, 3))
    # state =state.reshape(1,2*M*N)
    # state=np.zeros(4)
    # state = np.expand_dims(state, axis=0)
    print(state.shape)
    # Q=QLearn( grid_dim, action_size, num_taxis, max_num_requests)
    Q = DQNAgent(action_size, state.shape)
    Q.epsilon = 0
    if(not Training):
        Q.load("ProbMODEL20x20T15R5")
        Q.epsilon = 0
    E = Enviroment(num_taxis, M, N, Q)
    print(E.taxi_grid)

    taxi_image = pygame.image.load(
        "./taxi.png").convert()
    taxi_image = pygame.transform.scale(taxi_image, (32, 32))
    passenger_image = pygame.image.load(
        "./passenger.png").convert()
    passenger_image = pygame.transform.scale(
        passenger_image, (32, 32))

    rate = 60
    done = False
    count = 0
    while not done:
        E.step()
        if(Training):
            if len(Q.memory) >= batch_size:
                Q.replay(batch_size)

            if(count % avgLen == 0):
                Q.updateEps()

            count += 1
            if (count > numofpoints*avgLen):
                done = True
            # --- Main event loop
            if(count % 1000 == 0):
                Q.save("MODEL"+str(M)+"x"+str(N)+"T"+str(num_taxis) +"R"+str(max_num_requests))

        # pygame.quit()
            # np.save(".\\S" + str(num_taxis)+"_"+str(max_num_requests)+str(grid_dim)+"x" +
            #        str(grid_dim) + "\\Setup"+str(setup)+"_avg_reward_Exp"+str(Q.epsilon_decay)+".npy", ravg)
            # np.save(".\\S" + str(num_taxis)+"_"+str(max_num_requests)+str(grid_dim)+"x" +
            #        str(grid_dim) + "\\Setup"+str(setup)+"_avg_waiting_Exp"+str(Q.epsilon_decay)+".npy", pwavg)
            # np.save(".\\S" + str(num_taxis)+"_"+str(max_num_requests)+str(grid_dim)+"x" +
            #        str(grid_dim) + "\\Setup"+str(setup)+"_avg_Idle_Exp"+str(Q.epsilon_decay)+".npy", Idleavg)

        # --- Game logic should go here
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                disp = True
            if event.type == pygame.KEYUP:
                disp = False
        if disp:
            for i in range(grid_dim):
                for j in range(grid_dim):
                    row = i*(width+margin)
                    column = j*(height+margin)
                    pygame.draw.rect(
                        screen, WHITE, [row, column, width, height])
                    if(E.taxi_grid[i, j] > 0):
                        screen.blit(taxi_image, [row, column])
                    if(E.request_grid[i, j] > 0):
                        screen.blit(
                            passenger_image, [row, column])

            # screen.fill(WHITE, rect=[i*width + margin, 0, width, height])

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            # if(Q.epsilon < 0.15):
            #    c = 8
            clock.tick(2)

    print(E.taxi_grid)
    print(E.request_grid)
    print(Total_waiting/num_req_ser)
    print(str(num_req_ser)+":served / "+str(num_req)+":created ")
    print("statexaction space")
    ravg = list(itertools.islice(runningAvg, 1, len(runningAvg)-1))
    pwavg = list(itertools.islice(psgwaitavg, 1, len(psgwaitavg)-1))
    Idleavg = list(itertools.islice(IdleDistavg, 1, len(IdleDistavg)-1))

    # Close the window and quit.
    # Q.save("ProbMODEL"+str(M)+"x"+str(N)+"T"+str(num_taxis) +
    #       "R"+str(max_num_requests))
    # pygame.quit()
    '''
    np.save(".\\S" + str(num_taxis)+"_"+str(max_num_requests)+str(grid_dim)+"x" +
            str(grid_dim) + "\\Setup"+str(setup)+"_avg_reward_Exp"+str(Q.epsilon_decay)+".npy", ravg)
    np.save(".\\S" + str(num_taxis)+"_"+str(max_num_requests)+str(grid_dim)+"x" +
            str(grid_dim) + "\\Setup"+str(setup)+"_avg_waiting_Exp"+str(Q.epsilon_decay)+".npy", pwavg)
    np.save(".\\S" + str(num_taxis)+"_"+str(max_num_requests)+str(grid_dim)+"x" +
            str(grid_dim) + "\\Setup"+str(setup)+"_avg_Idle_Exp"+str(Q.epsilon_decay)+".npy", Idleavg)
    '''
    t = range(len(ravg))
    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    axs[0].plot(t, ravg, lw=2, label='mean population 1', color='blue')
    # axs[0].fill_between(t, mu1+sigma1, mu1-sigma1,
    #                    facecolor='blue', alpha=0.5)
    # axs[0].set_title('average waiting time')mu3
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('average reward return')
    fig.suptitle('Exploration:'+str(Q.epsilon_decay) +
                 ' Discount Factor:'+str(Q.gamma), fontsize=16)

    axs[1].plot(t, pwavg, lw=2, label='mean population 2', color='yellow')
    # axs[1].fill_between(t, mu2+sigma2, mu2-sigma2,
    #                    facecolor='yellow', alpha=0.5)
    axs[1].set_xlabel('Episode')
    # axs[1].set_title('subplot 2')
    axs[1].set_ylabel('average waiting time')

    axs[2].plot(t, Idleavg, lw=2, label='mean population 3', color='red')
    # axs[2].fill_between(t, mu3+sigma3, mu3-sigma3,
    #                    facecolor='red', alpha=0.5)
    axs[2].set_xlabel('Episode')
    # axs[1].set_title('subplot 2')
    axs[2].set_ylabel('average Idle Distance')

    plt.show()
