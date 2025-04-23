### main.py
# This file contains the main function that initializes the environment and the DQAgent, and runs the training loop.

from map import Map
from DQAgent import DQAgent
import random
import matplotlib.pyplot as plt

def moving_avg(seq, window):
    return [sum(seq[max(0, i-window+1): i+1]) / min(window, i+1)
            for i in range(len(seq))]


MAXSIZE = 100
MAXBANDS = 3

def reset():
    m = Map(100, 100, 10, MAXSIZE, MAXBANDS)
    m.add_wall(0, 0, 100, 10)
    m.add_wall(0, 0, 10, 100)
    m.add_wall(90, 0, 100, 100)
    m.add_wall(0, 90, 100, 100)
    m.add_agent(random.randint(20, 70), random.randint(20, 70))
    return m


agent = DQAgent((MAXSIZE, MAXSIZE, MAXBANDS), 16)
#agent.loadModel("models/Model-latest.weights.h5")



roundNum = 0
EveryReward = []
PercentVisited = []
m = reset()
mapsize = m.getMovableCount()
print("Map size:", mapsize)
GOAL_REWARD      = +1
STEP_PENALTY     = -1.0/mapsize     # every time step
NEW_CELL_REWARD  = +5.0/mapsize
while roundNum < 10000:
    roundNum += 1
    m = reset()
    allRewards = 0
    print("Round:", roundNum, "Epsilon:", agent.epsilon)
    if roundNum % 100 == 0:
        oldEps = agent.epsilon
    for i in range(300):
        startLen = m.grid[:,:,2].sum()
        m.move_direction(agent.act(m.getGrid3D()))
        afterLen = m.grid[:,:,2].sum()

        if roundNum % 10 == 0:
            m.displayBase()
        if afterLen >= mapsize:
            print("-------------------------------------Visited all cells")
            reward = GOAL_REWARD
        elif afterLen > startLen:
            reward = NEW_CELL_REWARD * (afterLen - startLen)
        else:
            reward = STEP_PENALTY
        agent.remember(m.getGrid3D(), reward, reward == GOAL_REWARD)
        allRewards += reward
        if reward == GOAL_REWARD:
            break
    print("Total rewards:", allRewards)
    if roundNum % 100 == 0:
        agent.epsilon = oldEps
    
    EveryReward.append(allRewards)
    visited_pct = (m.grid[:, :, 2].sum() / mapsize) * 100.0   # % of cells cleaned this episode
    PercentVisited.append(visited_pct)

    # --- plotting ---
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

    # rewards (existing plot)
    ax1.plot(moving_avg(EveryReward, 100), label="100-pt MA")
    ax1.plot(moving_avg(EveryReward, 10) , label="10-pt MA")
    ax1.set_ylabel("Total reward")
    ax1.legend()
    ax1.grid(True, linestyle=":")

    # NEW: % of spaces visited
    ax2.plot(moving_avg(PercentVisited, 100), label="% visited (100-pt MA)")
    ax2.plot(moving_avg(PercentVisited, 10), label="% visited (10-pt MA)")
    ax2.set_ylabel("Visited %")
    ax2.set_xlabel("Episode")
    ax2.legend()
    ax2.grid(True, linestyle=":")

    fig.tight_layout()
    fig.savefig("rewards.png")

m.displayMove()
m.close()