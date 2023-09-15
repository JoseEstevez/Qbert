import argparse
import sys
import pdb
import gymnasium as gym
import numpy as np
from gymnasium import wrappers, logger

import time


class Agent(object):
    """The world's simplest agent!"""
    beforeColor = np.array([45, 87, 176])
    qbertColor = np.array([181, 83, 40])
    coilyColor = np.array([146, 70, 192])
    defaultActions = []
    preferredActions = []

    def __init__(self, action_space):
        self.action_space = action_space

    # You should modify this function
    def act(self, observation, reward, done):
        vertInc = 30
        horizInc = 14
        self.actions = [5, 4, 3, 2]

        if reward == 3100:
            time.sleep(5)
            beforeColor = observation[36][77]
            self.beforeColor = beforeColor
        else:
            beforeColor = self.beforeColor

        # get the location of the qbert
        if np.where(observation == self.qbertColor)[0].size == 0:
            return self.action_space.sample()

        # 'top left' of qbert

        # 'bottom right' of qbert
        qbertcoord2 = [np.where(observation == self.qbertColor)[0][-1], np.where(observation == self.qbertColor)[1][-1]]

        # look down and left for unflipped block

        brightx = qbertcoord2[0] + vertInc
        brighty = qbertcoord2[1] - horizInc
        self.chooseMovement(brightx, brighty, 5)

        # look down and right for unflipped block

        brightx = qbertcoord2[0] + vertInc
        brighty = qbertcoord2[1] + horizInc
        self.chooseMovement(brightx, brighty, 3)
        # look up and left for unflipped block

        brightx = qbertcoord2[0] - vertInc
        brighty = qbertcoord2[1] - horizInc
        self.chooseMovement(brightx, brighty, 4)

        # look up and right for unflipped block
        brightx = qbertcoord2[0] - vertInc
        brighty = qbertcoord2[1] + horizInc
        self.chooseMovement(brightx, brighty, 2)

        if (self.preferredActions.__sizeof__() != 0) :
            return np.random.choice(self.preferredActions)
        else :
            return np.random.choice(self.actions)

    def chooseMovement(self, brightx, brighty, action):
        topleft, botright = makeLookBox(np.array([brightx, brighty]))
        try:
            if ((observation[topleft[0]:botright[0], topleft[1]:botright[1]] == self.coilyColor).any() or
                    (observation[topleft[0]:botright[0], topleft[1]:botright[1]] == np.array([0, 0, 0])).all()):
                self.actions.remove(action)
            elif (observation[topleft[0]:botright[0], topleft[1]:botright[1]] == self.beforeColor).any():
                self.preferredActions.append(action)
        except IndexError:
            pass


def makeLookBox(coord):
    topLeft = [coord[0] - 20, coord[1] - 10]
    botRight = [coord[0] + 10, coord[1] + 10]

    return topLeft, botRight

## YOU MAY NOT MODIFY ANYTHING BELOW THIS LINE OR USE
## ANOTHER MAIN PROGRAM
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', nargs='?', default='Qbert', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id, render_mode="human")

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = 'random-agent-results'

    env.unwrapped.seed(0)
    agent = Agent(env.action_space)

    episode_count = 100
    reward = 0
    terminated = False
    score = 0
    special_data = {}
    special_data['ale.lives'] = 3
    observation = env.reset()[0]

    while not terminated:
        action = agent.act(observation, reward, terminated)
        # pdb.set_trace()
        observation, reward, terminated, truncated, info = env.step(action)
        score += reward
        env.render()

    # Close the env and write monitor result info to disk
    print("Your score: %d" % score)
    env.close()
