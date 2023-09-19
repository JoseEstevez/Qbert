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
    lastReward = 0.0
    lvlchange = 0

    def __init__(self, action_space):
        self.action_space = action_space

    # You should modify this function
    def act(self, observation, reward, done):
        vertInc = 30
        horizInc = 14
        self.defaultActions = [5, 4, 3, 2]
        self.preferredActions = []

        if reward == 100:
            # beforeColor = observation[92][77]
            # self.beforeColor = observation[92][77]
            # self.lastReward = reward
            self.lvlchange = 2
            return 0
        elif self.lvlchange > 0:
            self.beforeColor = observation[92][77]
            self.lvlchange -= 1

        # self.lastReward = reward
        # get the location of the qbert
        if np.where(observation == self.qbertColor)[0].size == 0:
            return self.action_space.sample()

        # 'top left' of qbert

        # 'bottom right' of qbert
        qbertcoord = [np.where(observation == self.qbertColor)[0][-1], np.where(observation == self.qbertColor)[1][-1]]

        # Check if Coily is at any 2 block distance from Qbert

        upx = qbertcoord[0] - 2 * vertInc
        upy = qbertcoord[1]

        downx = qbertcoord[0] + 2 * vertInc
        downy = qbertcoord[1]

        leftx = qbertcoord[0]
        lefty = qbertcoord[1] - 2 * horizInc

        rightx = qbertcoord[0]
        righty = qbertcoord[1] + 2 * horizInc

        if (self.checkCoilyNear(upx, upy) or self.checkCoilyNear(downx, downy) or
                self.checkCoilyNear(leftx, lefty) or self.checkCoilyNear(rightx, righty)):
            return 0

        # look down and left for unflipped block

        # nearx = qbertcoord[0] + vertInc
        # neary = qbertcoord[1] - horizInc
        # farx = qbertcoord[0] + 2 * vertInc
        # fary = qbertcoord[1] - 2 * horizInc
        near = np.array([qbertcoord[0] + vertInc, qbertcoord[1] - horizInc])
        far = np.array([qbertcoord[0] + 2 * vertInc, qbertcoord[1] - 2 * horizInc])
        self.chooseMovement(near, far, 5)

        # look down and right for unflipped block

        # nearx = qbertcoord[0] + vertInc
        # neary = qbertcoord[1] + horizInc
        # farx = qbertcoord[0] + 2 * vertInc
        # fary = qbertcoord[1] + 2 * horizInc
        near = np.array([qbertcoord[0] + vertInc, qbertcoord[1] + horizInc])
        far = np.array([qbertcoord[0] + 2 * vertInc, qbertcoord[1] + 2 * horizInc])
        self.chooseMovement(near, far, 3)
        # look up and left for unflipped block

        # nearx = qbertcoord[0] - vertInc
        # neary = qbertcoord[1] - horizInc
        # farx = qbertcoord[0] - 2 * vertInc
        # fary = qbertcoord[1] - 2 * horizInc
        near = np.array([qbertcoord[0] - vertInc, qbertcoord[1] - horizInc])
        far = np.array([qbertcoord[0] - 2 * vertInc, qbertcoord[1] - 2 * horizInc])
        self.chooseMovement(near, far, 4)

        # look up and right for unflipped block
        # nearx = qbertcoord[0] - vertInc
        # neary = qbertcoord[1] + horizInc
        # farx = qbertcoord[0] - 2 * vertInc
        # fary = qbertcoord[1] + 2 * horizInc
        near = np.array([qbertcoord[0] - vertInc, qbertcoord[1] + horizInc])
        far = np.array([qbertcoord[0] - 2 * vertInc, qbertcoord[1] + 2 * horizInc])
        self.chooseMovement(near, far, 2)

        if (len(self.preferredActions) != 0):
            return np.random.choice(self.preferredActions)
        elif (len(self.defaultActions) != 0):
            return np.random.choice(self.defaultActions)
        return 0

    def chooseMovement(self, near, far, action):
        tlnear, brnear = makeLookBox(near)
        tlfar, brfar = makeLookBox(far)

        # we need to check the color of qberts current condition, then look to the look box and see if there's some color
        # there that isn't in qberts current box, do it after the first if because it's more important to check that coily
        # isn't there first, somehow put it in the elif.
        # box = observation[tlnear[0]:brnear[0], tlnear[1]:brnear[1]]
        # box.reshape((brnear[0] - tlnear[0]) * (brnear[1] - tlnear[1]), 3)
        # colors = tuple

        try:
            if ((observation[tlnear[0]:brnear[0], tlnear[1]:brnear[1]] == self.coilyColor).any() or
                    self.isEmpty(observation, tlnear, brnear) or
                    (observation[tlfar[0]:brfar[0], tlfar[1]:brfar[1]] == self.coilyColor).any()):
                self.defaultActions.remove(action)
            elif (observation[tlnear[0]:brnear[0], tlnear[1]:brnear[1]] == self.beforeColor).any():
                self.preferredActions.append(action)
        except IndexError:
            pass

    def isEmpty(self, observation, topleft, botright):
        grayscale = np.mean(observation, axis=2)

        colors = np.unique(grayscale[topleft[0]:botright[0], topleft[1]:botright[1]])

        if (colors.size > 2):
            return False

        return True

    def checkCoilyNear(self, x, y):
        topleft, botright = makeLookBox(np.array([x, y]))
        if x < 0 or y < 0:
            return False
        if (observation[topleft[0]:botright[0], topleft[1]:botright[1]] == self.coilyColor).any():
            return True
        return False


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
