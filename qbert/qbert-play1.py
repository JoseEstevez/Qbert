"""qbert-play1.py

AI agent fof Qbert
"""

__author__ = "Brett Lubberts, Jose Estevez"

import argparse
import sys
import pdb
import gymnasium as gym
import numpy as np
from gymnasium import wrappers, logger

import time


class Agent(object):
    """The world's simplest agent!"""
    initBeforeColor = False
    beforeColor = 0
    qbertColor = np.dot(np.array([181, 83, 40]), [0.299, 0.587, 0.114])
    coilyColor = np.dot(np.array([146, 70, 192]), [0.299, 0.587, 0.114])
    defaultActions = []
    preferredActions = []
    lvlchange = 0
    grayscale = np.zeros((210, 160))

    def __init__(self, action_space):
        """Initialize the agent"""
        self.action_space = action_space

    def act(self, observation, reward, done):
        """Choose an action for the agent

        Keyword arguments:
        observation -- the current observation
        reward -- the reward from the last action
        done -- whether the episode is over
        """
        # make the observation grayscale for easier manipulation
        self.grayscale = np.dot(observation[:, :, :3],
                                [0.299, 0.587, 0.114])

        # initialize the color for the first time
        if not self.initBeforeColor:
            self.beforeColor = self.grayscale[92][77]
            self.initBeforeColor = True

        vertInc = 30
        horizInc = 14
        self.defaultActions = [5, 4, 3, 2]
        self.preferredActions = []

        if reward == 100:
            self.lvlchange += 1
            return 0
        elif self.lvlchange > 1:
            self.beforeColor = self.grayscale[92][77]
            self.lvlchange -= 1
            return 0

        # if Qbert is not on screen, do nothing
        if np.where(self.grayscale == self.qbertColor)[0].size == 0:
            return 0

        # 'bottom right' of qbert
        qbertcoord = [np.where(self.grayscale == self.qbertColor)[0][-1],
                      np.where(self.grayscale == self.qbertColor)[1][-1]]

        # Check if Coily is at any horizontal or vertical 2 block distance
        # from Qbert

        upx = qbertcoord[0] - 2 * vertInc
        upy = qbertcoord[1]

        downx = qbertcoord[0] + 2 * vertInc
        downy = qbertcoord[1]

        leftx = qbertcoord[0]
        lefty = qbertcoord[1] - 2 * horizInc

        rightx = qbertcoord[0]
        righty = qbertcoord[1] + 2 * horizInc

        if (self.checkCoilyNear(upx, upy) or self.checkCoilyNear(downx, downy)
                or self.checkCoilyNear(leftx, lefty) or self.checkCoilyNear(
                    rightx, righty)):
            return 0

        # look down and left for unflipped block

        near = np.array([qbertcoord[0] + vertInc, qbertcoord[1] - horizInc])
        far = np.array(
            [qbertcoord[0] + 2 * vertInc, qbertcoord[1] - 2 * horizInc])
        self.chooseMovement(near, far, 5)

        # look down and right for unflipped block

        near = np.array([qbertcoord[0] + vertInc, qbertcoord[1] + horizInc])
        far = np.array(
            [qbertcoord[0] + 2 * vertInc, qbertcoord[1] + 2 * horizInc])
        self.chooseMovement(near, far, 3)

        # look up and left for unflipped block

        near = np.array([qbertcoord[0] - vertInc, qbertcoord[1] - horizInc])
        far = np.array(
            [qbertcoord[0] - 2 * vertInc, qbertcoord[1] - 2 * horizInc])
        self.chooseMovement(near, far, 4)

        # look up and right for unflipped block

        near = np.array([qbertcoord[0] - vertInc, qbertcoord[1] + horizInc])
        far = np.array(
            [qbertcoord[0] - 2 * vertInc, qbertcoord[1] + 2 * horizInc])
        self.chooseMovement(near, far, 2)

        # do preferred action if there is one, otherwise do safe default action

        if (len(self.preferredActions) != 0):
            return np.random.choice(self.preferredActions)
        elif (len(self.defaultActions) != 0):
            return np.random.choice(self.defaultActions)
        return 0

    def chooseMovement(self, near, far, action):
        """Choose a movement based on the near and far blocks

        Keyword arguments:
        near -- the block that is one block away from Qbert
        far -- the block that is two blocks away from Qbert
        action -- the action that corresponds to the movement
        """
        tlnear, brnear = makeLookBox(near)
        tlfar, brfar = makeLookBox(far)

        try:

            # if the action is not safe, remove it from the list of default
            # actions, otherwise see if it is preferred
            if ((self.grayscale[tlnear[0]:brnear[0],
                 tlnear[1]:brnear[1]] == self.coilyColor).any() or
                    self.isEmpty(observation, tlnear, brnear) or
                    (self.grayscale[tlfar[0]:brfar[0],
                     tlfar[1]:brfar[1]] == self.coilyColor).any()):
                self.defaultActions.remove(action)
            elif (self.grayscale[tlnear[0]:brnear[0],
                  tlnear[1]:brnear[1]] == self.beforeColor).any():
                self.preferredActions.append(action)

            # stop traps
            # stop trap in bottom right
            if action == 3 and brnear[1] > 140 and brnear[0] < 210:
                # check up and left
                tmptl = [tlnear[0] - 60, tlnear[1] - 28]
                tmpbr = [brnear[0] - 60, brnear[1] - 28]
                # check down and left
                tmptl2 = [tlnear[0], tlnear[1] - 14]
                tmpbr2 = [brnear[0], brnear[1] - 14]
                if ((self.grayscale[tmptl[0]:tmpbr[0],
                     tmptl[1]:tmpbr[1]] == self.coilyColor).any() or
                        (self.grayscale[tmptl2[0]:tmpbr2[0],
                         tmptl2[1]:tmpbr2[1]] == self.coilyColor).any()):
                    if action in self.defaultActions:
                        self.defaultActions.remove(action)
            # stop trap in bottom left
            elif action == 5 and tlnear[1] < 10 and brnear[0] < 210:
                # check up and right
                tmptl = [tlnear[0] - 60, tlnear[1] + 28]
                tmpbr = [brnear[0] - 60, brnear[1] + 28]
                # check down and right
                tmptl2 = [tlnear[0], tlnear[1] + 14]
                tmpbr2 = [brnear[0], brnear[1] + 14]
                if ((self.grayscale[tmptl[0]:tmpbr[0],
                     tmptl[1]:tmpbr[1]] == self.coilyColor).any() or
                        (self.grayscale[tmptl2[0]:tmpbr2[0],
                         tmptl2[1]:tmpbr2[1]] == self.coilyColor).any()):
                    if action in self.defaultActions:
                        self.defaultActions.remove(action)

        except IndexError:
            pass

    def isEmpty(self, observation, topleft, botright):
        """Check if a block is empty

        Keyword arguments:
        observation -- the current observation
        topleft -- the top left corner of the block
        botright -- the bottom right corner of the block
        """

        grayscale = np.mean(observation, axis=2)
        colors = np.unique(
            grayscale[topleft[0]:botright[0], topleft[1]:botright[1]])

        if (colors.size > 2):
            return False

        return True

    def checkCoilyNear(self, x, y):
        """Check if Coily is at a certain block

        Keyword arguments:
        x -- the x coordinate of the block
        y -- the y coordinate of the block
        """
        topleft, botright = makeLookBox(np.array([x, y]))
        if x < 0 or y < 0:
            return False
        if (self.grayscale[topleft[0]:botright[0],
            topleft[1]:botright[1]] == self.coilyColor).any():
            return True
        return False


def makeLookBox(coord):
    """Make a box around a block to look at

    Keyword arguments:
    coord -- the coordinate of the block
    """
    topLeft = [coord[0] - 20, coord[1] - 10]
    botRight = [coord[0] + 10, coord[1] + 10]

    return topLeft, botRight


# YOU MAY NOT MODIFY ANYTHING BELOW THIS LINE OR USE
# ANOTHER MAIN PROGRAM
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', nargs='?', default='Qbert',
                        help='Select the environment to run')
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
