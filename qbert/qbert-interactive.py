import argparse
import sys
import pdb
import gymnasium as gym
import time
from gymnasium import wrappers, logger

"""
Action space
0 = stop
1 = stop
2 = up right
3 = down right
4 = up left
5 = down left
"""
from pynput import keyboard

action = 0


def on_press(key):
    global action
    try:
        if key.char == 'h':
            action = 2
        elif key.char == 'f':
            action = 4
        elif key.char == 'v':
            action = 5
        elif key.char == 'b':
            action = 3
        else:
            action = int(key.char)
        # print(last_key_pressed)
    except AttributeError:
        pass
    except ValueError:
        pass


def on_release(key):
    global action
    action = 0


# ...or, in a non-blocking fashion:
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()


class Agent(object):
    """The world's simplesst agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    # You should modify this function
    def act(self, observation, reward, done):
        global action
        return action


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
    done = False
    score = 0
    special_data = {}
    special_data['ale.lives'] = 3
    observation = env.reset()

    terminated = False
    while not terminated:
        action = agent.act(observation, reward, done)
        time.sleep(.05)

        # pdb.set_trace()
        observation, reward, terminated, truncated, info = env.step(action)

        if reward > 25:
            print(reward)
        score += reward

        env.render()

    # Close the env and write monitor result info to disk
    print("Your score: %d" % score)
    env.close()
