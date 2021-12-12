"""This module defines the utility functions."""

import numpy as np
import os
import math
import moviepy.editor as mpy
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def softmax(X):
    """Softmax function"""

    exps = np.exp(X)
    return exps / np.sum(exps, axis=1).reshape(-1, 1)


def normalized_gaussian(x, mu, sig):
    """Function defining normalized gaussian
    Args:
        x: input array for the gaussian
        mu: mean of the gaussian
        sig: standard deviation of the gaussian

    Returns: normalized gaussian array
    """

    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * (
                1 / np.sqrt(2 * np.pi * np.power(sig, 2.)))


def set_image_bandit(type_rew, valuess, probs, selection, trial):
    """Function to create an image depicting model behavior"""

    input_img = os.path.join('helper_files/bandit.png')
    levers_image = Image.open(input_img)
    draw = ImageDraw.Draw(levers_image)
    font = ImageFont.truetype("./FreeSans.ttf", 24)

    draw.text((10, 10), str(float("{0:.2f}".format(probs[0]))), (0, 0, 0),
              font=font)
    draw.text((70, 10), str(float("{0:.2f}".format(probs[1]))), (0, 0, 0),
              font=font)
    draw.text((140, 10), 'fix', (0, 0, 0), font=font)

    draw.text((10, 370), ' Trial: ' + str(trial), (0, 0, 0), font=font)

    levers_image = np.array(levers_image)

    if type_rew == -1:
        levers_image[115:115 + math.floor(valuess[1] * 2.5), 20:55, :] = [255.0,
                                                                          0.0,
                                                                          0]
        levers_image[115:115 + math.floor(valuess[2] * 2.5), 80:115, :] = [
            255.0, 0.0, 0]
        levers_image[115:115 + math.floor(valuess[0] * 2.5), 140:175, :] = [
            255.0, 0.0, 0]

    else:
        levers_image[115:115 + math.floor(valuess[1] * 2.5), 20:55, :] = [0,
                                                                          255.0,
                                                                          0]
        levers_image[115:115 + math.floor(valuess[2] * 2.5), 80:115, :] = [0,
                                                                           255.0,
                                                                           0]
        levers_image[115:115 + math.floor(valuess[0] * 2.5), 140:175, :] = [0,
                                                                            255.0,
                                                                            0]

    if selection == 0:
        levers_image[101:107,
        10 + ((selection + 2) * 60):10 + ((selection + 2) * 60) + 50, :] = [
            80.0, 80.0, 225.0]
    else:
        levers_image[101:107,
        10 + ((selection - 1) * 60):10 + ((selection - 1) * 60) + 50, :] = [
            80.0, 80.0, 225.0]

    return levers_image


def create_gif(episode_frames):
    """Function to create GIF images"""

    img_list = []
    ep_reward = [0, 0, 0]
    for i in range(len(episode_frames[0])):
        type_rew = -1 if episode_frames[0][i] == -1 else 1
        img_list.append(
            set_image_bandit(type_rew, ep_reward, episode_frames[1][i],
                                 episode_frames[2][i], episode_frames[3][i]))
        ep_reward[episode_frames[2][i]] += abs(episode_frames[0][i])

    return img_list


def make_gif(images, fname, duration=2, true_image=False):
    """Function to create GIF out of the GIF images"""

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.write_gif(fname, fps=len(images) / duration, verbose=False)
