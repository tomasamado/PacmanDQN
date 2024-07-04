# PacmanDQN

![](https://github.com/tomasamado/PacmanDQN/blob/main/pacman_gameplay.gif)

Attempting to play pacman (Atari 5200/800) with a reinforcement learning approach using a Deep Q-Network without any internal information from the states in the game. Implemented with Pytorch. Game played through Altirra emulator
 
## Goal and motivation

Most of the projects I see using RL to solve games involve using a Gym Environment that provides the rewards directly. Also most RL pacman projects use the Atari 2600 version instead of the most popular 5200/arcade one. 
Here I'm attempting to use computer vision methods to extract the relevant information for rewards and other relevant information.
This same pipeline can technically be applied to future projects for games that don't have a dedicated Gym Environment. The only requirement is having a visual interface and a little bit of ingenuity to find end states from the image.

