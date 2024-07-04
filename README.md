# PacmanDQN

![](https://github.com/tomasamado/PacmanDQN/blob/main/images/pacman_gameplay.gif)

Attempting to play pacman (Atari 5200/800) with a reinforcement learning approach using [Deep Q-Networks](https://arxiv.org/abs/1312.5602) without any internal information from the states in the game. Implemented with Pytorch. Game played through Altirra emulator
 
## Goal and motivation

Most of the projects I see using RL to solve games involve using a Gym Environment that provides the rewards directly. 

Also most RL pacman projects use the **Atari 2600** version instead of the most popular **Atari 5200** version that resembles the classic arcade pacman interface. 

Here I'm attempting to use computer vision methods to extract the relevant information for rewards and other relevant information.

This same pipeline can technically be applied to future projects for games that don't have a dedicated Gym Environment. The only requirement is having a visual interface and a little bit of ingenuity to find end states from the image.

## Reward structure and environment

I use primarily the score displayed on the top left section of the screen as a reward. Specifically the diference between the score displayed on the previous frame and on the current one.

In order to capture the score I crop a window that contains the text and use pytesseract's [pytesseract's](https://pypi.org/project/pytesseract/) OCR to read it after some preprocessing to make the text clearer.

![Score section on the top left corner](https://github.com/tomasamado/PacmanDQN/blob/main/images/score_capture.png)

I also assign positive and negative based on losing and winning the game capturing visual elements that indicate so. 

![I keep track of the Pac-man life on the bottom left. When it dissapears, the game is over](https://github.com/tomasamado/PacmanDQN/blob/main/images/live_detection.png)

![When the border of the maze turns white, the game is won](https://github.com/tomasamado/PacmanDQN/blob/main/images/game_won.png)

Finally I give a constant negative reward every time Pac-man moves without eating a pellet. Trying to induce the model to always persue them instead of going through empty sections of the maze.

The reward structure is summarized as:


| Action  | Reward |
| ------------- | ------------- |
| Moving without eating pellet.  | -5  |
| Score increase (eating a pellet).  | 10 |
| Score increase (eating a ghost. Depends on how many were eating consecutively).  | 200, 400, 800 or 1600 |
| Game Over.  | -350 |
| Game Won.  | 100 |
Initial reward values inspired by ([Meo, 2018](https://reposit.haw-hamburg.de/handle/20.500.12738/8222))



