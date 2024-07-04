import subprocess
import time
import keyboard
from PIL import ImageGrab, Image, ImageEnhance
import cv2
import pytesseract
import numpy as np
import pygetwindow as gw
import re
import collections


class PacmanEnv:
    """
    Pacman Environment for training the agent using the Altirra emulator.
    
    Attributes:
        altirra_path (str): Path to the Altirra emulator executable.
        rom_path (str): Path to the Pacman ROM file.
        window_title (str): Title of the Altirra emulator window.
        action_space (list): List of possible actions.
        num_actions (int): Number of possible actions.
        previous_score (int): Score from the previous step.
        frame_stack (collections.deque): Stack of the last four frames.
    """
    def __init__(self, altirra_path, rom_path, window_title="Altirra"):
        self.altirra_path = altirra_path 
        self.rom_path = rom_path 
        self.window_title = window_title 
        self.action_space = ['up', 'down', 'left', 'right', 'no_action'] 
        self.num_actions = len(self.action_space) 
        self.previous_score = 0
        self.start_altirra()
        self.frame_stack = collections.deque(maxlen=4)

    def start_altirra(self):
        """
        Start the Altirra emulator with the specified ROM.
        """
        try:
            self.altirra_process = subprocess.Popen([self.altirra_path, "/cart", "/cartmapper", "8K", self.rom_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            time.sleep(2)
            self.start_game()
        except FileNotFoundError as e:
            print(f"Error launching Altirra: {e}")
            exit(1)

    def start_game(self):
        """
        Start the game by sending the Ctrl key.
        """
        print("Sending Ctrl key...")
        time.sleep(2) # Delay for game to open
        self._send_key("ctrl")
        print("Ctrl key sent.")

    def _get_state(self):
        """
        Capture and preprocess the current game state.
        
        Returns:
            np.ndarray: Preprocessed image of the current state.
            int: Current score.
            np.ndarray: Original screenshot.
        """
        screenshot = self._capture_screen() #Obtain whole image
        preprocessed_image = self._preprocess_image(screenshot) #Obtain image for DQN
        score = self._get_score(screenshot)
        return preprocessed_image, score, screenshot

    def step(self, action):
        """
        Take a step in the environment using the given action.
        
        Args:
            action (int): Index of the action to be taken.
        
        Returns:
            np.ndarray: Stacked states after taking the action.
            int: Reward obtained from the action.
            bool: Whether the game is over or the level is won.
        """
        if self.action_space[action] != 'no_action':
            self._send_key(self.action_space[action])

        next_state, score, original_screenshot = self._get_state()
        #Game ending flags
        game_over = self._is_game_over(original_screenshot)
        level_won = self._is_level_won(original_screenshot)

        done = game_over or level_won

        if game_over:
            score = score - 350  # Negative reward for losing
        
        if level_won:
            score = score + 100  # Reward for winning

        reward = score - self.previous_score #Reward for increasing in score

        if score == self.previous_score: # Negative reward for not collecting pellets
            reward -= 5
 
        self.previous_score = score
        self.frame_stack.append(next_state)
        
        return np.array(self.frame_stack), reward, done

    def _send_key(self, key):
        """
        Send a key press event.
        
        Args:
            key (str): Key to be pressed.
        """
        keyboard.press(key)
        time.sleep(0.005) #Short delay for key to register
        keyboard.release(key)

    def _capture_screen(self):
        """
        Capture a screenshot of the game window.
        
        Returns:
            np.ndarray: Captured screenshot as a numpy array.
        
        Raises:
            Exception: If the game window is not found after maximum retries.
        """
        max_retries = 5
        for attempt in range(max_retries):
            try:
                game_window = gw.getWindowsWithTitle(self.window_title)[0]
                #Remove uncecessary space in window capture
                left = game_window.left + 100 
                top = game_window.top + 85
                right = game_window.right - 100
                bottom = game_window.bottom - 65
                screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))
                return np.array(screenshot)
            except IndexError:
                if attempt < max_retries - 1:
                    print(f"Window '{self.window_title}' not found. Retrying {attempt + 1}/{max_retries}...")
                    time.sleep(2) 
                else:
                    raise Exception(f"Window '{self.window_title}' not found after {max_retries} attempts.")

    def _preprocess_image(self, image):
        """
        Preprocess the captured image for the model.
        
        Args:
            image (np.ndarray): Captured screenshot.
        
        Returns:
            np.ndarray: Preprocessed image.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert image to grayscale
        resized_image = cv2.resize(gray_image, (100, 100)) # Resize image to (84, 84)
        enhanced_image = self._enhance_contrast(resized_image) 
        normalized_image = enhanced_image / 255.0 
        return normalized_image


    def _enhance_contrast(self, image):
        """
        Enhance the contrast of the image.
        
        Args:
            image (np.ndarray): Image to be enhanced.
        
        Returns:
            np.ndarray: Contrast-enhanced image.
        """
        if len(image.shape) == 3 and image.shape[0] == 1:
            image = image.squeeze(0)
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced_image = enhancer.enhance(2)
        return np.array(enhanced_image)

    def _get_score(self, image):
        """
        Extract the score from the captured image using OCR.
        
        Args:
            image (np.ndarray): Captured screenshot.
        
        Returns:
            int: Extracted score.
        """
        #Capture top left corner where score is presnted and process it for better OCR capture
        score_region = image[30:68, 41:163]
        score_region = self._enhance_contrast(score_region)
        _, binary_image = cv2.threshold(score_region, 150, 255, cv2.THRESH_BINARY)
        binary_image = cv2.resize(binary_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        score_text = pytesseract.image_to_string(binary_image, config='--oem 3')
        
        #Clean up text captured
        score_text = re.sub(r'1LUP', '', score_text) #Remove 1LUP 
        score_text = re.sub(r'\D', '', score_text) #Only allow digits

        #Handle case where 0 unit is identified as 6
        if score_text.endswith('6'): 
            score_text = score_text[:-1] + '0'
        score = int(score_text) if score_text.isdigit() else 0

        #Prevent really big jumps in score
        if score < self.previous_score or abs(score - self.previous_score) > 500:
            score = self.previous_score
        score = max(score, self.previous_score)

        return score

    def _is_game_over(self, image):
        """
        Check if the game is over.
        
        Args:
            image (np.ndarray): Captured screenshot.
        
        Returns:
            bool: True if the game is over, False otherwise.
        """
        lives_region = image[580, 113] #Region where pacman life is placed  
        if np.all(lives_region == [0, 0, 0]): # Check if the life icon dissapears
            return True
        else:
            return False

    def _is_level_won(self, image):
        """
        Check if the level is won.
        
        Args:
            image (np.ndarray): Captured screenshot.
        
        Returns:
            bool: True if the level is won, False otherwise.
        """
        border_pixel = image[149,16]  #Border of the map pixel      
        if np.all(border_pixel == [204,204,204]): #If border changes color, game is won
            return True
        else:
            return False

    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            np.ndarray: Initial stacked state.
            int: Initial score.
            np.ndarray: Initial screenshot.
        """
        self.frame_stack.clear()
        self.previous_score = 0
        state, score, screenshot = self._get_state()
        for _ in range(4):
            self.frame_stack.append(state)
        return np.array(self.frame_stack), score, screenshot
    
    def reset_game(self):
        """
        Reset the game by restarting the emulator and getting the initial state.
        
        Returns:
            np.ndarray: Initial preprocessed image.
            int: Initial score.
            np.ndarray: Initial screenshot.
        """
        self.close()
        self.start_altirra()
        self.reset()
        return self._get_state()

    def close(self):
        try:
            self.altirra_process.terminate()
            print("Altirra terminated successfully.")
        except Exception as e:
            print(f"Error terminating Altirra: {e}")

