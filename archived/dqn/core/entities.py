import pygame
import random
import numpy as np
from .config import *

class Food:
    def __init__(self, x=None, y=None):
        if x is None:
            x = random.randint(50, SCREEN_WIDTH - 50)
        if y is None:
            y = random.randint(50, SCREEN_HEIGHT - 50)
        self.pos = np.array([x, y], dtype=float)
        self.color = FOOD_COLOR

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, self.pos.astype(int), FOOD_RADIUS)
