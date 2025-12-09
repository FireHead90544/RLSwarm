import math
import numpy as np
import pygame
from .config import *
from .utils import clamp, distance

class Agent:
    def __init__(self, x, y, color=AGENT_COLOR):
        self.pos = np.array([x, y], dtype=float)
        self.angle = np.random.uniform(0, 360)
        self.color = color
        self.speed = AGENT_SPEED
        self.carrying_food = False
        self.reward = 0.0

        # structure: [ray0..rayN-1, nf_dist, nf_dx, nf_dy, speed, sin, cos, na_dist, na_bearing, carrying, nest_dist, nest_dx, nest_dy]
        self.last_sensors = np.zeros(NUM_RAYS + 8, dtype=np.float32)

    def rotate(self, direction):
        self.angle = (self.angle + AGENT_TURN_RATE * direction) % 360

    def accelerate(self, factor):
        self.speed = max(0, min(AGENT_MAX_SPEED, self.speed + factor))

    def take_action(self, action: int):
        """
        Executes a discrete action:
        0 = No-op
        1 = Rotate left
        2 = Rotate right
        3 = Accelerate
        4 = Decelerate
        """
        if action == 1:
            self.rotate(-1)
        elif action == 2:
            self.rotate(1)
        elif action == 3:
            self.accelerate(0.1)
        elif action == 4:
            self.accelerate(-0.1)

    def update(self):
        dx = self.speed * math.cos(math.radians(self.angle))
        dy = self.speed * math.sin(math.radians(self.angle))
        self.pos += np.array([dx, dy])

        # wall collisions (reflect)
        bounced = False
        if self.pos[0] < AGENT_RADIUS:
            self.pos[0] = AGENT_RADIUS
            self.angle = 180 - self.angle
            self.reward += REWARD_WALL_COLLISION
            bounced = True
        elif self.pos[0] > SCREEN_WIDTH - AGENT_RADIUS:
            self.pos[0] = SCREEN_WIDTH - AGENT_RADIUS
            self.angle = 180 - self.angle
            self.reward += REWARD_WALL_COLLISION
            bounced = True

        if self.pos[1] < AGENT_RADIUS:
            self.pos[1] = AGENT_RADIUS
            self.angle = -self.angle
            self.reward += REWARD_WALL_COLLISION
            bounced = True
        elif self.pos[1] > SCREEN_HEIGHT - AGENT_RADIUS:
            self.pos[1] = SCREEN_HEIGHT - AGENT_RADIUS
            self.angle = -self.angle
            self.reward += REWARD_WALL_COLLISION
            bounced = True

        self.angle %= 360
        return bounced

    def draw(self, surface):
        SCALING = 1.5
        color = (255, 60, 60) if self.carrying_food else self.color

        tip = (
            self.pos[0] + math.cos(math.radians(self.angle)) * AGENT_RADIUS * SCALING,
            self.pos[1] + math.sin(math.radians(self.angle)) * AGENT_RADIUS * SCALING
        )

        left = (
            self.pos[0] + math.cos(math.radians(self.angle + 120)) * AGENT_RADIUS,
            self.pos[1] + math.sin(math.radians(self.angle + 120)) * AGENT_RADIUS
        )

        right = (
            self.pos[0] + math.cos(math.radians(self.angle - 120)) * AGENT_RADIUS,
            self.pos[1] + math.sin(math.radians(self.angle - 120)) * AGENT_RADIUS
        )

        pygame.draw.polygon(surface, color, [tip, left, right])

    # -------------------------
    # Sensors (raycasts + nearest-food)
    # -------------------------
    def compute_sensors(self, foods, agents):
        """
        Compute ray distances and contextual info for RL input.
        Returns vector:
        [rays..., nearest_food_dist_norm, nf_dx, nf_dy,
         speed_norm, sin(angle), cos(angle),
         nearest_agent_dist_norm, nearest_agent_bearing]
        """
        rays = []
        half_span = RAY_SPAN_DEG / 2.0
        for i in range(NUM_RAYS):
            frac = 0.5 if NUM_RAYS == 1 else i / (NUM_RAYS - 1)
            ray_ang = self.angle - half_span + frac * RAY_SPAN_DEG
            dist = self._ray_distance(ray_ang, RAY_MAX_DIST, agents)
            rays.append(dist / RAY_MAX_DIST)

        # Nearest food
        nearest_d = float('inf')
        nearest_vec = np.zeros(2)
        if foods:
            for f in foods:
                d = np.linalg.norm(f.pos - self.pos)
                if d < nearest_d:
                    nearest_d = d
                    nearest_vec = (f.pos - self.pos) / (d + 1e-8)
        nearest_d_norm = 1.0 - clamp(nearest_d / RAY_MAX_DIST, 0.0, 1.0)

        # Nest info
        nest_pos = np.array(NEST_POSITION)
        nest_d = np.linalg.norm(nest_pos - self.pos)
        nest_vec = (nest_pos - self.pos) / (nest_d + 1e-8)
        nest_d_norm = 1.0 - clamp(nest_d / SCREEN_WIDTH, 0.0, 1.0)

        # Nearest agent
        nearest_a_d = float('inf')
        nearest_a_bearing = 0.0
        for a in agents:
            if a is self:
                continue
            d = np.linalg.norm(a.pos - self.pos)
            if d < nearest_a_d:
                nearest_a_d = d
                vec = a.pos - self.pos
                bearing = math.degrees(math.atan2(vec[1], vec[0])) - self.angle
                nearest_a_bearing = math.sin(math.radians(bearing))
        nearest_a_d_norm = 1.0 - clamp(nearest_a_d / RAY_MAX_DIST, 0.0, 1.0)

        # Orientation features
        sin_o = math.sin(math.radians(self.angle))
        cos_o = math.cos(math.radians(self.angle))

        # Speed
        speed_norm = self.speed / AGENT_MAX_SPEED
        
        # Carrying food flag
        carrying_food_f = 1.0 if self.carrying_food else 0.0

        obs = np.array(
            rays + [
                nearest_d_norm, nearest_vec[0], nearest_vec[1],
                speed_norm,
                carrying_food_f,
                nest_d_norm, nest_vec[0], nest_vec[1],
            ],
            dtype=np.float32
        )
        self.last_sensors = obs
        return obs

    def _ray_distance(self, angle_deg, max_dist, agents):
        """
        Compute distance using geometric ray-casting.
        - Ray is a line segment from agent pos to max_dist
        - Check intersection with walls and other agents (circles)
        - Return the minimum distance found.
        """
        ang_rad = math.radians(angle_deg)
        p1 = self.pos
        p2 = self.pos + np.array([math.cos(ang_rad), math.sin(ang_rad)]) * max_dist

        min_dist = max_dist

        # Wall intersections
        # Top wall (y=0)
        if p2[1] < p1[1]:
            d = p1[1] / (p1[1] - p2[1]) * max_dist if (p1[1] - p2[1]) != 0 else float('inf')
            if 0 < d < min_dist: min_dist = d
        # Bottom wall (y=SCREEN_HEIGHT)
        if p2[1] > p1[1]:
            d = (SCREEN_HEIGHT - p1[1]) / (p2[1] - p1[1]) * max_dist if (p2[1] - p1[1]) != 0 else float('inf')
            if 0 < d < min_dist: min_dist = d
        # Left wall (x=0)
        if p2[0] < p1[0]:
            d = p1[0] / (p1[0] - p2[0]) * max_dist if (p1[0] - p2[0]) != 0 else float('inf')
            if 0 < d < min_dist: min_dist = d
        # Right wall (x=SCREEN_WIDTH)
        if p2[0] > p1[0]:
            d = (SCREEN_WIDTH - p1[0]) / (p2[0] - p1[0]) * max_dist if (p2[0] - p1[0]) != 0 else float('inf')
            if 0 < d < min_dist: min_dist = d

        # Agent intersections (line-circle intersection)
        for agent in agents:
            if agent is self:
                continue
            
            v = p2 - p1
            w = agent.pos - p1
            
            a = np.dot(v, v)
            b = 2 * np.dot(w, v)
            c = np.dot(w, w) - AGENT_RADIUS**2
            
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                sqrt_d = math.sqrt(discriminant)
                t1 = (-b - sqrt_d) / (2*a)
                t2 = (-b + sqrt_d) / (2*a)
                
                # Check if intersection point is on the ray segment
                if 0 <= t1 <= 1:
                    dist = t1 * max_dist
                    if dist < min_dist:
                        min_dist = dist
                elif 0 <= t2 <= 1 and t2 < t1: # check if closer point is valid
                    dist = t2 * max_dist
                    if dist < min_dist:
                        min_dist = dist
        
        return min_dist
