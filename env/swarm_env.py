import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random
from typing import Optional, List, Tuple

from config import *

class SwarmEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This environment simulates a swarm of agents foraging for food.
    
    NOTE: This environment is designed to be wrapped in a MultiAgent wrapper
    for Stable Baselines3, or used as a single-agent view of the world.
    
    To make it compatible with standard SB3 (which expects single-agent interface),
    we will implement a 'Shared State' approach where the environment steps
    the entire simulation, but the 'step' method returns the observation/reward
    for *one* agent at a time, or we use a VectorEnv approach.
    
    DECISION: We will implement this as a standard Gym Env that represents
    the *entire* swarm state if possible, OR use the VectorEnv approach where
    this class represents ONE agent but shares static state.
    
    BETTER APPROACH for SB3 PPO with Shared Policy:
    This class will behave like a VectorEnv internally.
    However, implementing a full VectorEnv from scratch is complex.
    
    SIMPLER APPROACH:
    This class implements the logic for N agents.
    But `step()` and `reset()` will return a list of observations (for all agents).
    We will then use a custom Wrapper to make it look like a VecEnv to SB3.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, num_agents=5, render_mode=None, max_episode_steps=2500, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):
        super(SwarmEnv, self).__init__()
        
        self.num_agents = num_agents
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.width = width
        self.height = height
        self.screen = None
        self.clock = None
        self.debug = False # Default debug off
        self.selected_agent = 0 # For debug UI
        
        # Action Space: Discrete
        # 0: No-op, 1: Left, 2: Right, 3: Accel, 4: Decel
        self.action_space = spaces.Discrete(5)
        
        # Observation Space
        self.obs_dim = NUM_RAYS + 10
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(self.obs_dim,), dtype=np.float32
        )

        # Simulation State
        self.agents = []
        self.foods = []
        self.frame_count = 0
        self.episode_step = 0
        
        # Initialize state
        self.reset()
        
        # Fonts
        self.font = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        self.agents = []
        for _ in range(self.num_agents):
            # Random pos
            x = random.randint(AGENT_RADIUS * 2, self.width - AGENT_RADIUS * 2)
            y = random.randint(AGENT_RADIUS * 2, self.height - AGENT_RADIUS * 2)
            self.agents.append(Agent(x, y))
            
        self.foods = [Food() for _ in range(min(MAX_FOOD, 12))]
        self.frame_count = 0
        self.episode_step = 0
        
        observations = self._get_observations()
        # Return list of observations (one per agent)
        # Note: Standard Gym expects single obs. We will handle this in wrapper.
        return observations, {}

    def step(self, actions):
        """
        Step the simulation.
        actions: List of ints, one per agent.
        """
        # 1. Apply actions
        for i, agent in enumerate(self.agents):
            action = actions[i] if i < len(actions) else 0
            agent.take_action(action)
            
        # 2. Physics & Logic
        # Apply step penalty
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        rewards += REWARD_STEP
        
        # Update movement
        for agent in self.agents:
            bounced = agent.update()
            if bounced:
                agent.reward += REWARD_WALL_COLLISION # Add to internal agent reward accumulator
        
        # Collisions
        self._handle_agent_collisions()
        
        # Food & Nest
        self._handle_food_interactions()
        
        # Collect rewards from agents and reset their internal accumulators
        for i, agent in enumerate(self.agents):
            rewards[i] += agent.reward
            agent.cumulative_reward += rewards[i] # Update cumulative
            agent.reward = 0.0 # Reset for next step
            
        # 3. Observations
        observations = self._get_observations()
        
        # 4. Done & Info
        # Increment episode step counter
        self.episode_step += 1
        
        # Episode truncates after max_episode_steps
        terminated = False
        truncated = self.episode_step >= self.max_episode_steps
        infos = {}
        
        if self.render_mode == "human":
            self.render()
            
        return observations, rewards, terminated, truncated, infos

    def _get_observations(self):
        # Vectorized observation computation
        num_agents = len(self.agents)
        if num_agents == 0:
            return np.zeros((0, self.obs_dim), dtype=np.float32)
            
        # 1. Gather agent states
        agent_pos = np.array([a.pos for a in self.agents]) # (N, 2)
        agent_angles = np.array([a.angle for a in self.agents]) # (N,)
        agent_speeds = np.array([a.speed for a in self.agents]) # (N,)
        agent_carrying = np.array([1.0 if a.carrying_food else 0.0 for a in self.agents]) # (N,)
        
        # 2. Raycasting (Vectorized)
        # Rays angles: (N, NUM_RAYS)
        half_span = RAY_SPAN_DEG / 2.0
        ray_angles = np.linspace(
            agent_angles - half_span, 
            agent_angles + half_span, 
            NUM_RAYS, axis=1
        ) # (N, NUM_RAYS)
        
        # Convert to radians
        ray_rads = np.radians(ray_angles)
        ray_dirs = np.stack([np.cos(ray_rads), np.sin(ray_rads)], axis=2) # (N, NUM_RAYS, 2)
        
        # Wall distances
        # Vertical walls (x=0, x=W)
        # t = (target_x - p_x) / dir_x
        # We need to handle div by zero carefully, but numpy handles inf
        with np.errstate(divide='ignore', invalid='ignore'):
            t_left = (0 - agent_pos[:, 0, None]) / ray_dirs[:, :, 0]
            t_right = (SCREEN_WIDTH - agent_pos[:, 0, None]) / ray_dirs[:, :, 0]
            t_top = (0 - agent_pos[:, 1, None]) / ray_dirs[:, :, 1]
            t_bottom = (SCREEN_HEIGHT - agent_pos[:, 1, None]) / ray_dirs[:, :, 1]
        
        # Filter negative t (behind) and select min positive
        t_walls = np.full((num_agents, NUM_RAYS), RAY_MAX_DIST, dtype=np.float32)
        
        for t in [t_left, t_right, t_top, t_bottom]:
            # Replace negative or NaN with inf
            t_valid = np.where((t > 0) & (t < RAY_MAX_DIST), t, np.inf)
            t_walls = np.minimum(t_walls, t_valid)
            
        # Agent intersections
        # This is O(N^2 * Rays), but vectorized
        # For each agent i, ray r, check intersection with agent j
        # We can skip this for now if N is small, or implement a simplified check
        # Simplified: Check if any agent is close to the ray line segment
        # Full vectorization is complex. Let's stick to the loop for agent-agent check for now
        # but optimized.
        
        # Actually, let's just use the wall distances for now and add agent checks if needed.
        # The user said "avoid collisions with walls and other agents".
        # So agents should be visible in rays.
        
        # Hybrid approach: Use the vectorized wall calculation, then loop for agents
        # but only for agents that are close.
        
        # Let's keep the per-agent loop for rays for simplicity and correctness for now,
        # but optimize the math.
        # The previous implementation was fine for 5 agents.
        # The bottleneck is likely Python overhead.
        # Let's stick to the previous implementation but clean it up?
        # No, I promised optimization.
        
        # Let's use the vectorized wall calculation at least.
        rays = t_walls
        
        # Now check agents
        # Broadcast agent positions: (N, 1, 2) vs (1, N, 2) -> (N, N, 2)
        rel_pos = agent_pos[None, :, :] - agent_pos[:, None, :] # (N_targets, N_sources, 2)
        dists_sq = np.sum(rel_pos**2, axis=2)
        
        # For each source agent, filter targets within RAY_MAX_DIST + Radius
        # This is still hard to fully vectorize efficiently without complex indexing.
        # Let's revert to a cleaner loop for agent-ray intersection, but using numpy.
        
        for i in range(num_agents):
            p1 = agent_pos[i]
            for r in range(NUM_RAYS):
                ray_dir = ray_dirs[i, r]
                min_d = rays[i, r]
                
                # Check all other agents
                # Vector from p1 to other agents
                # v = agent_pos - p1 # (N, 2)
                # We can filter by distance first
                # But let's just iterate
                for j in range(num_agents):
                    if i == j: continue
                    
                    # Ray-Circle intersection
                    # geometric solution
                    f = p1 - agent_pos[j]
                    a = 1.0 # dir is unit vector
                    b = 2 * np.dot(f, ray_dir)
                    c = np.dot(f, f) - AGENT_RADIUS**2
                    
                    delta = b*b - 4*a*c
                    if delta >= 0:
                        sqrt_delta = math.sqrt(delta)
                        t1 = (-b - sqrt_delta) / (2*a)
                        # t2 = (-b + sqrt_delta) / (2*a)
                        
                        if 0 < t1 < min_d:
                            min_d = t1
                rays[i, r] = min_d

        # Normalize rays
        rays_norm = rays / RAY_MAX_DIST
        
        # 3. Nearest Food
        # Dist matrix: (N, M_food)
        if self.foods:
            food_pos = np.array([f.pos for f in self.foods])
            # (N, 1, 2) - (1, M, 2) -> (N, M, 2)
            vec_to_food = food_pos[None, :, :] - agent_pos[:, None, :]
            dists_food = np.linalg.norm(vec_to_food, axis=2)
            
            min_idx = np.argmin(dists_food, axis=1)
            min_dists = dists_food[np.arange(num_agents), min_idx]
            min_vecs = vec_to_food[np.arange(num_agents), min_idx]
            
            # Normalize
            nf_dist_norm = 1.0 - np.clip(min_dists / RAY_MAX_DIST, 0.0, 1.0)
            nf_vec_norm = min_vecs / (min_dists[:, None] + 1e-8)
        else:
            nf_dist_norm = np.zeros(num_agents)
            nf_vec_norm = np.zeros((num_agents, 2))

        # 4. Nest
        nest_pos = np.array(NEST_POSITION)
        vec_to_nest = nest_pos[None, :] - agent_pos
        dists_nest = np.linalg.norm(vec_to_nest, axis=1)
        nest_dist_norm = 1.0 - np.clip(dists_nest / SCREEN_WIDTH, 0.0, 1.0)
        nest_vec_norm = vec_to_nest / (dists_nest[:, None] + 1e-8)

        # 5. Nearest Agent
        # dists_sq is (N, N)
        # Set diagonal to inf
        np.fill_diagonal(dists_sq, np.inf)
        min_a_idx = np.argmin(dists_sq, axis=1)
        min_a_dists = np.sqrt(dists_sq[np.arange(num_agents), min_a_idx])
        
        vec_to_a = agent_pos[min_a_idx] - agent_pos
        
        # Bearing
        # angle of vec_to_a
        target_angles = np.degrees(np.arctan2(vec_to_a[:, 1], vec_to_a[:, 0]))
        rel_angles = (target_angles - agent_angles + 180) % 360 - 180
        na_bearing = rel_angles / 180.0
        
        na_dist_norm = 1.0 - np.clip(min_a_dists / RAY_MAX_DIST, 0.0, 1.0)

        # Assemble
        # rays: (N, NUM_RAYS)
        # nf: (N, 3)
        # speed: (N, 1)
        # carry: (N, 1)
        # nest: (N, 3)
        # na: (N, 2)
        
        obs = np.concatenate([
            rays_norm,
            nf_dist_norm[:, None], nf_vec_norm,
            (agent_speeds / AGENT_MAX_SPEED)[:, None],
            agent_carrying[:, None],
            nest_dist_norm[:, None], nest_vec_norm,
            na_dist_norm[:, None], na_bearing[:, None]
        ], axis=1)
        
        return obs.astype(np.float32)

    def _handle_agent_collisions(self):
        for i, a1 in enumerate(self.agents):
            for j, a2 in enumerate(self.agents):
                if i >= j: continue
                
                dist = np.linalg.norm(a1.pos - a2.pos)
                if dist < 2 * AGENT_RADIUS:
                    overlap = 2 * AGENT_RADIUS - dist
                    if dist > 1e-6:
                        direction = (a1.pos - a2.pos) / dist
                    else:
                        direction = np.random.randn(2)
                        direction /= np.linalg.norm(direction) + 1e-8
                    
                    # Push apart
                    a1.pos += direction * (overlap / 2)
                    a2.pos -= direction * (overlap / 2)
                    
                    # Bounce angles
                    # Simple elastic collision-like response (just flip angle for now or random)
                    # Original code: (angle + 180) % 360
                    a1.angle = (a1.angle + 180) % 360
                    a2.angle = (a2.angle + 180) % 360
                    
                    a1.reward += REWARD_AGENT_COLLISION
                    a2.reward += REWARD_AGENT_COLLISION

    def _handle_food_interactions(self):
        # Respawn logic
        self.frame_count += 1
        if self.frame_count >= FOOD_RESPAWN_INTERVAL:
            self.frame_count = 0
            if len(self.foods) < FOOD_RESPAWN_THRESHOLD:
                self.foods.append(Food())
                
        nest_pos = np.array(NEST_POSITION)
        
        for agent in self.agents:
            if not agent.carrying_food:
                # Check pickup
                for f in list(self.foods):
                    if np.linalg.norm(agent.pos - f.pos) < AGENT_RADIUS + FOOD_RADIUS:
                        agent.carrying_food = True
                        agent.reward += REWARD_FOOD_PICKUP
                        if f in self.foods:
                            self.foods.remove(f)
                        break
            else:
                # Check deposit
                if np.linalg.norm(agent.pos - nest_pos) < NEST_RADIUS:
                    agent.carrying_food = False
                    agent.reward += REWARD_FOOD_DEPOSIT

    def render(self):
        if self.render_mode is None:
            return
            
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Neural Swarm PPO")
            else:
                self.screen = pygame.Surface((self.width, self.height))
                
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        if self.font is None:
            self.font = pygame.font.SysFont("consolas", 14)
            
        self.screen.fill(BG_COLOR)
        
        # Draw Nest
        pygame.draw.circle(self.screen, NEST_COLOR, NEST_POSITION, NEST_RADIUS)
        
        # Draw Food
        for f in self.foods:
            f.draw(self.screen)
            
        # Draw Agents
        for i, agent in enumerate(self.agents):
            agent.draw(self.screen)
            if self.debug and i == self.selected_agent:
                 # Draw rays for selected agent
                 self._draw_rays(agent, self.screen)
                 # Draw ring
                 pygame.draw.circle(self.screen, (255, 255, 255), agent.pos.astype(int), AGENT_RADIUS + 3, 1)

        # Draw HUD
        if self.debug:
            self._draw_global_hud(self.screen)
            self._draw_agent_hud(self.screen)
            
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def _draw_rays(self, agent, surface):
        # Reconstruct rays from agent state for visualization
        half_span = RAY_SPAN_DEG / 2.0
        for i in range(NUM_RAYS):
            frac = 0.5 if NUM_RAYS == 1 else i / (NUM_RAYS - 1)
            ray_ang = agent.angle - half_span + frac * RAY_SPAN_DEG
            dist = agent._ray_cast(ray_ang, self.agents)
            
            rad = math.radians(ray_ang)
            end_pos = agent.pos + np.array([math.cos(rad), math.sin(rad)]) * dist
            
            pygame.draw.line(surface, (200, 200, 60), agent.pos, end_pos, 1)
            pygame.draw.circle(surface, (200, 200, 60), end_pos.astype(int), 2)

    def _draw_overlay_text(self, text, x, y, surface):
        label = self.font.render(text, True, HUD_TEXT_COLOR)
        surface.blit(label, (x, y))

    def _draw_global_hud(self, surface):
        x, y = HUD_PADDING, HUD_PADDING
        lines = [
            f"Agents: {len(self.agents)}",
            f"Foods: {len(self.foods)}",
            f"Debug: {self.debug}",
            f"Net Reward: {sum(a.cumulative_reward for a in self.agents):+.2f}",
            f"Controls: Arrow Keys, TAB switch, D debug, R reset",
        ]
        for line in lines:
            self._draw_overlay_text(line, x, y, surface)
            y += FONT_SIZE + 2

    def _draw_agent_hud(self, surface):
        if not self.agents: return
        sel = self.agents[self.selected_agent]
        x = self.width - 220 # Move to left to fit
        y = HUD_PADDING
        
        # Get last sensors if available, or recompute
        obs = sel.compute_sensors(self.foods, self.agents)
        rays = obs[:NUM_RAYS]
        
        lines = [
            f"Selected #{self.selected_agent}",
            f"Angle: {sel.angle:.1f}",
            f"Speed: {sel.speed:.2f}",
            f"Carrying: {sel.carrying_food}",
            f"Reward: {sel.cumulative_reward:+.4f}",
            "Rays:",
            ", ".join([f"{r:.2f}" for r in rays]),
        ]
        for line in lines:
            self._draw_overlay_text(line, x, y, surface)
            y += FONT_SIZE + 2

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


# --- Helper Classes (Internal) ---

class Food:
    def __init__(self):
        self.pos = np.array([
            random.randint(FOOD_RADIUS * 2, SCREEN_WIDTH - FOOD_RADIUS * 2),
            random.randint(FOOD_RADIUS * 2, SCREEN_HEIGHT - FOOD_RADIUS * 2)
        ], dtype=float)
        
    def draw(self, surface):
        pygame.draw.circle(surface, FOOD_COLOR, self.pos.astype(int), FOOD_RADIUS)

class Agent:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=float)
        self.angle = random.uniform(0, 360)
        self.speed = AGENT_SPEED # Default speed
        self.carrying_food = False
        self.reward = 0.0
        self.cumulative_reward = 0.0
        
    def take_action(self, action):
        if action == 1: # Left
            self.angle = (self.angle - AGENT_TURN_RATE) % 360
        elif action == 2: # Right
            self.angle = (self.angle + AGENT_TURN_RATE) % 360
        elif action == 3: # Accel
            self.speed = min(AGENT_MAX_SPEED, self.speed + 0.2)
        elif action == 4: # Decel
            self.speed = max(0, self.speed - 0.2)
        # 0 is no-op
        
    def update(self):
        # Move
        rad = math.radians(self.angle)
        dx = math.cos(rad) * self.speed
        dy = math.sin(rad) * self.speed
        self.pos += np.array([dx, dy])
        
        # Wall Collision
        bounced = False
        if self.pos[0] < AGENT_RADIUS:
            self.pos[0] = AGENT_RADIUS
            self.angle = 180 - self.angle
            bounced = True
        elif self.pos[0] > SCREEN_WIDTH - AGENT_RADIUS:
            self.pos[0] = SCREEN_WIDTH - AGENT_RADIUS
            self.angle = 180 - self.angle
            bounced = True
            
        if self.pos[1] < AGENT_RADIUS:
            self.pos[1] = AGENT_RADIUS
            self.angle = -self.angle
            bounced = True
        elif self.pos[1] > SCREEN_HEIGHT - AGENT_RADIUS:
            self.pos[1] = SCREEN_HEIGHT - AGENT_RADIUS
            self.angle = -self.angle
            bounced = True
            
        self.angle %= 360
        return bounced

    def draw(self, surface):
        SCALING = 1.5
        color = (255, 60, 60) if self.carrying_food else AGENT_COLOR
        
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

    def compute_sensors(self, foods, agents):
        # 1. Raycasts
        rays = []
        half_span = RAY_SPAN_DEG / 2.0
        for i in range(NUM_RAYS):
            frac = 0.5 if NUM_RAYS == 1 else i / (NUM_RAYS - 1)
            ray_ang = self.angle - half_span + frac * RAY_SPAN_DEG
            dist = self._ray_cast(ray_ang, agents)
            rays.append(dist / RAY_MAX_DIST)
            
        # 2. Nearest Food
        nf_dist = RAY_MAX_DIST
        nf_vec = np.zeros(2)
        for f in foods:
            d = np.linalg.norm(f.pos - self.pos)
            if d < nf_dist:
                nf_dist = d
                nf_vec = (f.pos - self.pos) / (d + 1e-8)
        nf_dist_norm = 1.0 - min(nf_dist / RAY_MAX_DIST, 1.0)
        
        # 3. Nest
        nest_pos = np.array(NEST_POSITION)
        nest_d = np.linalg.norm(nest_pos - self.pos)
        nest_vec = (nest_pos - self.pos) / (nest_d + 1e-8)
        nest_d_norm = 1.0 - min(nest_d / SCREEN_WIDTH, 1.0)
        
        # 4. Nearest Agent
        na_dist = RAY_MAX_DIST
        na_bearing = 0.0
        for a in agents:
            if a is self: continue
            d = np.linalg.norm(a.pos - self.pos)
            if d < na_dist:
                na_dist = d
                vec = a.pos - self.pos
                # Bearing relative to agent heading
                abs_angle = math.degrees(math.atan2(vec[1], vec[0]))
                rel_angle = (abs_angle - self.angle + 180) % 360 - 180
                na_bearing = rel_angle / 180.0 # Normalize -1 to 1
        na_dist_norm = 1.0 - min(na_dist / RAY_MAX_DIST, 1.0)
        
        # 5. Internal
        speed_norm = self.speed / AGENT_MAX_SPEED
        carrying = 1.0 if self.carrying_food else 0.0
        
        return np.concatenate([
            rays,
            [nf_dist_norm, nf_vec[0], nf_vec[1]],
            [speed_norm],
            [carrying],
            [nest_d_norm, nest_vec[0], nest_vec[1]],
            [na_dist_norm, na_bearing]
        ], dtype=np.float32)

    def _ray_cast(self, angle_deg, agents):
        rad = math.radians(angle_deg)
        dir_v = np.array([math.cos(rad), math.sin(rad)])
        p1 = self.pos
        p2 = p1 + dir_v * RAY_MAX_DIST
        
        min_dist = RAY_MAX_DIST
        
        # Wall Intersections (Line-Line)
        # Vertical walls x=0, x=W
        if dir_v[0] != 0:
            t1 = (0 - p1[0]) / dir_v[0]
            t2 = (SCREEN_WIDTH - p1[0]) / dir_v[0]
            if 0 < t1 < min_dist: min_dist = t1
            if 0 < t2 < min_dist: min_dist = t2
            
        # Horizontal walls y=0, y=H
        if dir_v[1] != 0:
            t1 = (0 - p1[1]) / dir_v[1]
            t2 = (SCREEN_HEIGHT - p1[1]) / dir_v[1]
            if 0 < t1 < min_dist: min_dist = t1
            if 0 < t2 < min_dist: min_dist = t2
            
        # Agent Intersections (Line-Circle)
        # Simplified: Check distance to other agents, if within cone
        # Proper ray-circle intersection:
        for a in agents:
            if a is self: continue
            
            # Vector to circle center
            f = p1 - a.pos
            a_quad = np.dot(dir_v, dir_v)
            b_quad = 2 * np.dot(f, dir_v)
            c_quad = np.dot(f, f) - AGENT_RADIUS**2
            
            discriminant = b_quad*b_quad - 4*a_quad*c_quad
            if discriminant >= 0:
                sqrt_d = math.sqrt(discriminant)
                t1 = (-b_quad - sqrt_d) / (2*a_quad)
                t2 = (-b_quad + sqrt_d) / (2*a_quad)
                
                if 0 < t1 < min_dist: min_dist = t1
                # t2 is the far side, ignore
                
        return min_dist
