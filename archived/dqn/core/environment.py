import pygame
import numpy as np
import random
import math
from .config import *
from .agent import Agent
from .entities import Food

class GameEnvironment:
    def __init__(self, num_agents=5, manual_control=True, debug=DEBUG_MODE, headless=False):
        self.headless = headless
        pygame.init()
        
        if not self.headless:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Neural Swarm Simulation")

            global FONT
            FONT = pygame.font.SysFont(FONT_NAME, FONT_SIZE)
        else:
            self.screen = None

        self.clock = pygame.time.Clock()
        self.debug = debug
        self.manual_control = manual_control

        self.num_agents = num_agents
        self.agents = [Agent(random.randint(100, 700), random.randint(100, 500)) for _ in range(num_agents)]
        self.foods = [Food() for _ in range(min(MAX_FOOD, 12))]
        self.frame_count = 0
        self.running = True
        self.selected_agent = 0

    # ------------------------------
    # Input & Controls
    # ------------------------------
    def handle_input(self):
        keys = pygame.key.get_pressed()
        agent = self.agents[self.selected_agent]

        if keys[pygame.K_LEFT]:
            agent.rotate(-1)
        if keys[pygame.K_RIGHT]:
            agent.rotate(1)
        if keys[pygame.K_UP]:
            agent.accelerate(0.05)
        if keys[pygame.K_DOWN]:
            agent.accelerate(-0.05)
        if keys[pygame.K_TAB]:
            self.selected_agent = (self.selected_agent + 1) % len(self.agents)
        if keys[pygame.K_r]:
            self.reset()
        if keys[pygame.K_d]:
            self.debug = not self.debug

    # ------------------------------
    # Core Physics / Logic
    # ------------------------------
    def handle_agent_collisions(self):
        for i, a1 in enumerate(self.agents):
            for j, a2 in enumerate(self.agents):
                if i >= j:
                    continue
                dist = np.linalg.norm(a1.pos - a2.pos)
                if dist < 2 * AGENT_RADIUS:
                    overlap = 2 * AGENT_RADIUS - dist
                    if dist > 1e-6:
                        direction = (a1.pos - a2.pos) / dist
                    else:
                        direction = np.random.randn(2)
                        direction /= np.linalg.norm(direction) + 1e-8
                    a1.pos += direction * (overlap / 2)
                    a2.pos -= direction * (overlap / 2)
                    a1.angle = (a1.angle + 180) % 360
                    a2.angle = (a2.angle + 180) % 360
                    a1.reward += REWARD_AGENT_COLLISION
                    a2.reward += REWARD_AGENT_COLLISION

    def handle_food_interactions(self):
        for agent in self.agents:
            if not agent.carrying_food:
                for f in list(self.foods):
                    if np.linalg.norm(agent.pos - f.pos) < AGENT_RADIUS + FOOD_RADIUS:
                        agent.carrying_food = True
                        agent.reward += REWARD_FOOD_PICKUP
                        try:
                            self.foods.remove(f)
                        except ValueError:
                            pass
                        break
            else:
                if np.linalg.norm(agent.pos - np.array(NEST_POSITION)) < NEST_RADIUS:
                    agent.carrying_food = False
                    agent.reward += REWARD_FOOD_DEPOSIT

        # respawn check
        self.frame_count += 1
        if self.frame_count >= FOOD_RESPAWN_INTERVAL:
            self.frame_count = 0
            if len(self.foods) < FOOD_RESPAWN_THRESHOLD:
                self.foods.append(Food())

    def reset(self):
        self.agents = [Agent(random.randint(100, 700), random.randint(100, 500)) for _ in range(self.num_agents)]
        self.foods = [Food() for _ in range(min(MAX_FOOD, 12))]
        self.frame_count = 0
        self.selected_agent = 0
        for a in self.agents:
            a.carrying_food = False
            a.reward = 0.0

    def step(self):
        # apply per-step penalty (encourage efficiency)
        for agent in self.agents:
            agent.reward += REWARD_STEP

        # 1) update physics & compute sensors
        for agent in self.agents:
            agent.update()
        for agent in self.agents:
            agent.compute_sensors(self.foods, self.agents)

        # 2) collisions & interactions
        self.handle_agent_collisions()
        self.handle_food_interactions()

    # ------------------------------
    # Rendering & Debug Overlays
    # ------------------------------
    def _draw_rays_for_agent(self, agent, surface):
        half = RAY_SPAN_DEG / 2.0
        for i in range(NUM_RAYS):
            frac = i / (NUM_RAYS - 1) if NUM_RAYS > 1 else 0.5
            ray_ang = agent.angle - half + frac * RAY_SPAN_DEG
            ang_rad = math.radians(ray_ang)
            dist = int(agent.last_sensors[i] * RAY_MAX_DIST)
            sx, sy = int(agent.pos[0]), int(agent.pos[1])
            ex = int(sx + math.cos(ang_rad) * dist)
            ey = int(sy + math.sin(ang_rad) * dist)
            pygame.draw.line(surface, (200, 200, 60), (sx, sy), (ex, ey), 1)
            pygame.draw.circle(surface, (200, 200, 60), (ex, ey), 2)

    def _draw_overlay_text(self, text, x, y, surface):
        label = FONT.render(text, True, HUD_TEXT_COLOR)
        surface.blit(label, (x, y))

    def _draw_global_hud(self, surface):
        x, y = HUD_PADDING, HUD_PADDING
        lines = [
            f"Agents: {len(self.agents)}",
            f"Foods: {len(self.foods)}",
            f"Debug: {self.debug}",
            f"Net Reward: {sum(a.reward for a in self.agents):+.2f}",
            f"Controls: Arrow Keys, TAB switch, D debug, R reset",
        ]
        for line in lines:
            self._draw_overlay_text(line, x, y, surface)
            y += FONT_SIZE + 2

    def _draw_agent_hud(self, surface):
        sel = self.agents[self.selected_agent]
        x = 0.8 * SCREEN_WIDTH
        y = HUD_PADDING
        lines = [
            f"Selected #{self.selected_agent}",
            f"Angle: {sel.angle:.1f} | Speed: {sel.speed:.2f}",
            f"Carrying: {sel.carrying_food} | Reward: {sel.reward:+.2f}",
            "Rays:",
            ", ".join([f"{r:.2f}" for r in sel.last_sensors[:NUM_RAYS]]),
        ]
        for line in lines:
            self._draw_overlay_text(line, x, y, surface)
            y += FONT_SIZE + 2

    def render(self):
        if self.headless:
            return

        self.screen.fill(BG_COLOR)

        # --- Draw nest first ---
        pygame.draw.circle(self.screen, NEST_COLOR, NEST_POSITION, NEST_RADIUS)

        # --- Draw food and agents ---
        for f in self.foods:
            f.draw(self.screen)
        for agent in self.agents:
            agent.draw(self.screen)
            if self.debug:
                self._draw_rays_for_agent(agent, self.screen)

        # --- Draw selection ring if in manual control ---
        if self.manual_control and self.agents:
            selected = self.agents[self.selected_agent]
            pygame.draw.circle(self.screen, (255, 255, 255), selected.pos.astype(int), AGENT_RADIUS + 3, 1)

        # --- HUD overlays ---
        if self.debug:
            self._draw_global_hud(self.screen)
            self._draw_agent_hud(self.screen)

        pygame.display.flip()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            if self.manual_control:
                self.handle_input()

            self.step()
            self.render()
            self.clock.tick(FPS)

        pygame.quit()
