# main_simulation.py (Corrected)

import pygame
import numpy as np
import math
import random
import time
import heapq
from numpy.linalg import pinv
import socket
import pickle

# --- Network Settings for Broadcasting Data ---
HOST = '127.0.0.1'  # Localhost
PORT = 65432        # Port to send to (must match client)

# ==============================================================================
# 0. EKF + TTC ALGORITHM
# ==============================================================================
class Pathpredictor:
    def __init__(self, initial_state, initial_P):
        self.x = np.array(initial_state, dtype=float).reshape(-1, 1)
        self.P = np.array(initial_P, dtype=float)
        q_pos_vel, q_accel, q_bias = 0.01, 0.5, 1e-4
        self.Q = np.diag([q_pos_vel, q_pos_vel, q_pos_vel, q_pos_vel, q_accel, q_accel, q_bias, q_bias])
        self.R_COOP, self.R_SOFT_START = np.diag([0.1**2, 0.1**2]), np.diag([5.0**2, 5.0**2])
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]])
        self.VALIDATION_THRESHOLD, self.cycle_count, self.SOFT_START_CYCLES = 15.0, 0, 10

    def _calculate_Fk(self, dt, theta_z):
        phi = theta_z * dt
        cos_phi, sin_phi = math.cos(phi), math.sin(phi)
        R = np.array([[cos_phi, -sin_phi], [sin_phi, cos_phi]])
        Fk = np.identity(8)
        Fk[0, 2], Fk[1, 3] = dt, dt
        Fk[0, 4], Fk[1, 5] = 0.5 * dt**2, 0.5 * dt**2
        Fk[2, 2], Fk[2, 3] = R[0, 0], R[0, 1]
        Fk[3, 2], Fk[3, 3] = R[1, 0], R[1, 1]
        Fk[2, 4], Fk[2, 5] = R[0, 0] * dt, R[0, 1] * dt
        Fk[3, 4], Fk[3, 5] = R[1, 0] * dt, R[1, 1] * dt
        return Fk

    def prediction_step(self, x_prev, P_prev, dt, ax_imu, ay_imu, theta_z):
        x, y, vx, vy, ax, ay, _, _ = x_prev.flatten()
        x_pred = x + vx * dt + 0.5 * ax * dt**2
        y_pred = y + vy * dt + 0.5 * ay * dt**2
        phi = theta_z * dt
        R_phi = np.array([[math.cos(phi), -math.sin(phi)], [math.sin(phi), math.cos(phi)]])
        v_vec, a_vec = np.array([[vx], [vy]]), np.array([[ax], [ay]])
        v_new = R_phi @ v_vec + R_phi @ a_vec * dt
        vx_pred, vy_pred = v_new.flatten()
        x_predicted = np.array([x_pred, y_pred, vx_pred, vy_pred, ax, ay, 0, 0]).reshape(-1, 1)
        Fk = self._calculate_Fk(dt, theta_z)
        P_predicted = Fk @ P_prev @ Fk.T + self.Q
        return x_predicted, P_predicted

    def correction_step(self, x_pred, P_pred, z_coop_data):
        self.cycle_count += 1
        if z_coop_data is None: return x_pred, P_pred
        R_current = self.R_COOP if self.cycle_count >= self.SOFT_START_CYCLES else self.R_SOFT_START
        h_x = self.H @ x_pred
        y_k = z_coop_data.reshape(-1, 1) - h_x
        S_k = self.H @ P_pred @ self.H.T + R_current + 1e-6 * np.eye(2)
        mahalanobis_d2 = float(y_k.T @ pinv(S_k) @ y_k)
        if mahalanobis_d2 > self.VALIDATION_THRESHOLD: return x_pred, P_pred
        K_k = P_pred @ self.H.T @ pinv(S_k)
        x_corrected = x_pred + K_k @ y_k
        P_corrected = (np.identity(len(self.x)) - K_k @ self.H) @ P_pred
        return x_corrected, P_corrected

    def extrapolate(self, x, y, vx, vy, ax, ay, t_horizon, n_steps):
        path = []
        path_eq = lambda s, v, a, t: s + v * t + 0.5 * a * t * t
        for i in range(1, n_steps + 1):
            t = t_horizon * (i / n_steps)
            path.append((path_eq(x, vx, ax, t), path_eq(y, vy, ay, t)))
        return path

    def predict_ttc(self, my_state_6dof, peer_state_6dof, t_horizon, n_steps, collision_radius):
        my_path = self.extrapolate(*my_state_6dof, t_horizon, n_steps)
        peer_path = self.extrapolate(*peer_state_6dof, t_horizon, n_steps)
        dt_sample = t_horizon / n_steps
        for i, (my_xy, peer_xy) in enumerate(zip(my_path, peer_path)):
            dist = math.hypot(my_xy[0] - peer_xy[0], my_xy[1] - peer_xy[1])
            if dist < collision_radius:
                return (i + 1) * dt_sample
        return None

# ==============================================================================
# 1. SIMULATION SETUP & CONSTANTS
# ==============================================================================
SIM_WIDTH, SIM_HEIGHT = 900, 700
FPS = 60
GREEN, BLACK, WHITE, RED = (20, 80, 30), (0, 0, 0), (255, 255, 255), (255, 50, 50)
CRASHED_COLOR = (80, 0, 0)
SIMULATION_SPEED_FACTOR, VEHICLE_COUNT = 0.5, 20
SENSOR_RADIUS, TTC_THRESHOLD, COLLISION_RADIUS, MAX_SPEED = 300, 2.5, 12, 60
COOP_UPDATE_PROB = 0.2

# ==============================================================================
# 2. MAP & ENVIRONMENT CLASS
# ==============================================================================
class RoadMap:
    def __init__(self, width, height, cell_size=50):
        self.cell_size = cell_size
        self.grid_w, self.grid_h = width // cell_size, height // cell_size
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=int)
        self.road_points = []
        self._create_layout()

    def _create_layout(self):
        layout = [
            " RRR   RRRRRRRRR "," R R   R       R "," R RRRRR R RRRRR ",
            " R   R   R R   R "," RRR R RRR R R R ","   R R R   R R R ",
            " RRR R R RRR R R "," R   R R     R R "," R RRR RRRRRRR R ",
            " R R       R   R "," R R RRRRRRRRRRR "," R R R           ",
            " RRR R RRRRRRRRR ","     R           ",
        ]
        for r, row_str in enumerate(layout):
            for c, char in enumerate(row_str):
                if 0 <= r < self.grid_h and 0 <= c < self.grid_w and char == 'R':
                    self.grid[r, c] = 1
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                if self.grid[r, c] == 1:
                    self.road_points.append(self.get_world_coords(r, c))

    def get_random_road_pos(self): return random.choice(self.road_points).copy()
    def get_grid_coords(self, pos): return int(pos.y / self.cell_size), int(pos.x / self.cell_size)
    def get_world_coords(self, r, c): return pygame.Vector2(c * self.cell_size + self.cell_size / 2, r * self.cell_size + self.cell_size / 2)

    def find_path(self, start_pos, end_pos):
        start_node, end_node = self.get_grid_coords(start_pos), self.get_grid_coords(end_pos)
        open_set = [(0, start_node)]
        came_from = {}
        g_score = { (r, c): float('inf') for r in range(self.grid_h) for c in range(self.grid_w) }
        g_score[start_node] = 0
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == end_node:
                path = []
                while current in came_from:
                    path.append(self.get_world_coords(current[0], current[1]))
                    current = came_from[current]
                return path[::-1]
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                nr, nc = neighbor
                if not (0 <= nr < self.grid_h and 0 <= nc < self.grid_w and self.grid[nr, nc] == 1): continue
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor], g_score[neighbor] = current, tentative_g_score
                    h_score = abs(nr - end_node[0]) + abs(nc - end_node[1])
                    heapq.heappush(open_set, (tentative_g_score + h_score, neighbor))
        return None

    def draw(self, surface):
        surface.fill(GREEN)
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                if self.grid[r, c] == 1:
                    pygame.draw.rect(surface, BLACK, (c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size))

# ==============================================================================
# 3. VEHICLE CLASS
# ==============================================================================
class Car:
    def __init__(self, car_id, road_map):
        self.id = car_id
        self.map = road_map
        self.pos = self.map.get_random_road_pos()
        self.path_waypoints = []
        self.vel, self.acc, self.angle = pygame.Vector2(0, 0), pygame.Vector2(0, 0), 0
        self.color = [random.randint(100, 255) for _ in range(3)]
        self.ekf = Pathpredictor([self.pos.x, self.pos.y, 0, 0, 0, 0, 0, 0], np.eye(8) * 0.1)
        self.ttc_info = {'ttc': float('inf'), 'is_warning': False}
        self.predicted_path = []
        self.is_finished, self.is_crashed = False, False
        self.current_waypoint_idx = 0

    def set_new_destination(self):
        while True:
            destination = self.map.get_random_road_pos()
            if self.pos.distance_to(destination) > 200:
                path = self.map.find_path(self.pos, destination)
                if path:
                    self.path_waypoints = path
                    self.current_waypoint_idx = 0
                    return

    def get_current_waypoint(self):
        if self.path_waypoints and self.current_waypoint_idx < len(self.path_waypoints):
            return self.path_waypoints[self.current_waypoint_idx]
        return None

    def update(self, dt, all_cars, simulation_stats):
        if self.is_finished or self.is_crashed: return
        if not self.path_waypoints: self.set_new_destination()
        
        waypoint = self.get_current_waypoint()
        if not waypoint:
            self.is_finished = True
            return
        
        if self.pos.distance_to(waypoint) < self.map.cell_size * 0.6:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(self.path_waypoints):
                self.is_finished = True
                return
        
        self._update_navigation(dt, self.get_current_waypoint())
        self._run_ekf_cycle(dt)
        self._check_for_ttc(all_cars, simulation_stats)
        
        # FIX: Synchronize the EKF state with the ground truth before extrapolation
        self.ekf.x[0] = self.pos.x
        self.ekf.x[1] = self.pos.y
        self.ekf.x[2] = self.vel.x
        self.ekf.x[3] = self.vel.y
        self.predicted_path = self.ekf.extrapolate(*self.ekf.x[0:6].flatten(), t_horizon=1.5, n_steps=20)

    def _update_navigation(self, dt, waypoint):
        dist = self.pos.distance_to(waypoint)
        target_speed = MAX_SPEED * (dist / self.map.cell_size) if dist < self.map.cell_size else MAX_SPEED
        desired_vel = (waypoint - self.pos)
        if desired_vel.length() > 0: desired_vel.scale_to_length(target_speed)
        
        self.acc = (desired_vel - self.vel)
        self.vel += self.acc * dt
        if self.vel.length() > MAX_SPEED: self.vel.scale_to_length(MAX_SPEED)
        self.pos += self.vel * dt
        if self.vel.length() > 1: self.angle = self.vel.angle_to(pygame.Vector2(1, 0))

    def _run_ekf_cycle(self, dt):
        theta_z = self.vel.as_polar()[1] - self.angle if self.vel.length() > 0 else 0
        self.ekf.x, self.ekf.P = self.ekf.prediction_step(self.ekf.x, self.ekf.P, dt, self.acc.x, self.acc.y, theta_z)
        if random.random() < COOP_UPDATE_PROB:
            coop_data = np.array([self.pos.x + np.random.normal(0, 0.5), self.pos.y + np.random.normal(0, 0.5)])
            self.ekf.x, self.ekf.P = self.ekf.correction_step(self.ekf.x, self.ekf.P, coop_data)

    def _check_for_ttc(self, all_cars, simulation_stats):
        min_ttc = float('inf')
        my_state = self.ekf.x[0:6].flatten()
        
        for peer in all_cars:
            if peer.id == self.id or peer.is_crashed: continue
            if self.pos.distance_to(peer.pos) < SENSOR_RADIUS:
                ttc = self.ekf.predict_ttc(my_state, peer.ekf.x[0:6].flatten(), 3.0, 30, COLLISION_RADIUS)
                if ttc is not None:
                    if ttc < min_ttc: min_ttc = ttc
                    if ttc < TTC_THRESHOLD:
                        simulation_stats['predicted_collisions'].add(frozenset([self.id, peer.id]))

        self.ttc_info['ttc'] = min_ttc if min_ttc != float('inf') else -1
        self.ttc_info['is_warning'] = (self.ttc_info['ttc'] != -1 and self.ttc_info['ttc'] < TTC_THRESHOLD)

    def draw(self, surface):
        if self.predicted_path and not self.is_crashed:
            # This now draws from the correct, synchronized starting point.
            path_points = [self.pos] + [(int(p[0]), int(p[1])) for p in self.predicted_path]
            if len(path_points) > 1: pygame.draw.lines(surface, self.color, False, path_points, 1)

        draw_color = CRASHED_COLOR if self.is_crashed else self.color
        pygame.draw.circle(surface, draw_color, self.pos, 8)
        pygame.draw.circle(surface, BLACK, self.pos, 8, 1)
        if not self.is_crashed:
            end_pos = self.pos + pygame.Vector2(10, 0).rotate(self.angle)
            pygame.draw.line(surface, WHITE, self.pos, end_pos, 2)
            if self.ttc_info['is_warning'] and int(time.time() * 10) % 2 == 0:
                pygame.draw.circle(surface, RED, self.pos, 10, 2)

# ==============================================================================
# 5. MAIN SIMULATION MANAGER
# ==============================================================================
class SimulationManager:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((SIM_WIDTH, SIM_HEIGHT))
        pygame.display.set_caption("EKF Collision Prediction Simulation (Server)")
        self.clock = pygame.time.Clock()
        self.is_running = True
        self.map = RoadMap(SIM_WIDTH, SIM_HEIGHT)
        self.cars = [Car(i, self.map) for i in range(VEHICLE_COUNT)]
        for car in self.cars: car.set_new_destination()
        
        self.simulation_stats = {'predicted_collisions': set(), 'real_collisions': set()}
        self.game_over = False
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_crash_info = None
        self.crash_font = pygame.font.SysFont('Arial', 20, bold=True)


    def get_car_data_packet(self):
        data_packet = []
        for car in self.cars:
            if car.is_crashed: status = "CRASHED"
            elif car.is_finished: status = "FINISHED"
            else: status = "MOVING"
            gyro_rate = car.vel.angle_to(pygame.Vector2(1, 0)) - car.angle if car.vel.length() > 0 else 0
            car_dict = {
                'id': car.id, 'status': status, 'color': car.color,
                'waypoints': f"{car.current_waypoint_idx}/{len(car.path_waypoints)}",
                'pred_coords': f"({car.ekf.x[0][0]:.1f}, {car.ekf.x[1][0]:.1f})",
                'accel': math.hypot(car.ekf.x[4][0], car.ekf.x[5][0]),
                'gyro_rate': gyro_rate,
            }
            data_packet.append(car_dict)
        return data_packet

    def run(self):
        while self.is_running:
            dt = (self.clock.tick(FPS) / 1000.0) * SIMULATION_SPEED_FACTOR
            self._handle_events()
            if not self.game_over: self._update(dt)
            self._draw()
            data_packet = self.get_car_data_packet()
            self.socket.sendto(pickle.dumps(data_packet), (HOST, PORT))
        pygame.quit()

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.is_running = False

    def _update(self, dt):
        self._check_for_real_collisions()
        for car in self.cars:
            car.update(dt, self.cars, self.simulation_stats)
        if all(c.is_finished or c.is_crashed for c in self.cars):
            self.game_over = True

    def _check_for_real_collisions(self):
        for i in range(len(self.cars)):
            for j in range(i + 1, len(self.cars)):
                car1, car2 = self.cars[i], self.cars[j]
                if car1.is_crashed or car2.is_crashed: continue
                if car1.pos.distance_to(car2.pos) < COLLISION_RADIUS:
                    pair = frozenset([car1.id, car2.id])
                    if pair not in self.simulation_stats['real_collisions']:
                        self.simulation_stats['real_collisions'].add(pair)
                        car1.is_crashed, car2.is_crashed = True, True
                        self.last_crash_info = {'ids': (car1.id, car2.id), 'time': time.time()}
                        if pair in self.simulation_stats['predicted_collisions']:
                            print(f"SUCCESS: Collision between {car1.id} and {car2.id} was correctly predicted.")
                        else:
                            print(f"FAILURE: Collision between {car1.id} and {car2.id} was NOT predicted.")

    def _draw(self):
        self.screen.fill(BLACK)
        self.map.draw(self.screen)
        for car in self.cars:
            car.draw(self.screen)
        
        self._draw_crash_notification()

        if self.game_over:
            self._display_summary()
        pygame.display.flip()
        
    def _draw_crash_notification(self):
        if self.last_crash_info:
            if time.time() - self.last_crash_info['time'] > 5:
                self.last_crash_info = None
                return
                
            id1, id2 = self.last_crash_info['ids']
            text = f"CRASH DETECTED: Car {id1} and Car {id2}"
            
            bg_rect = pygame.Rect(0, SIM_HEIGHT - 40, SIM_WIDTH, 40)
            bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surface.fill((0, 0, 0, 180))
            self.screen.blit(bg_surface, bg_rect.topleft)

            text_surface = self.crash_font.render(text, True, RED)
            text_rect = text_surface.get_rect(center=bg_rect.center)
            self.screen.blit(text_surface, text_rect)

    def _display_summary(self):
        overlay = pygame.Surface((SIM_WIDTH, SIM_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        font_title = pygame.font.SysFont('Arial', 40, bold=True)
        font_text = pygame.font.SysFont('Arial', 24)
        title = font_title.render("Simulation Complete", True, WHITE)
        total_predicted = len(self.simulation_stats['predicted_collisions'])
        total_real = len(self.simulation_stats['real_collisions'])
        lines = [f"Total Predicted Collisions: {total_predicted}", f"Total Actual Collisions: {total_real}"]
        overlay.blit(title, title.get_rect(center=(SIM_WIDTH / 2, SIM_HEIGHT / 2 - 80)))
        for i, line in enumerate(lines):
            txt = font_text.render(line, True, WHITE if "Actual" not in line else RED)
            overlay.blit(txt, txt.get_rect(center=(SIM_WIDTH / 2, SIM_HEIGHT / 2 + i * 40)))
        self.screen.blit(overlay, (0, 0))

# ==============================================================================
# 6. SCRIPT EXECUTION
# ==============================================================================
if __name__ == '__main__':
    sim = SimulationManager()
    sim.run()