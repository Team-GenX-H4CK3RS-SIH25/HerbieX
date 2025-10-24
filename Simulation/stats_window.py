# stats_window.py (Corrected and More Robust)

import pygame
import math
import socket
import pickle

# --- Constants ---
STATS_WIDTH, STATS_HEIGHT = 900, 400
STATS_BG = (10, 10, 25)
WHITE = (255, 255, 255)
RED = (255, 50, 50)
GREEN_STATUS = (50, 200, 50)
YELLOW_STATUS = (200, 200, 50)
TEXT_COLOR = (200, 200, 255)
CRASHED_COLOR = (100, 20, 20, 200)

# --- Network Settings ---
HOST = '127.0.0.1'  # Localhost
PORT = 65432        # Port to listen on (must match the server)

class StatsClient:
    """
    A standalone Pygame application that receives car data over a socket
    and displays it in a scrollable interface.
    """
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((STATS_WIDTH, STATS_HEIGHT))
        pygame.display.set_caption("Live Vehicle Stats (Client - Waiting for Server)")

        self.font = pygame.font.SysFont('Consolas', 12)
        self.font_header = pygame.font.SysFont('Consolas', 14, bold=True)

        self.scroll_y = 0
        self.max_scroll = 0
        self.is_running = True
        self.car_data = [] # This will hold the latest data received

        # --- Setup Socket with a Timeout ---
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((HOST, PORT))
        # FIX: Use a timeout instead of non-blocking for better stability.
        # This tells the socket to only wait 10ms for data before giving up.
        self.socket.settimeout(0.01)

    def receive_data(self):
        """Try to receive data from the socket without getting stuck."""
        try:
            # Continuously read to clear any backlog, ensuring we get the latest packet
            while True:
                data, _ = self.socket.recvfrom(4096) # Buffer size
                self.car_data = pickle.loads(data)
        except socket.timeout:
            # This is now the expected "error" when no data is available. It's safe to ignore.
            pass
        except Exception as e:
            print(f"Error receiving data: {e}")

    def handle_events(self):
        """Handles user input like quitting or scrolling."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            if event.type == pygame.MOUSEWHEEL:
                self.scroll_y -= event.y * 30
                self.scroll_y = max(0, min(self.scroll_y, self.max_scroll))

    def draw(self):
        """Renders the stat cards based on the latest self.car_data."""
        self.screen.fill(STATS_BG)
        if not self.car_data:
            waiting_text = self.font_header.render("Waiting for simulation data...", True, WHITE)
            self.screen.blit(waiting_text, (STATS_WIDTH // 2 - 150, STATS_HEIGHT // 2))
            pygame.display.flip()
            return

        card_w, card_h = 290, 125
        margin = 10
        cols = STATS_WIDTH // (card_w + margin)
        
        num_rows = (len(self.car_data) + cols - 1) // cols
        total_content_height = num_rows * (card_h + margin)
        self.max_scroll = max(0, total_content_height - STATS_HEIGHT)

        for i, car in enumerate(self.car_data):
            row = i // cols
            col = i % cols
            card_x = col * (card_w + margin) + margin
            card_y = row * (card_h + margin) + margin - self.scroll_y

            if card_y + card_h < 0 or card_y > STATS_HEIGHT:
                continue

            card_bg_color = CRASHED_COLOR if car['status'] == 'CRASHED' else car['color'] + [100]
            card_surface = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
            card_surface.fill(card_bg_color)
            
            status, status_color = car['status'], {'MOVING': YELLOW_STATUS, 'FINISHED': GREEN_STATUS, 'CRASHED': RED}.get(car['status'])

            header = self.font_header.render(f"CAR ID: {car['id']}", True, WHITE)
            line1 = self.font.render(f"Status: {status}", True, status_color)
            line2 = self.font.render(f"Waypoints: {car['waypoints']}", True, TEXT_COLOR)
            line3 = self.font.render(f"Pred Coords: {car['pred_coords']}", True, TEXT_COLOR)
            line4 = self.font.render(f"Accel (EKF): {car['accel']:.1f} px/s²", True, TEXT_COLOR)
            line5 = self.font.render(f"Gyro Rate:   {car['gyro_rate']:.2f} rad/s", True, TEXT_COLOR)

            y_pos = 5
            card_surface.blit(header, (10, y_pos)); y_pos += 20
            card_surface.blit(line1, (10, y_pos)); y_pos += 18
            card_surface.blit(line2, (10, y_pos)); y_pos += 18
            card_surface.blit(line3, (10, y_pos)); y_pos += 18
            card_surface.blit(line4, (10, y_pos)); y_pos += 18
            card_surface.blit(line5, (10, y_pos))
            
            self.screen.blit(card_surface, (card_x, card_y))

        pygame.display.flip()

    def run(self):
        """The main loop for the stats client window."""
        clock = pygame.time.Clock()
        print("Stats client is running. Waiting for data from main_simulation.py...")
        while self.is_running:
            self.handle_events()
            self.receive_data()
            self.draw()
            clock.tick(60) # Run at a responsive frame rate
        pygame.quit()
        print("Stats client has been closed.")


if __name__ == '__main__':
    stats_client = StatsClient()
    stats_client.run()