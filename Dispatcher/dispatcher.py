import pygame
import time
from pylsl import StreamInfo, StreamOutlet

# Define stream information: Name, Type, Number of channels, Rate, Format, Unique ID
info = StreamInfo('dispatcherStream', 'Markers', 1, 0, 'string', 'dispatcher1234')
outlet = StreamOutlet(info)

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)  # Set the screen to be resizable
pygame.display.set_caption("Blinking Text")
running = True
fullscreen = False

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Set up font
font = pygame.font.Font(None, 250)  # Large font size

# Blinking variable
show_text = False

start = False

timeline = []

# Phase 1: 2 minutes rest (120 seconds)
timeline += [6] * 120  # 6 = Resting Phase

# Phase 2: 120 blinks (1 blink every 3 seconds)
timeline.append(7)  # 7 = Blinking Phase
for _ in range(120):
    timeline.append(1)  # 1 = Blink
    timeline += [0, 0]  # 2 seconds of no blink

# Phase 3: 2 minutes rest (120 seconds)
timeline += [6] * 120 

# Phase 4: 120 blinks with head movement
timeline.append(5)  # 5 = Head Movement Phase
for _ in range(120):
    # first move head left for 40 blinks. Then move head right for 40 blinks. and up for 40 blinks.
    if _ == 0:
        timeline.append(2)
    elif _ == 40:
        timeline.append(3)
    # 2 = Move head left (example)
    elif _ == 80:
        timeline.append(4)  # 4 = Move head up (example)
    else:
        timeline.append(1)
        timeline += [0, 0]  # 2 seconds of no movement

# Phase 6: Repeat everything again for Lighting Phase
timeline.append(8)  # 8 = Lighting Phase

timeline += timeline.copy()

timeline += [9] * 10  # 9 = End of Experiment

i = 0

timeline_length = len(timeline)

dicto = {
    0: "NO BLINK", 
    1: "BLINK",
    2: "Move head left",
    3: "Move head right",
    4: "Move head up",

    5: "Head Movement Phase",
    6: "Resting Phase",
    7: "Blinking Phase",
    8: "Lighting Phase",
    9: "End of Experiment"
        }

start_time = time.time()

def send_lsl(sample):
    outlet.push_sample([sample])

def draw_text(txt):
    text_surface = font.render(txt, True, WHITE)
    text_rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text_surface, text_rect)

# Main game loop
while running:
    print(i)
    for event in pygame.event.get():
        if event.type == pygame.QUIT or pygame.key.get_pressed()[pygame.K_ESCAPE]:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                start = True
        if event.type == pygame.VIDEORESIZE:
            # Resize the screen if the window is resized
            screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            info = pygame.display.Info()
            WIDTH, HEIGHT = info.current_w, info.current_h
            
    screen.fill((BLACK))  # Clear the screen with black

    if start:
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= 1:
            start_time = current_time
            i += 1
            if i >= len(timeline):
                i = 0
        
            if timeline[i] != 0:
                show_text = True
                text = dicto[timeline[i]]
            else:
                show_text = False
                text = ""
            send_lsl(text) 
        # Render text if show_text is True
        if show_text:
            draw_text(text)

    pygame.display.flip()  # Update the display

pygame.quit()



