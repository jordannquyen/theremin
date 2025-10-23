# theremin/.venv/bin/python virtual_theremin.py
import cv2 # video
import mediapipe as mp # pre trained hand tracking w/skeleton
import numpy as np # math
import pygame # audio, math -> sound
import sys # python system

# sound
pygame.mixer.init(44100, -16, 1, 2048) # sample rate / bit / channels / buffer size
pygame.init()

# hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# camera
cap = cv2.VideoCapture(1)  # external camera
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
max_attempts = 3
attempt = 0
while not cap.isOpened() and attempt < max_attempts:
    cap.release()
    cap = cv2.VideoCapture(0)
    attempt += 1
    cv2.waitKey(1000)  # = 1 second

if not cap.isOpened():
    print("error, could not open camera. check camera permissions")
    sys.exit(1)

# camera size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

MIN_FREQ = 261.63  # C4 (middle c)
MAX_FREQ = 523.25  # C5
MAX_VOLUME = 0.5
BASELINE_Y = height - 100 

NOTES = [
    ("C5", 523.25),
    ("B4", 493.88),
    ("A#4/Bb4", 466.16),
    ("A4", 440.00),
    ("G#4/Ab4", 415.30),
    ("G4", 392.00),
    ("F#4/Gb4", 369.99),
    ("F4", 349.23),
    ("E4", 329.63),
    ("D#4/Eb4", 311.13),
    ("D4", 293.66),
    ("C#4/Db4", 277.18),
    ("C4", 261.63)
]

def get_note_position(freq):
    return int(BASELINE_Y - (BASELINE_Y * (np.log(freq/MIN_FREQ) / np.log(MAX_FREQ/MIN_FREQ))))

# sine wave for theremin sound
sample_rate = 44100
duration = 0.2
fade_duration = 0.05
t = np.linspace(0, duration, int(sample_rate * duration), False)
fade_samples = int(fade_duration * sample_rate)
fade_in = np.linspace(0, 1, fade_samples)
fade_out = np.linspace(1, 0, fade_samples)

def generate_tone(frequency, volume):
    tone = np.sin(2 * np.pi * frequency * t)
    tone[:fade_samples] *= fade_in
    tone[-fade_samples:] *= fade_out
    tone = (volume * tone * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(tone)

current_sound = None
current_freq = MIN_FREQ
current_volume = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # baseline
        cv2.line(frame, (0, BASELINE_Y), (width, BASELINE_Y), (0, 255, 0), 2)
        
        # note indicators
        for note_name, freq in NOTES:
            y_pos = get_note_position(freq)
            cv2.line(frame, (width//2, y_pos), (width, y_pos), (200, 200, 200), 1)
            cv2.putText(frame, note_name, (width - 120, y_pos - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
        if results.multi_hand_landmarks:
            freq_hand = None
            vol_hand = None
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                index_finger_y = hand_landmarks.landmark[8].y * height #index finger
                hand_x = hand_landmarks.landmark[8].x * width
                
                # left =volume, right = pitch
                if hand_x < width/2:
                    vol_hand = index_finger_y
                else:
                    freq_hand = index_finger_y
            
            if freq_hand is not None:
                freq_ratio = np.clip((BASELINE_Y - freq_hand) / BASELINE_Y, 0, 1)
                current_freq = MIN_FREQ * np.exp(np.log(MAX_FREQ/MIN_FREQ) * freq_ratio)
            
            if vol_hand is not None:
                current_volume = np.clip((BASELINE_Y - vol_hand) / BASELINE_Y, 0, 1) * MAX_VOLUME
            
            new_sound = generate_tone(current_freq, current_volume)
            new_sound.play(-1)
            if current_sound is not None:
                current_sound.fadeout(50)  # 50ms fadeout
            current_sound = new_sound
            
            cv2.putText(frame, f'Frequency: {current_freq:.1f} Hz', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f'Volume: {current_volume:.2f}', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            if current_sound is not None:
                current_sound.stop()
                current_volume = 0

        cv2.imshow('Virtual Theremin', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    if current_sound is not None:
        current_sound.stop()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()