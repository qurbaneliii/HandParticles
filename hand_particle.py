import cv2
import mediapipe as mp
import numpy as np
class ParticleSystem:
    def __init__(self, num_particles=600):
        self.num_particles = num_particles
        self.position = None
        self.velocities = None
        self.initialized = False

    def init_particles(self, width, height):
        self.position = np.column_stack((np.random.rand(self.num_particles) * width,
                                         np.random.rand(self.num_particles) * height)).astype(np.float32)

        angles = np.random.rand(self.num_particles) * 1.3 * np.pi
        speeds = np.random.rand(self.num_particles) * 3.0 + 1
        self.velocities = np.column_stack((np.cos(angles) * speeds, np.sin(angles) * speeds)).astype(np.float32)
        self.initialized = True

    def update(self, width, height, center, mode):
        if not self.initialized:
            self.init_particles(width, height)

        cx, cy = center
        center_vec = np.array([cx, cy], dtype=np.float32)
        to_center = center_vec - self.position
        distances = np.linalg.norm(to_center, axis=1, keepdims=True) + 1e-6
        dir_to_center = to_center / distances
        dir_from_center = -dir_to_center

        if mode == "gather":
            self.velocities = self.velocities + dir_to_center
        elif mode == "explode":
            randomness = (np.random.rand(self.num_particles, 2).astype(np.float32)) * 0.5
            self.velocities = self.velocities * 0.9 + dir_from_center * 1.2 + randomness
        else:
            self.velocities *= 0.98

        self.velocities *= 0.98
        self.position += self.velocities

        x = self.position[:, 0]
        y = self.position[:, 1]

        mask_left = x < 0
        mask_right = x > width
        self.velocities[mask_left | mask_right, 0] *= -1
        x[mask_left] = 0
        x[mask_right] = width

        mask_top = y < 0
        mask_bottom = y > height
        self.velocities[mask_top | mask_bottom, 1] *= -1
        y[mask_top] = 0
        y[mask_bottom] = height

        self.position[:, 0] = x
        self.position[:, 1] = y

    def draw(self, frame):
        pts = self.position.astype(np.int32)
        for (px, py) in pts:
            cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def classify_hand_state(landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    exit_count = 0

    for tip, pip in zip(finger_tips, finger_pips):
        if landmarks[tip].y < landmarks[pip].y:
            exit_count += 1

    return 'open' if exit_count >= 3 else 'closed'

def get_hand_center(landmarks, width, height):
    x5= [ln.x * width for ln in landmarks]
    y5= [ln.y * height for ln in landmarks]
    return int(sum(x5) / len(x5)), int(sum(y5) / len(y5))

def main():
    cap = cv2.VideoCapture(0)
    particle_system = ParticleSystem(num_particles=400)
    current_mode = "neutral"
    hand_center = (320, 240)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_state = "none"

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                hand_center = get_hand_center(hand_landmarks.landmark, w, h)
                hand_state = classify_hand_state(hand_landmarks.landmark)

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            if hand_state == "closed":
                current_mode = "gather"
                cv2.putText(frame, "CLOSED - Gather", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            elif hand_state == "open":
                current_mode = "explode"
                cv2.putText(frame, "OPEN - Explode", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                current_mode = "neutral"
                cv2.putText(frame, "No Hand", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

            cv2.circle(frame, hand_center, 8, (255, 0, 0), -1)
            particle_system.update(w, h, hand_center, current_mode)
            particle_system.draw(frame)

            cv2.imshow("Hand Particles", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()