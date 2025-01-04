from collections import deque
import numpy as np
import cv2
import imutils
import mido
from mido import Message, MidiFile, MidiTrack

class BallTracker:
    def __init__(self, color_name, lower_color, upper_color, channel, outport, track=None):
        self.color_name = color_name
        self.lower_color = lower_color
        self.upper_color = upper_color
        self.channel = channel
        self.outport = outport
        self.track = track

        self.pts = deque(maxlen=128)
        self.positions_to_measure = []
        self.points = []
        self.pointsflute = []
        self.stopping = []

        self.previous_angle = None
        self.last_position = None
        self.last_note = None
        self.noteflute_slow = None
        self.center_slow = None
        self.frame_count = 0
        self.compteur = 0
        self.cm = 0
        self.is_moving = False
        self.is_stopping = False
        self.slow = False
        self.debut_slow = None
        self.x1, self.y1 = 0, 0
        self.x2, self.y2 = 0, 0

        self.DISTANCE_THRESHOLD = 1
        self.INTERVAL = 4

    def process_frame(self, frame, hsv, battements):
        mask = cv2.inRange(hsv, self.lower_color, self.upper_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # Affichage
            x, y = center
            cv2.putText(frame, f"({self.cm}cm)", (self.x2, self.y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._get_color(), 1)
            cv2.line(frame, (self.x1, self.y1), (self.x2, self.y2), self._get_color(), 2)

            self._process_movement(center, battements)

        return battements

    def _process_movement(self, center, battements):
        if center is None:
            return battements

        distance = self._calculate_distance(center)

        if distance > self.DISTANCE_THRESHOLD:
            battements = self._handle_moving(center, distance, battements)

        if self.is_moving:
            battements = self._handle_stopping(center, distance, battements)

        return battements

    def _calculate_distance(self, center):
        if self.last_position is None:
            return self.DISTANCE_THRESHOLD + 1
        return np.sqrt((center[0] - self.last_position[0]) ** 2 +
                       (center[1] - self.last_position[1]) ** 2)

    def _handle_moving(self, center, distance, battements):
        self.points.append(tuple(center))
        self.positions_to_measure.append(center)
        self.last_position = center

        if self.frame_count >= self.INTERVAL:
            battements = self._process_positions(center, battements)
            self.frame_count = 0

        self.frame_count += 1
        return battements

    def _process_positions(self, center, battements):
        positions = self.positions_to_measure.copy()
        self.positions_to_measure.clear()

        if len(positions) >= 2:
            max_distance, max_points = self._find_max_distance_points(positions)

            if max_points is not None:
                battements = self._process_angle_change(center, max_points, battements)

        return battements

    def _find_max_distance_points(self, positions):
        max_distance = 0
        max_points = None

        for i in range(len(positions) - 1):
            for j in range(i + 1, len(positions)):
                distance = np.sqrt((positions[i][0] - positions[j][0]) ** 2 +
                                   (positions[i][1] - positions[j][1]) ** 2)
                if distance > max_distance:
                    max_distance = distance
                    max_points = (positions[i], positions[j])

        return max_distance, max_points

    def _process_angle_change(self, center, max_points, battements):
        angle = np.arctan2(max_points[1][1] - max_points[0][1],
                           max_points[1][0] - max_points[0][0])

        if self.previous_angle is not None and abs(angle - self.previous_angle) >= 0.3:
            battements = self._handle_angle_change(center, battements)

        self.previous_angle = angle
        return battements

    def _handle_angle_change(self, center, battements):
        if self.noteflute_slow is not None:
            self._send_note_off(self.noteflute_slow, battements)

        self.pointsflute.append(tuple(center))
        self._update_flute_points()

        for i in range(len(self.pointsflute) - 1):
            battements = self._calculate_and_play_note(i, center, battements)

        return battements

    def _update_flute_points(self):
        self.compteur += 1
        if self.compteur > 2:
            self.pointsflute.pop(0)
            self.compteur -= 1

    def _calculate_and_play_note(self, i, center, battements):
        distance = np.sqrt((self.pointsflute[i][0] - self.pointsflute[i + 1][0]) ** 2 +
                           (self.pointsflute[i][1] - self.pointsflute[i + 1][1]) ** 2)

        self.x1, self.y1 = self.pointsflute[i]
        self.x2, self.y2 = self.pointsflute[i + 1]

        midi_value = 69 - (10 / 75 * distance) + 24
        self.cm = int(distance / (111 / 47))
        note = int(midi_value)

        if self.cm >= 3:
            self._play_note(note, battements)

        self.is_moving = True
        self.slow = True
        self.center_slow = center

        return battements

    def _play_note(self, note, battements):
        """
        Joue une note MIDI et gère les notes précédentes si nécessaire.
        """
        if self.last_note is not None:
            self._send_note_off(self.last_note, battements)

        self._send_note_on(note)
        self.last_note = note

        if self.track:
            # Ajout d'un événement note_on dans la piste MIDI
            self.track.append(Message('note_on', note=note, velocity=64, time=0))

    def _handle_stopping(self, center, distance, battements):
        if 0 <= distance < 1:
            if self.slow:
                self.debut_slow = self.center_slow
                self.slow = False
            self.stopping.append(center)

        if len(self.stopping) >= 3 and center == self.stopping[-3]:
            battements = self._process_stopping(center, battements)

        return battements

    def _process_stopping(self, center, battements):
        self.stopping = []
        self.x1, self.y1 = self.debut_slow
        self.x2, self.y2 = center

        distance = ((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2) ** 0.5
        midi_value = 69 - (10 / 75 * distance) + 24
        self.cm = int(distance / (111 / 47))
        note = int(midi_value)

        if self.cm >= 3:
            if self.last_note is not None:
                self._send_note_off(self.last_note, battements)
            self.pointsflute = []
            self.compteur = 0
            self._send_note_on(note)
            if self.track:
                self.track.append(Message('note_on', note=note, velocity=64, time=0))

        self.is_moving = False
        return battements

    def _send_note_on(self, note):
        self.outport.send(mido.Message('note_on', note=note, velocity=64, channel=self.channel))

    def _send_note_off(self, note, battements):
        self.outport.send(mido.Message('note_off', note=note, velocity=64, channel=self.channel))
        if self.track:
            self.track.append(Message('note_off', note=note, velocity=64, time=battements))

    def _get_color(self):
        if self.color_name == "yellow":
            return (0, 121, 175)
        elif self.color_name == "red":
            return (0, 0, 255)
        else:  # white
            return (255, 255, 255)