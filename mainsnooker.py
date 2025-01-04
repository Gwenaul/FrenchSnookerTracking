from balltracker import BallTracker  # Mettre la classe dans un fichier balltracker.py
from imutils.video import VideoStream
import argparse
import cv2
import time
import mido
from mido import Message, MidiFile, MidiTrack

outport = mido.open_output('IAC Driver Bus 1')
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

yellowLower = (0, 116, 77)
yellowUpper = (23, 255, 255)
redLower = (164, 79, 81)
redUpper = (179, 255, 255)
whiteLower = (0, 0, 95)
whiteUpper = (26, 167, 255)

if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])
time.sleep(2.0)

frame_width = int(vs.get(3))
frame_height = int(vs.get(4))
frame_size = (frame_width, frame_height)
fps = 30.0

output_file = "SnookSnook.avi"
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

Battements = 0
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)
track.append(Message('note_on', note=1, velocity=100, time=0))

# Initialisation des trackers après les définitions de couleurs
yellow_tracker = BallTracker("yellow", yellowLower, yellowUpper, 0, outport)
red_tracker = BallTracker("red", redLower, redUpper, 1, outport)
white_tracker = BallTracker("white", whiteLower, whiteUpper, 2, outport, track)

# Remplacer tout le code de tracking des billes dans la boucle while par:
while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    if frame is None:
        break

    frame = cv2.resize(frame, frame_size)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    Battements = yellow_tracker.process_frame(frame, hsv, Battements)
    Battements = red_tracker.process_frame(frame, hsv, Battements)
    Battements = white_tracker.process_frame(frame, hsv, Battements)

    cv2.imshow("Frame", frame)
    out.write(frame)
    mid.save('MIDO_Write-Midi-File.mid')

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        allnotes = 1
        while allnotes < 128:
            outport.send(mido.Message('note_off', note=allnotes, velocity=64, channel=0))
            outport.send(mido.Message('note_off', note=allnotes, velocity=64, channel=1))
            outport.send(mido.Message('note_off', note=allnotes, velocity=64, channel=2))
            allnotes += 1
        break

if not args.get("video", False):
    vs.stop()
else:
    vs.release()
cv2.destroyAllWindows()