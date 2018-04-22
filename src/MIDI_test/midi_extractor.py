import midi
import numpy as np

TRACK_NUMBER = 16
MAX_DATA_PER_EVENT = 18 # [statusmsg, metacommand, tick data[0],...,data[13]]
# This max is arbitrary, we noticed the events with max data
# are InstrumentNameEvent. They are rarely over 12 datas in them

pattern = midi.read_midifile("1492.mid")

END_TRACK = [0]*TRACK_NUMBER # Index of end for each track
MAX_TRACK_SIZE = 0

# Determine input size, maybe more effective way to do
for track_i in range(len(pattern)):
    for e in pattern[track_i]:
        if type(e) == midi.events.EndOfTrackEvent:
            END_TRACK[track_i]
            if e.tick > MAX_TRACK_SIZE:
                MAX_TRACK_SIZE = e.tick

input = np.zeros(TRACK_NUMBER, DATA_PER_EVENT, MAX_TRACK_SIZE)

for track_i in range(TRACK_NUMBER):
    for event_i in range(MAX_TRACK_SIZE):
        input[track_i][event_i][0] = pattern[track_i][event_i]
        input[track_i][event_i][1]
        min_data = min(len(pattern[track_i][event_i].data), MAX_TRACK_SIZE-2)
        for data_i in range(min_data):
            input[track_i][event_i][data_i + 2] = pattern[track_i][event_i].data[data_i] # 2 is statusmsg + metacommand size
        input[track_i][event_i]
