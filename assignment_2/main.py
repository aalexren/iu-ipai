#!/usr/bin/env python3
import sys, random
import numpy as np
from music21 import *

SETTINGS = {
    'VOLUME': 50,
    'CHORD_DURATION': 1.0,
    'OCTAVE_OFFSET': 4
}

'''
Firstly, we need read the midi file and convert it to music21.Stream object.

It will allow to duplicate object keeping the all headers settings from the input midi file.
Also, it will make possible to change chords sequence not affecting the other populations.
'''
def read_midi(path: str) -> stream.Stream:
    print(f'Reading midi file {path}...')
    midi_file = converter.parse(path)
    print('Done!')
    return midi_file

'''
Returns tone of the melody in midi file.
'''
def get_midi_tone(midi_file: stream.Stream) -> tuple[str]:
    key = midi_file.analyze('key')
    return (key.tonic.name, key.mode)

'''
Set volume for the accompaniment.
Count average between the notes and takes away some points.
'''
def set_settings(midi: stream.Stream, vadjust: int):
    SETTINGS['VOLUME'] = np.average(np.array([_.volume.velocity for _ in midi.recurse() if type(_) is note.Note])) - vadjust

'''
Simply make random chord in the range of two octaves.
Duration is in ratios (в долях), e.g. quarter, whole etc.
'''
def make_random_chord(volume_: int, octave_: int, duration_: float) -> chord.Chord:
    c = chord.Chord()
    
    notes = random.sample(range(0,23), 3)
    n1 = note.Note(notes.pop() + (octave_ * 12))
    n2 = note.Note(notes.pop() + (octave_ * 12))
    n3 = note.Note(notes.pop() + (octave_ * 12))
    n1.volume.velocity = volume_
    n2.volume.velocity = volume_
    n3.volume.velocity = volume_

    c.duration = duration.Duration(duration_)
    c.add([n1,n2,n3])

    return c

'''
Generate the chord sequence using the random chords generator.
'''
def make_chord_seq(count_: int) -> list[chord.Chord]:
    ret: list[chord.Chord] = []
    for _ in range(0, count_):
        ret.append(make_random_chord(SETTINGS['VOLUME'], SETTINGS['OCTAVE_OFFSET'], SETTINGS['CHORD_DURATION']))
    return ret

'''
Make population melody + chords.
'''
def make_chord_track(chords: list[chord.Chord]) -> stream.Stream:
    cstream = stream.Stream()
    cstream.append(instrument.Piano())
    for _ in chords:
        cstream.append(_)
    return cstream

'''
If needs to determine length of the midi,
i.e. to understand how many bars (тактов) in the song.
This information allows to count the chords number.
'''
def get_bars_count(song: stream.Stream) -> int:
    return int(np.sum(np.array([1 for _ in song.recurse() if type(_) is stream.Measure])))


def chord_fitness(song: stream.Stream, chord: chord.Chord) -> float:
    

def fitness() -> float:
    pass


def save_midi(song: stream.Stream, fname='output.mid'):
    song.write('midi', fp=fname)

def main(infile: str, outfile: str):
    mstream = read_midi(infile)
    set_settings(mstream, vadjust=5)
    bars = get_bars_count(mstream)
    track = make_chord_track(make_chord_seq(bars * int(4 / SETTINGS['CHORD_DURATION'])))
    mstream.append(track)
    save_midi(mstream, outfile)

if __name__ == '__main__':
    # Check if we are inside the virtualenv
    # import os
    # print(os.environ['VIRTUAL_ENV'])
    main(sys.argv[1], sys.argv[2])
