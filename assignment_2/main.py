#!/usr/bin/env python3
import sys, random
import numpy as np
import copy
from music21 import *

SETTINGS = {
    'VOLUME': 50,
    'CHORD_DURATION': 1.0,
    'OCTAVE_OFFSET': 3,
    'TONE': 'C'
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
def get_midi_tone(midi_file: stream.Stream) -> key.Key:
    key_ = midi_file.analyze('key')
    return key_

'''
Set volume for the accompaniment.
Count average between the notes and takes away some points.
'''
def set_settings(midi: stream.Stream, vadjust: int):
    SETTINGS['VOLUME'] = np.average(np.array([_.volume.velocity for _ in midi.recurse() if type(_) is note.Note])) - vadjust
    SETTINGS['TONE'] = get_midi_tone(midi)

'''
Simply make random chord in the range of two octaves.
Duration is in ratios (в долях), e.g. quarter, whole etc.
'''
def make_random_chord(volume_: int, octave_: int, duration_: float) -> chord.Chord:
    c = chord.Chord()
    
    notes = random.sample(range(0,24), 3)
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


def fitness(notes, chord_: chord.Chord) -> float:
    tone_notes = SETTINGS['TONE'].pitches # extract all (ступени)
    # print(notes, chord_, tone_notes, SETTINGS['TONE'].mode)

    f = 0 # sum of functions
    k = 0 # coefficient for functions

    # c = chord.Chord('C#3 E3 G#3')

    if SETTINGS['TONE'].mode == 'minor':
        T35 = chord.Chord([tone_notes[0], tone_notes[0].midi + 3, tone_notes[0].midi + 7])
        T6 = chord.Chord([tone_notes[0].midi + 3, tone_notes[0].midi + 7, tone_notes[0].midi + 12])
        T64 = chord.Chord([tone_notes[0].midi + 7, tone_notes[0].midi + 12, tone_notes[0].midi + 15])

        S35 = chord.Chord([tone_notes[3], tone_notes[3].midi + 3, tone_notes[3].midi + 7])
        S6 = chord.Chord([tone_notes[3].midi + 3, tone_notes[3].midi + 7, tone_notes[3].midi + 12])
        S64 = chord.Chord([tone_notes[3].midi + 7, tone_notes[3].midi + 12, tone_notes[3].midi + 15])

        D35 = chord.Chord([tone_notes[4], tone_notes[4].midi + 3, tone_notes[4].midi + 7])
        D6 = chord.Chord([tone_notes[4].midi + 3, tone_notes[4].midi + 7, tone_notes[4].midi + 12])
        D64 = chord.Chord([tone_notes[4].midi + 7, tone_notes[4].midi + 12, tone_notes[4].midi + 15])
        # print("keks", T35, T6, T64, S35, S6, S64, D35, D6, D64)

    if SETTINGS['TONE'].mode == 'major':
        T35 = chord.Chord([tone_notes[0], tone_notes[0].midi + 4, tone_notes[0].midi + 7])
        T6 = chord.Chord([tone_notes[0].midi + 4, tone_notes[0].midi + 7, tone_notes[0].midi + 12])
        T64 = chord.Chord([tone_notes[0].midi + 7, tone_notes[0].midi + 12, tone_notes[0].midi + 16])

        S35 = chord.Chord([tone_notes[3], tone_notes[3].midi + 4, tone_notes[3].midi + 7])
        S6 = chord.Chord([tone_notes[3].midi + 4, tone_notes[3].midi + 7, tone_notes[3].midi + 12])
        S64 = chord.Chord([tone_notes[3].midi + 7, tone_notes[3].midi + 12, tone_notes[3].midi + 16])

        D35 = chord.Chord([tone_notes[4], tone_notes[4].midi + 4, tone_notes[4].midi + 7])
        D6 = chord.Chord([tone_notes[4].midi + 4, tone_notes[4].midi + 7, tone_notes[4].midi + 12])
        D64 = chord.Chord([tone_notes[4].midi + 7, tone_notes[4].midi + 12, tone_notes[4].midi + 16])

    pf = chord_.primeForm
    # print(type(pf))
    if len(set(pf).intersection(T35.primeForm)) > 0:
        if chord_.primeForm == T35.primeForm:
            k = 5
            f = k**3
        elif len(set(pf).intersection(T35.primeForm)) == 2:
            k = 3
            f = k**3
        else:
            k = 2
            f = k**3
    if len(set(pf).intersection(T6.primeForm)) > 0 or len(set(pf).intersection(T64.primeForm)) > 0:
        if pf == T6.primeForm or pf == T64.primeForm:
            k = 4
            f = k**3
        elif len(set(pf).intersection(T6.primeForm)) == 2 or len(set(pf).intersection(T64.primeForm)) == 2:
            k = 3
            f = k**3
        else:
            k = 2
            f = k**3
    if len(set(pf).intersection(S35.primeForm)) > 0 or len(set(pf).intersection(D35.primeForm)) > 0:
        if pf == S35.primeForm or D35.primeForm:
            k = 5
            f = k**2
        elif len(set(pf).intersection(S35.primeForm)) == 2 or len(set(pf).intersection(D35.primeForm)) == 2:
            k = 4
            f = k**2
        else:
            k = 3
            f = k**2
    if len(set(pf).intersection(S6.primeForm)) > 0 or len(set(pf).intersection(D6.primeForm)) > 0 or len(set(pf).intersection(S64.primeForm)) > 0 or len(set(pf).intersection(D64.primeForm)) > 0:
        if pf == S6 or pf == D6 or pf == S64 or pf == D64:
            k = 4
            f = k**2
        elif len(set(pf).intersection(S6.primeForm)) == 2 or len(set(pf).intersection(D6.primeForm)) == 2 or len(set(pf).intersection(S64.primeForm)) == 2 or len(set(pf).intersection(D64.primeForm)) == 2:
            k = 3
            f = k**2
        else:
            k = 2
            f = k**2
    else:
        f -= 100

    # print(f)
    return f

def population_fitness(song_: stream.Stream, chords: list[chord.Chord]) -> float:
    dnotes = np.array([n for n in song_.recurse() if type(n) in [note.Note, note.Rest]])
    dntime = np.zeros(len(dnotes) + 1)
    for i in range(1, len(dntime)):
        dntime[i] = dntime[i-1] + dnotes[i-1].duration.quarterLength
    dntime = dntime[1:]

    dchords = chords
    dctime = np.zeros(len(dchords) + 1)
    for i in range(1, len(dctime)):
        dctime[i] = dctime[i-1] + dchords[i-1].duration.quarterLength
    dctime = dctime[1:]

    # print(dctime[np.where((12 < dctime[:]) & (20 > dctime[:]))])
    # print(dntime, dntime.size, len(dnotes))
    # print(dctime, dctime.size, len(dchords))

    fit = 0
    lhs = 0
    for i in range(0, len(dctime)):
        # print(next(x[0] for x in enumerate(dntime) if x[1] > dctime[i]))
        rhs = np.argmax(dntime > dctime[i])
        m_for_c = dnotes[lhs:rhs] # notes for chord
        fit += fitness(m_for_c, dchords[i])
        if type(dchords[i]) == type(list):
            print(dchords[i])
            break
        lhs = rhs
        # break

    return fit

'''
Create chord sequence.
Returns list of Chords.
'''
def get_population(mstream: stream.Stream) -> list[chord.Chord]:
    bars = get_bars_count(mstream)
    chord_seq = make_chord_seq(bars * int(4 / SETTINGS['CHORD_DURATION']))

    return chord_seq

def crossover(mstream, populations):
    # l = len(populations) // 2
    for i, j in zip(populations[::2], populations[1::2]):
        p_len = len(i)
        a, b = i[1][:p_len] + j[1][p_len:], j[1][:p_len] + i[1][p_len:]
        song = copy.deepcopy(mstream)
        song.append(make_chord_track(a))
        populations.append((population_fitness(song, a), a))
        song = copy.deepcopy(mstream)
        song.append(make_chord_track(b))
        populations.append((population_fitness(song, b), b))



def mutation():
    pass

def evolution(mstream: stream.Stream, generations: int):
    pops = []
    for gen in range(generations):
        for i in range(100):
            song = copy.deepcopy(mstream)
            population = get_population(mstream)
            song.append(make_chord_track(population))
            pops.append((population_fitness(song, population), population))
        pops.sort(key = lambda x: x[0], reverse = True)
        crossover(mstream, pops)
        pops.sort(key = lambda x: x[0], reverse = True)
        pops = pops[:len(pops)]
        print(f'epoch {gen}, population size is {len(pops)}')
    # print(pops)
    return pops[0]


def save_midi(song: stream.Stream, fname='output.mid'):
    song.write('midi', fp=fname)

def main(infile: str, outfile: str):
    mstream = read_midi(infile)
    set_settings(mstream, vadjust=5)
    # bars = get_bars_count(mstream)
    # chord_seq = make_chord_seq(bars * int(4 / SETTINGS['CHORD_DURATION']))
    # cc = get_population(mstream)

    output = evolution(mstream, 100)
    track = make_chord_track(output[1])
    mstream.append(track)

    # fitness(mstream, cc)
    save_midi(mstream, outfile)

if __name__ == '__main__':
    # Check if we are inside the virtualenv
    # import os
    # print(os.environ['VIRTUAL_ENV'])
    main(sys.argv[1], sys.argv[2])
    # main('barbiegirl_mono.mid', 'kek.mid')
