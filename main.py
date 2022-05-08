#!/usr/bin/env python3
import sys, random, math, time
import numpy as np
from music21 import *

SETTINGS = {
    'VOLUME': 50,
    'CHORD_DURATION': 1.0,
    'OCTAVE_OFFSET': 3,
    'TONE': 'C',
    'NOTES': [],
    'BARS_COUNT': 0,
    'KEY_CHORDS': [],
    'TONE_CHORDS': [],
    'TONE_NOTES': []
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
    SETTINGS['NOTES'] = get_notes(midi)
    SETTINGS['BARS_COUNT'] = get_bars_count(midi)
    SETTINGS['KEY_CHORDS'] = fitness_help() # define key chords for the fitness function
    SETTINGS['OCTAVE_OFFSET'] = get_octave_offset(midi) - 1

'''
Define octave offset for the chords.
Find the minimal octave between notes.
'''
def get_octave_offset(midi: stream.Stream):
    mini = 100
    for _ in midi.recurse():
        if type(_) is note.Note:
            mini = min(_.octave, mini)
    return mini

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

'''
Extract all the notes and rests from the melody (midi file uploaded to Stream object).
'''
def get_notes(mstream: stream.Stream):
    return [n for n in mstream.recurse() if type(n) in [note.Note, note.Rest]]

'''
Define key chord, key notes and some useful inversions of the triads.
'''
def fitness_help():
    tone_notes = SETTINGS['TONE'].pitches # extract all (ступени)

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

        SETTINGS['TONE_CHORDS'] = [
            chord.Chord(roman.RomanNumeral('i', SETTINGS['TONE']).pitches),
            chord.Chord(roman.RomanNumeral('II', SETTINGS['TONE']).pitches),
            chord.Chord(roman.RomanNumeral('III', SETTINGS['TONE']).pitches),
            chord.Chord(roman.RomanNumeral('iv', SETTINGS['TONE']).pitches),
            chord.Chord(roman.RomanNumeral('v', SETTINGS['TONE']).pitches),
            chord.Chord(roman.RomanNumeral('VI', SETTINGS['TONE']).pitches),
            chord.Chord(roman.RomanNumeral('VII', SETTINGS['TONE']).pitches)
        ]

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

        SETTINGS['TONE_CHORDS'] = [
            chord.Chord(roman.RomanNumeral('I', SETTINGS['TONE']).pitches),
            chord.Chord(roman.RomanNumeral('ii', SETTINGS['TONE']).pitches),
            chord.Chord(roman.RomanNumeral('iii', SETTINGS['TONE']).pitches),
            chord.Chord(roman.RomanNumeral('IV', SETTINGS['TONE']).pitches),
            chord.Chord(roman.RomanNumeral('V', SETTINGS['TONE']).pitches),
            chord.Chord(roman.RomanNumeral('vi', SETTINGS['TONE']).pitches),
            chord.Chord(roman.RomanNumeral('vii', SETTINGS['TONE']).pitches)
        ]


    SETTINGS['TONE_NOTES'] = []
    for _ in SETTINGS['TONE_CHORDS']:
        SETTINGS['TONE_NOTES'].append(_.pitches[0].name)

    return [T35, T6, T64, S35, S6, S64, D35, D6, D64]
    

'''
Fitness function for the exact chord and corresponding notes.
Follows the different rules describes in the report.
'''
def fitness(notes, chord_: chord.Chord) -> float:

    f = 0 # sum of functions
    k = 2.7 # coefficient for functions

    # if it's at least consonant
    if chord_.isConsonant():
        f += k * 7
    else:
        f = 0

    # check if here is obvious wrong chord
    pf = [m.pitch.ps for m in chord_.notes]
    for i in range(1, len(pf)):
        dif = abs(pf[i-1] - pf[i])
        if dif < 2 or dif >= 12:
            f = 0
            return f

    # check if the notes of the chord matched notes of the melody
    cnt_from_tone = 0 # count of notes in chord from tone notes
    for _ in chord_.pitchNames:
        if _ in SETTINGS['TONE_NOTES']:
            f += k**2 * 3
            cnt_from_tone += 1
    if cnt_from_tone == 0:
        f = 0
        return f

    # check if the chord hits the of on best between tone chords
    best_chords = np.zeros(7) # count matched notes for the every of 7 tone chords
    for n in notes:
        for c in range(len(SETTINGS['TONE_CHORDS'])):
            if n.name in set(SETTINGS['TONE_CHORDS'][c].pitchNames):
                best_chords[c] += 1
    bch = np.argmax(best_chords)
    prf = set(SETTINGS['TONE_CHORDS'][bch].pitchNames)
    matched = len(set(chord_.pitchNames).intersection(prf))

    if matched == 3:
        f += k**3 * matched
        return f
    elif matched == 2:
        f += k**2.5 * matched
    elif matched == 1:
        f += k * matched

    # check if matches at least one tone chord
    chord_set = set(chord_.pitchNames)
    for _ in SETTINGS['TONE_CHORDS']:
        if chord_set == set(_.pitchNames):
            f += math.log(k**2) * 10
            return f

    return f

'''
Exctract right notes for the chord.
Count overall fitness score over all chords.
'''
def individual_fitness(notes, chords: list[chord.Chord]) -> float:
    dnotes = notes
    dntime = np.zeros(len(dnotes) + 1)
    for i in range(1, len(dntime)):
        dntime[i] = dntime[i-1] + dnotes[i-1].duration.quarterLength
    dntime = dntime[1:]

    dchords = chords
    dctime = np.zeros(len(dchords) + 1)
    for i in range(1, len(dctime)):
        dctime[i] = dctime[i-1] + dchords[i-1].duration.quarterLength
    dctime = dctime[1:]

    fit = 0
    lhs = 0
    for i in range(0, len(dctime)):
        rhs = np.argmax(dntime > dctime[i])
        m_for_c = dnotes[lhs:rhs] # notes for chord
        fit += fitness(m_for_c, dchords[i])
        lhs = rhs

    return fit

'''
Create chord sequence.
Returns list of Chords.
'''
def get_individual() -> list[chord.Chord]:
    bars = SETTINGS['BARS_COUNT']
    chord_seq = make_chord_seq(bars * int(4 / SETTINGS['CHORD_DURATION']))

    return chord_seq

'''
See report for detailed explanation.
'''
def crossover(individuals):
    l = len(individuals) // 2
    for i in range(0, l, 2):
        x = individuals[i][1]
        y = individuals[i+1][1]
        
        new_child = []
        for j in range(len(x)):
            prob = random.random()
            if prob <= 0.7:
                new_child.append(x[j])
            else:
                new_child.append(y[j])
        if random.random() <= 0.1:
            child = mutation(new_child)
        else:
            child = new_child
        individuals.append((individual_fitness(SETTINGS['NOTES'], child), child))

'''
See report for detailed explanation.
'''
def mutation(child):
    for ind in range(len(child)): # chord seq
        tone = random.sample(range(0, 3), 3)
        sign = [random.random() for _ in range(0, 3)]
        midis = [p.midi for p in child[ind].pitches] # midi for this chord
        for i in range(0, 3):
            if sign[i] >= 0.5:
                midis[i] += -1 * tone[i]
            else:
                midis[i] += tone[i]
        
        n1 = note.Note(midis[0])
        n2 = note.Note(midis[1])
        n3 = note.Note(midis[2])
        n1.volume.velocity = SETTINGS['VOLUME']
        n2.volume.velocity = SETTINGS['VOLUME']
        n3.volume.velocity = SETTINGS['VOLUME']
        ch = chord.Chord()
        ch.duration = duration.Duration(SETTINGS['CHORD_DURATION'])
        ch.add([n1,n2,n3])

        child[ind] = ch

    return child


def evolution(generations: int):
    indvs = []
    pop_size = 1000
    print('Generate base population...')
    for i in range(pop_size):
        individual = get_individual()
        indvs.append((individual_fitness(SETTINGS['NOTES'], individual), individual))
    print('Base population is generated!')

    for gen in range(generations):
        print(f'Started epoch {gen}')
        indvs.sort(key = lambda x: x[0], reverse = True)
        crossover(indvs)
        indvs.sort(key = lambda x: x[0], reverse = True)
        indvs = indvs[:pop_size]
        print(f'Finished epoch {gen}')
    return indvs[0]

def save_midi(song: stream.Stream, fname='output.mid'):
    song.write('midi', fp=fname)

def main(infile: str, outfile: str):
    mstream = read_midi(infile)
    set_settings(mstream, vadjust=5)

    output = evolution(100)
    print(output[0])
    track = make_chord_track(output[1])
    mstream.append(track)

    save_midi(mstream, outfile)

if __name__ == '__main__':
    # Check if we are inside the virtualenv
    # import os
    # print(os.environ['VIRTUAL_ENV'])
    start_time = time.time()
    main(sys.argv[1], sys.argv[2])
    print(f'--- {time.time() - start_time} seconds ---')
