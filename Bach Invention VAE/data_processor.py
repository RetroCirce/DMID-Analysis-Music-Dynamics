'''
 Music2Vec data procesor
 process the data to different model repr.
'''
import numpy as np
import pretty_midi as pyd
import random
from tqdm import tqdm

VAEs = ["MeasureVAE","EC2VAE","SketchVAE","PianoTree","SignalVAE","PianorollVAE","EventVAE","MVVAE"]
mono_cg = {
    "max_note":107,
    "min_note":24,
    "hold":84,
    "rest":85 
}

class DatasetProcessor:
    def __init__(self, data):
        self.data = np.load(data, allow_pickle = True)
        print("dataset size: %d" %(len(self.data)))

    def split(self, ratio, shuffle = True):
        # ratio: [train, valid, test]
        assert len(ratio) == 3 and sum(ratio) == 1.0, "the sum of ratio must be one"
        if shuffle:
            random.seed(19991217)
            random.shuffle(self.data)
        ratio[1] += ratio[0]
        ratio[2] += ratio[1]
        l = len(self.data)
        self.trainset = self.data[:int(l * ratio[0])]
        self.validset = self.data[int(l * ratio[0]):int(l * ratio[1])]
        self.testset = self.data[int(l * ratio[1]):int(l * ratio[2])]
        print("train:%d \t validate:%d \t test:%d" %(len(self.trainset),len(self.validset), len(self.testset)))       

    def process(self, dataset, vae = "MeasureVAE", spb = 48, bar_num = 2, shift_note = [0]):
        assert vae in VAEs, "VAE structure must be in proposed models"
        errors = []
        tokens = []
        if vae == "MeasureVAE":
            bar_set = tqdm(dataset)
            for st in shift_note:
                for seq in bar_set:
                    msg, token = self.mono_tokenize(seq, spb, bar_num, st)
                    if token is None:
                    #    print(seq)
                        errors.append(seq["filename"] + " " + msg)
                    #    break
                    else:
                        tokens += np.split(token, len(token) // (spb * bar_num)) 
                        bar_set.set_description(seq["filename"] + " " + msg)
                        # self.mono_generate(token = token, metadata = seq)
            print("total len:%d" % len(tokens))
        elif vae == "EC2VAE":
            bar_set = tqdm(dataset)
            for st in shift_note:
                for seq in bar_set:
                    msg, token = self.mono_tokenize(seq, spb, bar_num, st)
                    if token is None:
                        errors.append(seq["filename"] + " " + msg)
                    else:
                        tokens += np.split(token, len(token) // (spb * bar_num)) 
                        bar_set.set_description(seq["filename"] + " " + msg)
                        # self.mono_generate(token = token, metadata = seq)
            # convert onehot
            print("total len:%d" % len(tokens))
            oh_tokens = []
            for token in tokens:
                token = [int(d) for d in token]
                midi_vec = np.zeros((len(token),86))
                k = np.arange(len(token))
                midi_vec[k,token] = 1
                oh_tokens.append(midi_vec)
            tokens = [tokens, oh_tokens]
        elif vae == "SketchVAE":
            bar_set = tqdm(dataset)
            for st in shift_note:
                for seq in bar_set:
                    msg, token = self.mono_tokenize(seq, spb, bar_num, st)
                    if token is None:
                        errors.append(seq["filename"] + " mono ext " + msg)
                    else:
                        tokens += np.split(token, len(token) // (spb * bar_num)) 
                        bar_set.set_description(seq["filename"] + " " + msg)
                    # self.mono_generate(token = token, metadata = seq)
            print("total len:%d" % len(tokens))
            p_tokens = []
            l_tokens = []
            or_tokens = []
            bar_set = tqdm(tokens)
            for token in bar_set:
                token = [int(d) for d in token]
                temp = []
                r_temp = np.zeros((len(token), 3))
                for i,d in enumerate(token):
                    if d < mono_cg["hold"]:
                        temp.append(d)
                        r_temp[i, 0] = 1
                    elif d == mono_cg["hold"]:
                        r_temp[i, 1] = 1
                    else:
                        r_temp[i, 2] = 1
                len_temp = len(temp)
                if len_temp == 0:
                    len_temp = 1
                temp.extend([mono_cg["hold"]] * (len(token) - len(temp)))
                p_tokens.append(temp)
                l_tokens.append(len_temp)
                or_tokens.append(r_temp)
            tokens = [tokens, p_tokens, l_tokens, or_tokens]
        print("error files:%d" % len(errors))
        print(errors)
        return tokens

    def mono_tokenize(self, seq, spb = 96, bar_num = 1, shift_note = 0):
        if len(seq["data"]) == 0:
            return ["no data", None]
        if seq["tempo"][0][0] != 0.0:
            return ["tempo error", None]
        if max(seq["data"], key = lambda x:x.pitch).pitch + shift_note > mono_cg["max_note"]:
            return ["too high note", None]
        if min(seq["data"], key = lambda x:x.pitch).pitch + shift_note < mono_cg["min_note"]:
            return ["too low note", None]
        if seq["ts"].time != 0:
            return ["time signature error", None]
        bar_time = 120.0 / seq["tempo"][1][0] # * seq["ts"].numerator / seq["ts"].denominator # round(240.0 / seq["tempo"][1][0] * seq["ts"].numerator / seq["ts"].denominator,7)
        min_step = bar_time / spb
        endtime = seq["data"][-1].start
        total_bar = round(endtime / bar_time) + 1
        # print(bar_time, endtime, total_bar)
        token = np.ones((int(total_bar * spb))) * mono_cg["hold"]
        for note in seq["data"]:
            pos_bar = note.start // bar_time
            pos_tick = (note.start - bar_time * pos_bar) / min_step
            # print(pos_tick)
            if abs(pos_tick - round(pos_tick)) > 1e-3:
                print(pos_tick ,round(pos_tick), pos_tick - round(pos_tick))
                # print(note.start, bar_time, min_step)
                return ["too quantized data", None]
            token[int(round(pos_bar * spb + pos_tick))] = note.pitch - mono_cg["min_note"] + shift_note
        has_note = False
        for i in range(len(token)):
            if i % (bar_num * spb) == 0:
                has_note = False
            if has_note == False and token[i] == mono_cg["hold"]:
                token[i] = mono_cg["rest"]
            if token[i] < mono_cg["hold"]:
                has_note = True
        while (len(token) // spb) % bar_num != 0:
            token = token[:int(len(token) - spb)]
        return ["process ok", token]

    def mono_generate(self, token, metadata, spb = 48, filename = "test.mid"):
        gen_midi = pyd.PrettyMIDI(initial_tempo=metadata["tempo"][1][0])
        melodies = pyd.Instrument(program = 0)
        ct = 0.0
        lt = 0.0
        min_step = 120.0 / metadata["tempo"][1][0] / spb # * metadata["ts"].numerator / metadata["ts"].denominator / spb
        cp = mono_cg["rest"]
        for note in token:
            if note >= mono_cg["hold"] and cp == mono_cg["rest"]:
                ct += min_step
                lt += min_step
            elif note < mono_cg["hold"] and cp == mono_cg["rest"]:
                cp = note
                lt = ct
                ct += min_step
            elif note < mono_cg["hold"] and cp < mono_cg["hold"]:
                melodies.notes.append(
                    pyd.Note(
                        pitch = int(cp + mono_cg["min_note"]),
                        start = lt, end = ct, velocity = 90
                    )
                )
                cp = note
                lt = ct
                ct += min_step
            elif note == mono_cg["rest"] and cp < mono_cg["hold"]:
                melodies.notes.append(
                    pyd.Note(
                        pitch = int(cp + mono_cg["min_note"]),
                        start = lt, end = ct, velocity = 90
                    )
                )
                cp = note
                ct += min_step
                lt = ct
            elif note == mono_cg["hold"] and cp < mono_cg["hold"]:
                ct += min_step
        gen_midi.instruments.append(melodies)
        gen_midi.write(filename)
        
