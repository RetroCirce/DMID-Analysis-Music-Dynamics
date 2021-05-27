'''
Music2Vec data loader
load data from different datasets
'''
import numpy as np
import os
import random
from tqdm import tqdm
import pretty_midi as pyd

Dataset = {"Nottingham", "Irish", "Wikifornia", "Hym"}

class DatasetLoader:
    def __init__(self, dataset_name, instrack = [0]):
        assert dataset_name in Dataset, "input dataset is not allowed"
        self.dataset_name = dataset_name
        self.instrack = instrack
        # if self.dataset_name == "Nottingham":
        #     self.loader = self.load_mono
        # elif self.dataset_name == "Irish":
        #     self.loader = self.load_mono
        # elif self.dataset_name == "Wikifornia":
        #     self.loader = self.load_mono
        # elif self.dataset_name == "Hym":
        #     self.loader = self.load_mono
        print("initialized a loader for " + dataset_name)

    def load(self, folder):
        songs =  tqdm(os.listdir(folder))
        data = []
        errors = []
        for song in songs:
            songs.set_description("processing %s" %(song))
            f = os.path.join(folder, song)
            try:
                midi_file = pyd.PrettyMIDI(f)
                if len(midi_file.time_signature_changes) != 1 or len(midi_file.get_tempo_changes()[1]) != 1:
                    raise BaseException
                temp = []
                for ins in self.instrack:
                    temp += midi_file.instruments[ins].notes
                temp.sort(key = lambda x:x.start)
                data.append({
                    "filename": f,
                    "data":temp,
                    "tempo": midi_file.get_tempo_changes(), 
                    "ts": midi_file.time_signature_changes[0]
                }
                )
            except BaseException:
                 errors.append("load " + song + " failed")
        print("%d in total" % len(data))
        print(errors)
        return data


