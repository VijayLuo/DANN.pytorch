from torch.utils.data import Dataset
import pickle
import config
import torch.nn.functional as F
import torch


class SEEDDataset(Dataset):
    def __init__(self, target_subject, train):
        self.train = train
        with open('data.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.target_subject = target_subject
        self.source_subject = [
            s for s in config.SUBJECTS_NAME if s != target_subject]

    def __len__(self):
        if(self.train):
            return config.DATA_NUMBER_OF_SUBJECT * len(self.source_subject)
        else:
            return config.DATA_NUMBER_OF_SUBJECT

    def __getitem__(self, index):
        if(self.train):
            subject_index = index // config.DATA_NUMBER_OF_SUBJECT
            subject_name = self.source_subject[subject_index]
            subject = self.data[subject_name]

            data = subject['data'][index % config.DATA_NUMBER_OF_SUBJECT]
            label = subject['label'][index % config.DATA_NUMBER_OF_SUBJECT]

            one_hot_label = torch.zeros(config.OUTPUT_SIZE)
            one_hot_label[int(label+1)] = 1

            one_hot_subject_index = torch.zeros(len(self.source_subject))
            one_hot_subject_index[subject_index] = 1

            return data, one_hot_label, one_hot_subject_index
        else:
            return self.data[self.target_subject]['data'], self.data[self.target_subject]['label']
