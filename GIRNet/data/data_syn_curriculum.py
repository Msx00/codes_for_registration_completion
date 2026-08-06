import torch
from torch.utils.data import Dataset, DataLoader
from data.data import DisplDataset, LiverSample
from random import shuffle
# from data.data_syn import LiverSampleCombined
import math


class DisplDatasetCurriculum(DisplDataset):
    def __init__(self, 
            # path, 
            num_stages=5,
            num_epochs=100,
            shuffle_stage=False,
            combine_stages=False,
            bs=1,
            **kwargs,
    ):
        super(DisplDatasetCurriculum, self).__init__(**kwargs)
        self.num_stages = num_stages
        self.num_epochs = num_epochs
        self.num_epochs_per_stage = num_epochs // num_stages
        self.cur_stage = 0
        self.bs = bs
        self.num_samples_per_stage = len(self._samples_list) // num_stages # fixed
        self.num_samples_tail = len(self._samples_list) % num_stages
        # self.cur_sample_list = self._samples_list[self.cur_stage * self.num_samples_per_stage : self.num_samples_per_stage]
        self.num_samples = self.num_samples_per_stage # flexible, the last stage will change this value, this will determin the range of index
        self.shuffle_stage = shuffle_stage
        # self.index_list = list(range(self.num_samples))
        # if self.shuffle_stage:
        #     shuffle(self.index_list)
        self.combine_stages = combine_stages

        self.index_list_stage_total = self.create_index_list()


    def __len__(self):
        # return self.num_samples
        return len(self.index_list_stage_total[self.cur_stage])


    def __epochs_num_stage__(self,):
        if self.cur_stage == self.num_stages - 1:
            return self.num_epochs - self.num_epochs_per_stage * self.cur_stage
        else:
            return self.num_epochs_per_stage


    def __stage_steps__(self,):
        assert self.bs > 0, "batch size must be greater than 0"
        # calculate the total steps of the training, this is used for the OneCycleLR scheduler
        steps = math.ceil(len(self.index_list_stage_total[self.cur_stage] ) / self.bs)
        return steps


    def __total_steps__(self,):
        total_steps = 0
        for idx in range(self.num_stages):
            total_steps += self.__stage_steps__()
        return total_steps


    def create_index_list(self, ):
        index_list_stage_total = []
        for idx in range(self.num_stages):
            if self.combine_stages:
                if idx < self.num_stages - 1:
                    num_samples = (idx + 1) * self.num_samples_per_stage
                else:
                    num_samples = len(self._samples_list)
            else:
                if idx == self.num_stages - 1:
                    num_samples = self.num_samples_per_stage + self.num_samples_tail
                else:
                    num_samples = self.num_samples_per_stage
            index_list_stage = list(range(num_samples))
            if self.shuffle_stage:
                shuffle(index_list_stage)

            index_list_stage_total.append(index_list_stage)
        return index_list_stage_total



    def update_stage(self, cur_epoch):
        """Functions determin whether to update the stage based on the criteria (atm only epochs are supported)
        If the current epoch is greater than the number of epochs per stage, update the stage
        Note: run this function at the end of each epoch
        TODO: add more criteria for updating the stage
        Args:
            cur_epoch (int): the current epoch
        """
        updated = False
        if cur_epoch // self.num_epochs_per_stage > self.cur_stage and self.cur_stage + 1 < self.num_stages:
            # time to update the stage
            self.cur_stage += 1
            # if self.cur_stage == self.num_stages - 1:
            #     if not self.combine_stages:
            #         # reached the last stage, take all remaining samples for training
            #         self.num_samples = len(self._samples_list) - self.cur_stage * self.num_samples_per_stage
            #     else:
            #         self.num_samples = len(self._samples_list)
            # else:
            #     if self.combine_stages:
            #         self.num_samples = (self.cur_stage + 1) * self.num_samples_per_stage

            # self.index_list = list(range(self.num_samples))
            # if self.shuffle_stage:
            #     shuffle(self.index_list)
            
            print("********Epoch {} UPDATED to the stage {} with {} number of samples*********".format(cur_epoch, self.cur_stage, self.__len__()))
            updated = True
        else:
            print("Epoch {} still within stage {}  with {} number of samples".format(cur_epoch, self.cur_stage, self.__len__()))
        
        return updated


    def __getitem__(self, idx,):
        # if self.combine_stages:
        #     # if combined stages, there is NO offset
        #     idx_real = self.index_list[idx]
        # else:
        #     # if single stage, need to add up the offset of the previous stages
        #     idx_real = self.cur_stage * self.num_samples_per_stage + self.index_list[idx]
        idx_real = self.index_list_stage_total[self.cur_stage][idx]
        # if self.cur_stage == self.num_stages - 1:
        # reached the last stage, take all remaining samples for training
            # idx_real = self.cur_stage * self.num_samples_per_stage + idx 
        if idx < 5:
            # print the first 5 samples
            print("idx", idx, "---> idx_real:", idx_real,)
        res = super(DisplDatasetCurriculum, self).__getitem__(idx_real)
        return res



