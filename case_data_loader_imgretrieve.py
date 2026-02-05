import json
import logging
import numpy as np
from pprg_end2end.utils import Pack
from pprg_end2end.data_loaders import BaseDataLoader
import ipdb
logger = logging.getLogger()


class CaseDataLoader_imgretrieve(BaseDataLoader):

    def __init__(self, mode, data, config):
        super(CaseDataLoader_imgretrieve, self).__init__(mode)
        self.mode = mode  # 'train'/'val'/'test'
        self.config = config
        self.data = data
        self.data_size = len(self.data)
        self.num_sim = self.config.num_sim
        self.indexes = [id for id in range(self.data_size)]

    def epoch_init(self, config, shuffle=True, verbose=True, drop_last=True):
        self.ptr = 0

        self.batch_size = config.batch_size
        if drop_last:
            self.num_batch = self.data_size // config.batch_size
        else:
            sized = self.data_size % config.batch_size == 0
            num_batch = self.data_size // config.batch_size
            self.num_batch = num_batch if sized else (num_batch + 1)

        if shuffle:
            self._shuffle_indexes()

        self.batch_indexes = []  # updated by batch_size
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i * self.batch_size: (i + 1) * self.batch_size])

        if verbose:
            if drop_last:
                print('Number of left over sample = %d' % (self.data_size - config.batch_size * self.num_batch))
            else:
                print('Do not drop last sample')

        if verbose:
            print('%s begins with %d batches' % (self.name, self.num_batch))

    def _prepare_batch(self, selected_index):
        rows = [self.data[idx] for idx in selected_index]
        batch_size = len(rows)
#         ipdb.set_trace()
        caseid_in = np.full((batch_size, 1), -1, dtype=int)
        rspvec_in = np.zeros((batch_size, 600), dtype=float)
        weighted_sim_imgvec_in = np.zeros((batch_size, 128), dtype=float)

        if self.mode == 'train':
            gt_imgvec_in = np.zeros((batch_size, 128), dtype=float)
            rd_imgvec_in = np.zeros((batch_size, 128), dtype=float)

            # fill in values
            for b_id in range(batch_size):
                caseid_in[b_id, 0] = rows[b_id].caseid
                rspvec_in[b_id, :] = rows[b_id].rspvec
                weighted_sim_imgvec_in[b_id, :] = rows[b_id].weighted_sim_imgvec
                gt_imgvec_in[b_id, :] = rows[b_id].gt_imgvec
                rd_imgvec_in[b_id, :] = rows[b_id].rd_imgvec

            return Pack(
                caseid_in=caseid_in,
                rspvec_in = rspvec_in,
                weighted_sim_imgvec_in = weighted_sim_imgvec_in,
                gt_imgvec_in = gt_imgvec_in,
                rd_imgvec_in = rd_imgvec_in
            )
        else:
            for b_id in range(batch_size):
                caseid_in[b_id, 0] = rows[b_id].caseid
                rspvec_in[b_id, :] = rows[b_id].rspvec
                weighted_sim_imgvec_in[b_id, :] = rows[b_id].weighted_sim_imgvec


            return Pack(
                caseid_in=caseid_in,
                rspvec_in=rspvec_in,
                weighted_sim_imgvec_in=weighted_sim_imgvec_in,
            )


    def clone(self):
        return CaseDataLoader_imgretrieve(self.mode, self.data, self.config)
