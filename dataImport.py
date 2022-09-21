# To import raw data, spike sorted data from a local folder or S3 UUID

import numpy as np
import h5py
import itertools
import pandas as pd
import braingeneers
import posixpath
import braingeneers.utils.smart_open_braingeneers as smart_open
import zipfile
import os
import shutil
from braingeneers import analysis


class LoadData:
    def __init__(self):
        pass

    def load_phy_s3(self, batch_uuid: str, dataset_name: str, type='default'):
        """
        Load the spike times, channels and templates from phy numpy files after spike sorting.
        :param batch_uuid: the UUID of the dataset
        :param dataset_name: name of the dataset. Because a UUID can have multiple datasets.
        :param type: 'default' spike sorting output or 'curated' output
        :param fs: recording's sample rate
        :return: analysis.SpikeData class with a list of spike time lists and neuron_data.
                 neuron_data = {new_cluster_id:[channel_id, (chan_pos_x, chan_pos_y),
                                 [chan_template], {channel_id:cluster_templates}]}
        """
        # TODO: update metadata after sorting to allow loading by experiment_id
        base_path = 's3://braingeneers/' \
            if braingeneers.get_default_endpoint().startswith('http') \
            else braingeneers.get_default_endpoint()
        if type == 'curated':
            dataset = dataset_name + '_curated.zip'
        else:
            dataset = dataset_name + '_phy.zip'
        phy_full_path = \
            posixpath.join(base_path, 'ephys',
                           batch_uuid, 'derived/kilosort2', dataset)
        spikeData = self.read_phy_files(phy_full_path)
        return spikeData

    def load_phy_local(self, path: str):
        """
        Load phy files from a local folder or zipped file.
        If the input is a directory which has params.py, the content will be temporarily zipped.
        :param directory: A folder directory or a file directory
        :return: a spikeData object.
        """
        if os.path.isfile(path):
            assert path[-3:] == 'zip', "Input a zipp file path or folder path."
            spikeData = self.read_phy_files(path)
        elif os.path.isdir(path):
            zipped_file = shutil.make_archive('phy_files', 'zip', path)
            spikeData = self.read_phy_files(zipped_file)
        return spikeData

    def read_phy_files(self, path: str, fs=20000):
        try:
            if path[:2] == 's3' and path[-3:] == 'zip':
                f = smart_open.open(path, 'rb')
            elif os.path.isfile(path) and path[-3:] == 'zip':
                f = path
        except ValueError:
            print("Input must be a zip file path.")

        with zipfile.ZipFile(f, 'r') as f_zip:
            assert 'params.py' in f_zip.namelist(), "Wrong spike sorting output."
            if 'cluster_info.tsv' in f_zip.namelist():
                cluster_info = pd.read_csv(f_zip.open('cluster_info.tsv'), sep='\t')
                groups = list(cluster_info['group'])
                cluster_ids = list(cluster_info['cluster_id'])
                ch = list(cluster_info['ch'])
                labeled_clusters = []
                best_channels = []
                for i in range(len(groups)):
                    if groups[i] != 'noise':
                        labeled_clusters.append(cluster_ids[i])
                        best_channels.append(ch[i])
                clusters = np.load(f_zip.open('spike_clusters.npy'))
                templates = np.load(f_zip.open('templates.npy'))
                channels = np.load(f_zip.open('channel_map.npy'))
            else:
                clusters = np.load(f_zip.open('spike_clusters.npy'))
                templates = np.load(f_zip.open('templates.npy'))
                channels = np.load(f_zip.open('channel_map.npy'))
                labeled_clusters = np.unique(clusters)
                best_channels = [channels[np.argmax(np.ptp(templates[i], axis=0))][0]
                                 for i in labeled_clusters]

            spike_templates = np.load(f_zip.open('spike_templates.npy'))
            spike_times = np.load(f_zip.open('spike_times.npy')) / fs * 1e3
            positions = np.load(f_zip.open('channel_positions.npy'))

        if isinstance(channels[0], np.ndarray):
            channels = np.asarray(list(itertools.chain.from_iterable(channels)))
        if isinstance(clusters[0], np.ndarray):
            clusters = list(itertools.chain.from_iterable(clusters))
        if isinstance(spike_times[0], np.ndarray):
            spike_times = list(itertools.chain.from_iterable(spike_times))
        if isinstance(spike_templates[0], np.ndarray):
            spike_templates = np.asarray(list(itertools.chain.from_iterable(spike_templates)))

        df = pd.DataFrame({"clusters": clusters, "spikeTimes": spike_times})
        cluster_spikes = df.groupby("clusters").agg({"spikeTimes": lambda x: list(x)})
        cluster_spikes = cluster_spikes[cluster_spikes.index.isin(labeled_clusters)]

        labeled_clusters = np.asarray(labeled_clusters)
        if max(labeled_clusters) >= templates.shape[0]:  # units are spited or merged during curation
            ind = np.where(labeled_clusters >= templates.shape[0])[0]
            for i in ind:
                spike_ids = np.nonzero(np.in1d(clusters, labeled_clusters[i]))[0]
                original_cluster = np.unique((spike_templates[spike_ids]))[0]  # to simplify the process,
                # take the first original cluster for merged clusters because merge happens when
                # two clusters templates are similar
                labeled_clusters[i] = original_cluster

        chan_indices = np.searchsorted(channels, best_channels)
        cluster_indices = np.searchsorted(np.unique(spike_templates), labeled_clusters)
        chan_template = templates[cluster_indices, :, chan_indices]

        cluster_templates = []
        for i in cluster_indices:
            nbgh_chans = np.nonzero(templates[i].any(0))[0]
            nbgh_temps = np.transpose(templates[i][:, templates[i].any(0)])
            nbgh_dict = dict(zip(channels[nbgh_chans], nbgh_temps))
            cluster_templates.append(nbgh_dict)

        chan_pos = positions[chan_indices]

        # re-assign cluster id
        new_clusters = np.arange(len(labeled_clusters))
        neuron_data = dict(zip(new_clusters,
                               zip(best_channels, chan_pos, chan_template, cluster_templates)))
        neuron_dict = {0: neuron_data}
        spikedata = analysis.SpikeData(list(cluster_spikes["spikeTimes"]),
                                       neuron_data=neuron_dict)
        return spikedata