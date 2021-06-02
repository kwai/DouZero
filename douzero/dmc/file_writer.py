# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import datetime
import csv
import json
import logging
import os
import time
from typing import Dict

import git


def gather_metadata() -> Dict:
    date_start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    # gathering git metadata
    try:
        repo = git.Repo(search_parent_directories=True)
        git_sha = repo.commit().hexsha
        git_data = dict(
            commit=git_sha,
            branch=repo.active_branch.name,
            is_dirty=repo.is_dirty(),
            path=repo.git_dir,
        )
    except git.InvalidGitRepositoryError:
        git_data = None
    # gathering slurm metadata
    if 'SLURM_JOB_ID' in os.environ:
        slurm_env_keys = [k for k in os.environ if k.startswith('SLURM')]
        slurm_data = {}
        for k in slurm_env_keys:
            d_key = k.replace('SLURM_', '').replace('SLURMD_', '').lower()
            slurm_data[d_key] = os.environ[k]
    else:
        slurm_data = None
    return dict(
        date_start=date_start,
        date_end=None,
        successful=False,
        git=git_data,
        slurm=slurm_data,
        env=os.environ.copy(),
    )


class FileWriter:
    def __init__(self,
                 xpid: str = None,
                 xp_args: dict = None,
                 rootdir: str = '~/palaas'):
        if not xpid:
            # make unique id
            xpid = '{proc}_{unixtime}'.format(
                proc=os.getpid(), unixtime=int(time.time()))
        self.xpid = xpid
        self._tick = 0

        # metadata gathering
        if xp_args is None:
            xp_args = {}
        self.metadata = gather_metadata()
        # we need to copy the args, otherwise when we close the file writer
        # (and rewrite the args) we might have non-serializable objects (or
        # other nasty stuff).
        self.metadata['args'] = copy.deepcopy(xp_args)
        self.metadata['xpid'] = self.xpid

        formatter = logging.Formatter('%(message)s')
        self._logger = logging.getLogger('palaas/out')

        # to stdout handler
        shandle = logging.StreamHandler()
        shandle.setFormatter(formatter)
        self._logger.addHandler(shandle)
        self._logger.setLevel(logging.INFO)

        rootdir = os.path.expandvars(os.path.expanduser(rootdir))
        # to file handler
        self.basepath = os.path.join(rootdir, self.xpid)

        if not os.path.exists(self.basepath):
            self._logger.info('Creating log directory: %s', self.basepath)
            os.makedirs(self.basepath, exist_ok=True)
        else:
            self._logger.info('Found log directory: %s', self.basepath)

        # NOTE: remove latest because it creates errors when running on slurm 
        # multiple jobs trying to write to latest but cannot find it 
        # Add 'latest' as symlink unless it exists and is no symlink.
        # symlink = os.path.join(rootdir, 'latest')
        # if os.path.islink(symlink):
        #     os.remove(symlink)
        # if not os.path.exists(symlink):
        #     os.symlink(self.basepath, symlink)
        #     self._logger.info('Symlinked log directory: %s', symlink)

        self.paths = dict(
            msg='{base}/out.log'.format(base=self.basepath),
            logs='{base}/logs.csv'.format(base=self.basepath),
            fields='{base}/fields.csv'.format(base=self.basepath),
            meta='{base}/meta.json'.format(base=self.basepath),
        )

        self._logger.info('Saving arguments to %s', self.paths['meta'])
        if os.path.exists(self.paths['meta']):
            self._logger.warning('Path to meta file already exists. '
                                 'Not overriding meta.')
        else:
            self._save_metadata()

        self._logger.info('Saving messages to %s', self.paths['msg'])
        if os.path.exists(self.paths['msg']):
            self._logger.warning('Path to message file already exists. '
                                 'New data will be appended.')

        fhandle = logging.FileHandler(self.paths['msg'])
        fhandle.setFormatter(formatter)
        self._logger.addHandler(fhandle)

        self._logger.info('Saving logs data to %s', self.paths['logs'])
        self._logger.info('Saving logs\' fields to %s', self.paths['fields'])
        if os.path.exists(self.paths['logs']):
            self._logger.warning('Path to log file already exists. '
                                 'New data will be appended.')
            with open(self.paths['fields'], 'r') as csvfile:
                reader = csv.reader(csvfile)
                self.fieldnames = list(reader)[0]
        else:
            self.fieldnames = ['_tick', '_time']

    def log(self, to_log: Dict, tick: int = None,
            verbose: bool = False) -> None:
        if tick is not None:
            raise NotImplementedError
        else:
            to_log['_tick'] = self._tick
            self._tick += 1
        to_log['_time'] = time.time()

        old_len = len(self.fieldnames)
        for k in to_log:
            if k not in self.fieldnames:
                self.fieldnames.append(k)
        if old_len != len(self.fieldnames):
            with open(self.paths['fields'], 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(self.fieldnames)
            self._logger.info('Updated log fields: %s', self.fieldnames)

        if to_log['_tick'] == 0:
            # print("\ncreating logs file ")
            with open(self.paths['logs'], 'a') as f:
                f.write('# %s\n' % ','.join(self.fieldnames))

        if verbose:
            self._logger.info('LOG | %s', ', '.join(
                ['{}: {}'.format(k, to_log[k]) for k in sorted(to_log)]))

        with open(self.paths['logs'], 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(to_log)
            # print("\nadded to log file")

    def close(self, successful: bool = True) -> None:
        self.metadata['date_end'] = datetime.datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S.%f')
        self.metadata['successful'] = successful
        self._save_metadata()

    def _save_metadata(self) -> None:
        with open(self.paths['meta'], 'w') as jsonfile:
            json.dump(self.metadata, jsonfile, indent=4, sort_keys=True)
