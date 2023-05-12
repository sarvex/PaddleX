# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
import shutil
import requests
import tqdm
import time
import hashlib
import tarfile
import zipfile
import filelock
import paddle
from . import logging

DOWNLOAD_RETRY_LIMIT = 3


def md5check(fullname, md5sum=None):
    if md5sum is None:
        return True

    logging.info(f"File {fullname} md5 checking...")
    md5 = hashlib.md5()
    with open(fullname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        logging.info(
            f"File {fullname} md5 check failed, {calc_md5sum}(calc) != {md5sum}(base)"
        )
        return False
    return True


def move_and_merge_tree(src, dst):
    """
    Move src directory to dst, if dst is already exists,
    merge src to dst
    """
    if not osp.exists(dst):
        shutil.move(src, dst)
    else:
        for fp in os.listdir(src):
            src_fp = osp.join(src, fp)
            dst_fp = osp.join(dst, fp)
            if osp.isdir(src_fp):
                if osp.isdir(dst_fp):
                    move_and_merge_tree(src_fp, dst_fp)
                else:
                    shutil.move(src_fp, dst_fp)
            elif osp.isfile(src_fp) and \
                    not osp.isfile(dst_fp):
                shutil.move(src_fp, dst_fp)


def download(url, path, md5sum=None):
    """
    Download from url, save to path.

    url (str): download url
    path (str): download to given path
    """
    if not osp.exists(path):
        os.makedirs(path)

    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    retry_cnt = 0
    while not (osp.exists(fullname) and md5check(fullname, md5sum)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            logging.debug(f"{fname} download failed.")
            raise RuntimeError(f"Download from {url} failed. Retry limit reached")

        logging.info(f"Downloading {fname} from {url}")

        req = requests.get(url, stream=True)
        if req.status_code != 200:
            raise RuntimeError(
                f"Downloading from {url} failed with code {req.status_code}!"
            )

        # For protecting download interupted, download to
        # tmp_fullname firstly, move tmp_fullname to fullname
        # after download finished
        tmp_fullname = f"{fullname}_tmp"
        total_size = req.headers.get('content-length')
        with open(tmp_fullname, 'wb') as f:
            if total_size:
                download_size = 0
                current_time = time.time()
                for chunk in tqdm.tqdm(
                        req.iter_content(chunk_size=1024),
                        total=(int(total_size) + 1023) // 1024,
                        unit='KB'):
                    f.write(chunk)
                    download_size += 1024
                    if download_size % 524288 == 0:
                        total_size_m = round(
                            int(total_size) / 1024.0 / 1024.0, 2)
                        download_size_m = round(download_size / 1024.0 /
                                                1024.0, 2)
                        speed = int(524288 /
                                    (time.time() - current_time + 0.01) /
                                    1024.0)
                        current_time = time.time()
                        logging.debug(
                            f"Downloading: TotalSize={total_size_m}M, DownloadSize={download_size_m}M, Speed={speed}KB/s"
                        )
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_fullname, fullname)
        logging.debug(f"{fname} download completed.")

    return fullname


def decompress(fname):
    """
    Decompress for zip and tar file
    """
    logging.info(f"Decompressing {fname}...")

    # For protecting decompressing interupted,
    # decompress to fpath_tmp directory firstly, if decompress
    # successed, move decompress files to fpath and delete
    # fpath_tmp and remove download compress file.
    fpath = osp.split(fname)[0]
    fpath_tmp = osp.join(fpath, 'tmp')
    if osp.isdir(fpath_tmp):
        shutil.rmtree(fpath_tmp)
        os.makedirs(fpath_tmp)

    if fname.find('tar') >= 0 or fname.find('tgz') >= 0:
        with tarfile.open(fname) as tf:
            tf.extractall(path=fpath_tmp)
    elif fname.find('zip') >= 0:
        with zipfile.ZipFile(fname) as zf:
            zf.extractall(path=fpath_tmp)
    else:
        raise TypeError(f"Unsupport compress file type {fname}")

    for f in os.listdir(fpath_tmp):
        src_dir = osp.join(fpath_tmp, f)
        dst_dir = osp.join(fpath, f)
        move_and_merge_tree(src_dir, dst_dir)

    shutil.rmtree(fpath_tmp)
    logging.debug(f"{fname} decompressed.")


def url2dir(url, path):
    download(url, path)
    if url.endswith(('tgz', 'tar.gz', 'tar', 'zip')):
        fname = osp.split(url)[-1]
        savepath = osp.join(path, fname)
        decompress(savepath)


def download_and_decompress(url, path='.'):
    nranks = paddle.distributed.get_world_size()
    local_rank = paddle.distributed.get_rank()
    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    if url.endswith(('tgz', 'tar.gz', 'tar', 'zip')):
        fullname = osp.join(path, fname.split('.')[0])
    if nranks <= 1:
        url2dir(url, path)
    else:
        lock_path = f'{fullname}.lock'
        if not os.path.exists(fullname):
            with open(lock_path, 'w'):
                os.utime(lock_path, None)
            if local_rank == 0:
                url2dir(url, path)
                os.remove(lock_path)
            else:
                while os.path.exists(lock_path):
                    time.sleep(1)
    return fullname
