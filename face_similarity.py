import base64
import logging
import os
from distutils import util
from itertools import chain, islice
import json

import msgpack
import numpy as np
import requests
import shutil
import ujson

dir_path = os.path.dirname(os.path.realpath(__file__))
test_cat = os.path.join(dir_path, 'images')

session = requests.Session()
session.trust_env = False

face_data_folderpath = "/home/ubuntu/jh/InsightFace-REST/src/api_trt/face_data"
tmp_data_folderpath = "/home/ubuntu/jh/InsightFace-REST/src/api_trt/tmp"
host = 'http://localhost'
port = 18081

logging.basicConfig(
    level='INFO',
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)


def to_chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


def file2base64(path):
    with open(path, mode='rb') as fl:
        encoded = base64.b64encode(fl.read()).decode('ascii')
        return encoded


def save_crop(data, name):
    img = base64.b64decode(data)
    with open(name, mode="wb") as fl:
        fl.write(img)
        fl.close()


def to_bool(input):
    try:
        return bool(util.strtobool(input))
    except:
        return False


class IFRClient:

    def __init__(self, host: str = 'http://localhost', port: int = '18081'):
        self.server = f'{host}:{port}'
        self.sess = requests.Session()

    def server_info(self, server: str = None, show=True):
        if server is None:
            server = self.server

        info_uri = f'{server}/info'
        info = self.sess.get(info_uri).json()

        if show:
            server_uri = self.server
            backend_name = info['models']['inference_backend']
            det_name = info['models']['det_name']
            rec_name = info['models']['rec_name']
            rec_batch_size = info['models']['rec_batch_size']
            det_batch_size = info['models']['det_batch_size']
            det_max_size = info['models']['max_size']

            print(f'Server: {server_uri}\n'
                  f'    Inference backend:      {backend_name}\n'
                  f'    Detection model:        {det_name}\n'
                  f'    Detection image size:   {det_max_size}\n'
                  f'    Detection batch size:   {det_batch_size}\n'
                  f'    Recognition model:      {rec_name}\n'
                  f'    Recognition batch size: {rec_batch_size}')

        return info

    def extract(self, data: list,
                mode: str = 'paths',
                server: str = None,
                threshold: float = 0.6,
                extract_embedding=True,
                return_face_data=False,
                return_landmarks=False,
                embed_only=False,
                limit_faces=0,
                use_msgpack=True):

        if server is None:
            server = self.server

        extract_uri = f'{server}/extract'

        if mode == 'data':
            images = dict(data=data)
        elif mode == 'paths':
            images = dict(urls=data)

        req = dict(images=images,
                   threshold=threshold,
                   extract_ga=False,
                   extract_embedding=extract_embedding,
                   return_face_data=return_face_data,
                   return_landmarks=return_landmarks,
                   embed_only=embed_only,  # If set to true API expects each image to be 112x112 face crop
                   limit_faces=limit_faces,  # Limit maximum number of processed faces, 0 = no limit
                   api_ver='2',
                   msgpack=use_msgpack,
                   )

        resp = self.sess.post(extract_uri, json=req, timeout=120)
        if resp.headers['content-type'] == 'application/x-msgpack':
            content = msgpack.loads(resp.content)
        else:
            content = ujson.loads(resp.content)

        images = content.get('data')
        for im in images:
            status = im.get('status')
            if status != 'ok':
                return None
            faces = im.get('faces', [])
            if not faces:
                return None
            for i, face in enumerate(faces):
                norm = face.get('norm', 0)
                prob = face.get('prob')
                size = face.get('size')
                facedata = face.get('facedata')
                if facedata:
                    if size > 20 and norm > 14:
                        save_crop(facedata, f'crops/{i}_{size}_{norm:2.0f}_{prob}.jpg')

        return content

def refresh_all_face_data():
    client = IFRClient(host=host, port=port)
    for id in os.listdir(face_data_folderpath):
        for person in os.listdir(os.path.join(face_data_folderpath, id)):
            for file in os.listdir(os.path.join(face_data_folderpath, id, person)):
                if not file == "data.json":
                    data = client.extract([os.path.join("face_data", id, person, file)])
                    if data:
                        with open(os.path.join(face_data_folderpath, id, person, 'data.json'), 'w') as fp:
                            json.dump(data["data"][0]["faces"][0]["vec"], fp, indent=4)

def new_face_data(id, person_name, filename):
    client = IFRClient(host=host, port=port)
    data = client.extract([os.path.join("tmp", filename)])
    if data:
        if not os.path.exists(os.path.join(face_data_folderpath, id)):
            os.mkdir(os.path.join(face_data_folderpath, id))
        if not os.path.exists(os.path.join(face_data_folderpath, id, person_name)):
            os.mkdir(os.path.join(face_data_folderpath, id, person_name))
        with open(os.path.join(face_data_folderpath, id, person_name, 'data.json'), 'w') as fp:
            json.dump(data["data"][0]["faces"][0]["vec"], fp, indent=4)
        shutil.copyfile(os.path.join(tmp_data_folderpath, filename), os.path.join(face_data_folderpath, id, person_name, filename))
        os.remove(os.path.join(tmp_data_folderpath, filename))

def load_face_data(id):
    vec_dict = {}
    for person in os.listdir(os.path.join(face_data_folderpath, id)):
        with open(os.path.join(face_data_folderpath, id, person, "data.json"), 'r') as fp:
            data = json.load(fp)
        vec_dict[person] = data
    return vec_dict

def run_similarity(id, path):
    client = IFRClient(host=host, port=port)

    vec_dict = load_face_data(id)

    data = client.extract([path])

    if not data:
        return None

    test_vec = data["data"][0]["faces"][0]["vec"]

    highest = 0
    highest_name = None
    for name, vec in vec_dict.items():
        dot_prod = np.dot(test_vec, vec)
        normalized = (1. + dot_prod) / 2.
        if normalized > highest:
            highest_name = name
            highest = normalized

    return highest_name, highest
