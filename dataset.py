import requests
from tqdm import tqdm
import json
import os
from typing import List
from pydub import AudioSegment
import io
import argparse

import torch
import numpy as np

from transformers import Wav2Vec2FeatureExtractor, AutoModel
from datasets import Dataset

ALARSO_1_ID = "59265af2c61444b39b3b180f4864015d"
ALARSO_1_SECRET =  "fc04505b01b24981adc52984043161ab"

ALARSO_2_ID = "bf8823d9c28949baada2054f948fd5f9"
ALARSO_2_SECRET = "a8321197eab943de9458fcc266087f63"

GPTUNES_ID = "2b8eed681f1746e290fef8574c11d303"
GPTUNES_SECRET = "ab4b4d88e5da41d29b3d987a7fa98a10"

class Embedder:
    def __init__(self, reset: bool):
        self.reset = reset
        self.model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        ).cuda()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )

    def embed_songs(self, artist_id: str,  song_urls: List[str], max_songs: int):
        os.makedirs("embeddings", exist_ok=True)
        if not self.reset and os.path.exists(f"embeddings/{artist_id}"): return
        os.makedirs(f"embeddings/{artist_id}", exist_ok=True)
        # Select random song_urls
        if len(song_urls) > max_songs:
            song_urls = np.random.choice(song_urls, max_songs, replace=False)
        for idx, url in tqdm(enumerate(song_urls[:max_songs]), leave=False):
            if url is None: continue
            file = requests.get(url).content
            song = self._downsample(file)
            song = self._to_array(song)
            embedding = self._gen_embedding(song)

            path = f"embeddings/{artist_id}/{idx}.txt"
            with open(path, "w") as f:
                f.write(json.dumps(embedding.tolist()))

    def _downsample(self, audio: bytes):
        original_audio = AudioSegment.from_mp3(io.BytesIO(audio))
        downsampled_audio = original_audio.set_frame_rate(24000)
        return downsampled_audio

    def _to_array(self, song: AudioSegment):
        song = song.get_array_of_samples()
        audio_array = np.array(song) / 32768
        audio_array = np.vstack((audio_array, audio_array))
        audio_array = audio_array.astype(np.float32)
        return audio_array

    def _gen_embedding(self, song: np.array):
        song = song[0, :]
        inputs = self.processor(song, sampling_rate=24000, return_tensors="pt")
        inputs["input_values"] = inputs["input_values"].cuda()
        inputs["attention_mask"] = inputs["attention_mask"].cuda()
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        embedding = outputs["last_hidden_state"]
        pool_size = embedding.shape[1] // 5
        embedding = torch.nn.functional.avg_pool1d(
            embedding.squeeze(0).transpose(0, 1),
            kernel_size=pool_size
        ).transpose(0, 1)
        assert embedding.shape[0] == 5
        return embedding


class TooManyRequest(Exception):
    def __init__(self):
        super().__init__()


class Dataset:
    def __init__(self):
        self.nodes = [] # [node_id, [node_attr]]
        self.edges = [] # [[from_node_id, to_node_id], [edge_attr]]

        url = "https://accounts.spotify.com/api/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": ALARSO_1_ID,
            "client_secret": ALARSO_1_SECRET
        }
        response = requests.post(url, data=data)
        token_data = response.json()
        access_token = token_data["access_token"]
        self.headers = {"Authorization": f"Bearer {access_token}"}

    def _request(self, url):
        r = requests.get(url, headers=self.headers)
        if r.status_code == 429: raise TooManyRequest()
        return r.json()

    def add_node(self, node_id, node_attr):
        if node_id not in [i[0] for i in self.nodes]:
            self.nodes.append([node_id, node_attr])

    def add_edge(self, from_node_id, to_node_id, edge_attr):
        if [from_node_id, to_node_id] not in [i[0] for i in self.edges]:
            self.edges.append([[from_node_id, to_node_id], edge_attr])

    def get_artist_node_from_id(self, artist_id, expanded=True):
        # Get artist information
        url = f"https://api.spotify.com/v1/artists/{artist_id}"
        r = self._request(url)
        name = r["name"]
        popularity = r["popularity"]
        followers = r["followers"]["total"]
        genres = r["genres"]
        # Get the album ids
        url = f"https://api.spotify.com/v1/artists/{artist_id}/albums"
        r = self._request(url)
        i = r["items"]
        album_ids = [album["id"] for album in i]

        # Get track urls here
        for album_id in tqdm(album_ids, leave=False, desc="Get album"):
            url = f"https://api.spotify.com/v1/albums/{album_id}/tracks"
            r = self._request(url)
            i = r["items"]
            song_urls = [song["preview_url"] for song in i]
            song_urls = [url for url in song_urls if url is not None]

        # Create node attributes
        node_attr =  {
            "name": name, "expanded": expanded, "type": "artist",
            "popularity": popularity, "followers": followers, "genres": genres,
            "album_ids": album_ids, "song_urls": song_urls
        }
        return artist_id, node_attr

    def expand_album(self, album_id):
        # Get the album object
        url  = f"https://api.spotify.com/v1/albums/{album_id}"
        album = self._request(url)

        album_artist_id = album["artists"][0]["id"]

        # Get the songs of the album
        url = f"https://api.spotify.com/v1/albums/{album_id}/tracks"
        response = requests.get(url, headers=self.headers).json()
        songs = response["items"]

        song_urls = []
        for song in songs:
            song_url = song["preview_url"]
            if song_url is not None: song_urls.append(song_url)
            # TODO: save the url of the song here so that you don't need to
            # get in the album downloader class
            song_artists = song["artists"]
            # Check for artists
            for artist in song_artists:
                artist_id = artist["id"]
                if artist_id != album_artist_id:
                    if artist_id not in [i[0] for i in self.nodes]:
                        artist_node = self.get_artist_node_from_id(artist_id, expanded=False)
                        self.add_node(*artist_node)
                    self.add_edge(artist_id, album_artist_id, {"type": "collaboration"})
            # Change the album_artist node to include the song_url

        return song_urls

    def expand_artist(self, artist_id, n_followers=500000):
        # If the artist is alraady in the graph, check if it is expanded.
        if artist_id in [i[0] for i in self.nodes]:
            artist_node = [i for i in self.nodes if i[0] == artist_id][0]
            if artist_node[1]["expanded"]:
                return
        # If it is not in the graph or expanded, add it
        else:
            artist_node = self.get_artist_node_from_id(artist_id)
            self.add_node(*artist_node)

        if artist_node[1]["followers"] < n_followers:
            print(f"Skipping {artist_id} because of {n_followers} followers")
            return

        song_urls = []
        for album_id in artist_node[1]["album_ids"]:
            print(f"Expanding albums of {album_id}", end="\r")
            song_urls.extend(self.expand_album(album_id))

        for node in self.nodes:
            if node[0] == artist_id:
                node[1]["song_urls"] = song_urls
                node[1]["expanded"] = True

    def populate_graph(self, start_artist_id, n_nodes = 1000):
        self.expand_artist(start_artist_id)
        for node in tqdm(self.nodes, leave=False):
            print(">>> expanded artist:", node[1]["name"])
            print(">>> number of nodes in graph:", len(self.nodes))
            print(">>> number of edges in graph:", len(self.edges))
            self.expand_artist(node[0])
            self.save("data")
            if len(self.nodes) > n_nodes: break

    def embed_tracks(self, album_downloader: Embedder):
        """ Iterate over the nodes and download the tracks of the artist.
        """
        # artist/album/songs
        not_available = 0
        for node in tqdm(self.nodes, desc="Embedding tracks"):
            if "song_urls" not in node[1].keys(): continue
            album_ids = node[1]["album_ids"]
            if len(album_ids) == 0: not_available += 1
            album_downloader.embed_songs(artist_id=node[0],
                                         song_urls=node[1]["song_urls"],
                                         max_songs=30)

        print(">>> prop not available:", not_available/len(self.nodes))

    def save(self, dir):
        with open(dir+"/nodes.txt", 'w') as f:
            json.dump(self.nodes, f)
        with open(dir+"/edges.txt", 'w') as f:
            json.dump(self.edges, f)

    def load(self, dir):
        if "nodes.txt" in os.listdir(dir):
            with open(dir+"/nodes.txt", 'r') as f:
                self.nodes = json.load(f)
        if "edges.txt" in os.listdir(dir):
            with open(dir+"/edges.txt", 'r') as f:
                self.edges = json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    reset = args.reset
    if reset:
        ans = input("Are you sure you want to reset the data? (y/n)")
        if ans == "y":
            for file in os.listdir("data"): os.remove("data/"+file)
        else:
            print("Aborting")

    dataset = Dataset()
    dataset.load("data")
    # dataset.populate_graph('5K4W6rqBFWDnAN6FQUkS6x', n_nodes=40000)
    embedder = Embedder(reset=False)
    dataset.embed_tracks(embedder)

    # TODO: pick a node at random to expand. Or maybe not, because the ordering
    # of the artists is used now, and then you will expand artists first that
    # were added earlier and they are closer to the start artist.

    # TODO: only get the 3 latest albums?

    print(">>> number of artist nodes:", len(dataset.nodes))


if __name__ == "__main__":
    main()