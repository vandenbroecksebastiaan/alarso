# node_index| node_attr
# edge_index | edge_attr

import requests
from tqdm import tqdm
import json
import os


class Dataset:
    def __init__(self):
        self.nodes = [] # [node_id, [node_attr]]
        self.edges = [] # [[from_node_id, to_node_id], [edge_attr]]

        url = "https://accounts.spotify.com/api/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": "59265af2c61444b39b3b180f4864015d",
            "client_secret": "fc04505b01b24981adc52984043161ab"
        }
        response = requests.post(url, data=data)
        token_data = response.json()
        access_token = token_data["access_token"]
        self.headers = {"Authorization": f"Bearer {access_token}"}

    def add_node(self, node_id, node_attr):
        # If the node is an artist and already in the graph, update the expanded
        # attribute
        if node_id not in [i[0] for i in self.nodes]:
            self.nodes.append([node_id, node_attr])

    def add_edge(self, from_node_id, to_node_id, edge_attr):
        if [from_node_id, to_node_id] not in [i[0] for i in self.edges]:
            self.edges.append([[from_node_id, to_node_id], edge_attr])

    def expand_album(self, album_id, genres):
        # Get the album object
        url  = f"https://api.spotify.com/v1/albums/{album_id}"
        album = requests.get(url, headers=self.headers).json()

        album_artist_id = album["artists"][0]["id"]
        title = album["name"]
        release_date = album["release_date"]
        popularity = album["popularity"]
        total_tracks = album["total_tracks"]

        artist_ids = [i["id"] for i in album["artists"]]

        self.add_node(album_id, {"name": title, "type": "album",
                                 "year": release_date, "popularity": popularity,
                                 "n_tracks": total_tracks, "genres": genres})
        for id in artist_ids: self.add_edge(album_id, id, {"type": "album_artist"})

        # Get the songs of the album
        url = f"https://api.spotify.com/v1/albums/{album_id}/tracks"
        response = requests.get(url, headers=self.headers).json()
        songs = response["items"]

        for song in songs:
            song_id = song["id"]
            song_name = song["name"]
            song_artists = song["artists"]
            song_duration = song["duration_ms"]
            n_available_markets = len(song["available_markets"])

            self.add_node(song_id, {"name": song_name, "type": "song",
                                    "duration": song_duration,
                                    "n_markets": n_available_markets})
            for artist in song_artists:
                artist_id = artist["id"]
                if artist_id != album_artist_id:
                    artist_name = artist["name"]
                    self.add_node(artist_id, {"name": artist_name, "type": "artist", "expanded": False})
                    self.add_edge(album_id, artist_id, {"type": "album_artist"})

            self.add_edge(song_id, album_id, {"type": "song_album"})

    def expand_artist(self, artist_name, n_followers=100000):
        url = f"https://api.spotify.com/v1/search?q={artist_name}" \
               "&type=artist&limit=1&include_external=audio"
        r = requests.get(url, headers=self.headers).json()
        i = r["artists"]["items"]

        artist_id = i[0]["id"]
        popularity = i[0]["popularity"]
        followers = 0 if i[0]["followers"]["total"] is None else i[0]["followers"]["total"]
        genres = i[0]["genres"]

        if len(i) == 0:
            print(f"--- Skipping artist {artist_name} with no items")
            return

        if followers < n_followers:
            print(f"--- Skipping artist: {artist_name} with {followers} followers")
            return

        # This step will be skipped if the artist_id is already in the graph
        self.nodes = [i for i in self.nodes if i[0] != artist_id]
        self.add_node(artist_id, {"name": artist_name, "type": "artist",
                                  "expanded": True, "popularity": popularity,
                                  "followers": followers, "genres": genres})

        # Get the albums of the artist
        url = f"https://api.spotify.com/v1/artists/{artist_id}/albums"
        r = requests.get(url, headers=self.headers)
        r = r.json()
        i = r["items"]
        album_ids = [album["id"] for album in i]

        for id in album_ids: self.expand_album(id, genres)

        # Set expanded to True
        for n in self.nodes:
            if n[0] == artist_id:
                n[1]["expanded"] = True

    def populate_graph(self, start_artist, n_nodes = 1000):
        self.expand_artist(start_artist)
        artist_nodes = [i for i in self.nodes if i[1]["type"] == "artist"
                        and i[1]["expanded"] == False]
        artist_names_to_expand = [i[1]["name"] for i in artist_nodes]
        for i in tqdm(artist_names_to_expand, leave=False):
            print(">>> expanded artist:", i)
            print(">>> number of nodes in graph:", len(self.nodes))
            print(">>> number of edges in graph:", len(self.edges))
            self.expand_artist(i, n_followers=1000000)
            self.save("data")
            if len(self.nodes) > n_nodes: break

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
    dataset = Dataset()
    dataset.load("data")
    dataset.populate_graph('Kanye West', n_nodes=40000)

    # Add an embeddding from music2vec as a node attribute

    print(">>> number of album nodes:", len([i for i in dataset.nodes if i[1]["type"] == "album"]))
    print(">>> number of artist nodes:", len([i for i in dataset.nodes if i[1]["type"] == "artist"]))
    print(">>> number of song nodes:", len([i for i in dataset.nodes if i[1]["type"] == "song"]))


if __name__ == "__main__":
    main()