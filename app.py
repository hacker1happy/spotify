import numpy as np
import pandas as pd
import pickle

from flask import Flask, render_template, request
# from module import SpotifyAPI

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


client_id = '0ac8819bc5a440d5be6a8e2fbca6cb39'
client_secret = '362b5487027c4695971e69a12fbb9ccc'

import requests
import datetime
from urllib.parse import urlencode
import base64



# get_audio_features
class SpotifyAPI(object):
    access_token = None
    access_token_expires = datetime.datetime.now()
    access_token_did_expire = True
    client_id = None
    client_secret = None
    token_url = "https://accounts.spotify.com/api/token"
    track = None
    # features = []

    def __init__(self, client_id, client_secret, track,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_id = client_id
        self.client_secret = client_secret
        self.track = track



    def get_client_credentials(self):
        """
        Returns a base64 encoded string
        """
        client_id = self.client_id
        client_secret = self.client_secret
        if client_secret == None or client_id == None:
            raise Exception("You must set client_id and client_secret")
        client_creds = f"{client_id}:{client_secret}"
        client_creds_b64 = base64.b64encode(client_creds.encode())
        return client_creds_b64.decode()
    
    def get_token_headers(self):
        client_creds_b64 = self.get_client_credentials()
        return {
            "Authorization": f"Basic {client_creds_b64}"
        }
    
    def get_token_data(self):
        return {
            "grant_type": "client_credentials"
        } 
    
    def perform_auth(self):
        token_url = self.token_url
        token_data = self.get_token_data()
        token_headers = self.get_token_headers()
        r = requests.post(token_url, data=token_data, headers=token_headers)
        if r.status_code not in range(200, 299):
            raise Exception("Could not authenticate client.")
            # return False
        data = r.json()
        now = datetime.datetime.now()
        access_token = data['access_token']
        expires_in = data['expires_in'] # seconds
        expires = now + datetime.timedelta(seconds=expires_in)
        self.access_token = access_token
        self.access_token_expires = expires
        self.access_token_did_expire = expires < now
        return True
    
    def get_access_token(self):
        token = self.access_token
        expires = self.access_token_expires
        now = datetime.datetime.now()
        if expires < now:
            self.perform_auth()
            return self.get_access_token()
        elif token == None:
            self.perform_auth()
            return self.get_access_token() 
        return token
    
    def get_resource_header(self):
        access_token = self.get_access_token()
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        return headers
        
        
    def get_resource(self, lookup_id, resource_type='albums', version='v1'):
        endpoint = f"https://api.spotify.com/{version}/{resource_type}/{lookup_id}"
        headers = self.get_resource_header()
        r = requests.get(endpoint, headers=headers)
        if r.status_code not in range(200, 299):
            return {}
        return r.json()
   
    def get_audio_features(self,_id):
        return self.get_resource(_id, resource_type='audio-features')  
  

    def base_search(self, query_params): # type
        headers = self.get_resource_header()
        endpoint = "https://api.spotify.com/v1/search"
        lookup_url = f"{endpoint}?{query_params}"
        r = requests.get(lookup_url, headers=headers)
        if r.status_code not in range(200, 299):  
            return {}
        return r.json()

   
    def search(self, query=None, operator=None, operator_query=None, search_type='artist' ):
        if query == None:
            raise Exception("A query is required")
        if isinstance(query, dict):
            query = " ".join([f"{k}:{v}" for k,v in query.items()])
        if operator != None and operator_query != None:
            if operator.lower() == "or" or operator.lower() == "not":
                operator = operator.upper()
                if isinstance(operator_query, str):
                    query = f"{query} {operator} {operator_query}"
        query_params = urlencode({"q": query, "type": search_type.lower()})
        print(query_params)
        return self.base_search(query_params)


@app.route("/")
def home():
    return render_template ("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    # For rendering results on HTML GUI 

    if request.method == "POST":

        song = request.form["songName"]
        spotify = SpotifyAPI(client_id, client_secret,song)
        x = spotify.search({"track": song}, search_type="track")
        id = x['tracks']['items'][0]['id']
        abc = spotify.get_audio_features(id)
        rem_list = ['type', 'id', 'uri', 'track_href', 'analysis_url','speechiness']
        [abc.pop(key) for key in rem_list]
        l = list(abc.values())
        to_add = [0,0,0,0]
        if(l[-1]==1):
            to_add[0]=1
        elif(l[-1]==3):
            to_add[1]=1
        elif(l[-1]==4):
            to_add[2]=1
        else:
            to_add[3]=1
        l.pop(-1)
        l.extend(to_add)

        # x = [[0.719, 0.493, 8, -7.23, 1, 0.401, 0, 0.118, 0.124, 115.08, 224427, 0, 0, 1, 0]]
        final_features = [l]
        genre = model.predict(final_features)


        dataset = pd.read_csv('list.csv')
        users,c=[],0
        for i in dataset['Genre']:
            if(genre==i):
                users.append(dataset['Name'][c])
            c+=1
        return render_template ("predict.html",users=users)



if __name__ == "__main__":
    app.run()