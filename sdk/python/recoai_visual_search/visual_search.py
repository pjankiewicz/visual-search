import requests
from models import *

class BearerAuth(requests.auth.AuthBase):

    def __init__(self, token):
        self.token = token

    def __call__(self, r):
        r.headers["Authorization"] = "Bearer " + self.token
        return r

def remove_none(obj):
    if isinstance(obj, (list, tuple, set)):
        return type(obj)(remove_none(x) for x in obj if x is not None)
    elif isinstance(obj, dict):
        return type(obj)((remove_none(k), remove_none(v))
                         for k, v in obj.items() if k is not None and v is not None)
    else:
        return obj

class RecoAIVisualSearch():

    def __init__(self, bearer_token, address):
        self.bearer_token = bearer_token
        self.address = address

    def generic_request(self, obj, endpoint):
        json = remove_none(obj.to_dict())
        response = requests.post(url=self.address + endpoint,
                                 json=json,
                                 auth=BearerAuth(self.bearer_token))
        return response

    def upsert_collection(self, upsert_collection: UpsertCollection):
        return self.generic_request(upsert_collection, "/upsert_collection")

    def remove_collection(self, remove_collection: RemoveCollection):
        return self.generic_request(remove_collection, "/remove_collection")

    def add_image(self, add_image: AddImage):
        return self.generic_request(add_image, "/add_image")

    def remove_image(self, remove_image: RemoveImage):
        return self.generic_request(remove_image, "/remove_image")

    def search_image(self, search_image: SearchImage):
        return self.generic_request(search_image, "/search_image")
