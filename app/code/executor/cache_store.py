import os
import shutil

import json


class CacheSerialStore():
    def __init__(self, base_dir):
        self._cache_dir = os.path.join(base_dir, "_temp_cache")
        self._cache_file_path = os.path.join(
            self._cache_dir, "client_cache_serial_store.json")
        # succeeds even if directory exists.
        os.makedirs(self._cache_dir, exist_ok=True)

        self.client_cache_dict = {}
        if os.path.exists(self._cache_file_path):
            try:
                with open(self._cache_file_path, "r") as f:
                    self.client_cache_dict = json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error: Cache file")
                    self.client_cache_dict = {}

        self.client_cache_dir = os.path.join(
            self._cache_dir, "client_computation_cache")
        # succeeds even if directory exists.
        os.makedirs(self.client_cache_dir, exist_ok=True)

    def get_cache_dict(self):
        return self.client_cache_dict

    def get_cache_dir(self):
        return self.client_cache_dir

    def update_cache_dict(self, cache_dict):
        self.client_cache_dict.update(cache_dict)
        with open(self._cache_file_path, "w") as f:
            json.dump(self.client_cache_dict, f)

    def remove_cache(self):
        shutil.rmtree(self._cache_dir, ignore_errors=True)
