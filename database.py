import os
import lmdb
import utils
import numpy as np


class LMDB(object):
    def __init__(self, name):
        self.name = name
        self.db_path = os.path.join(utils.DB_PATH, name)
        self.lmdb_env = lmdb.open(self.db_path)

    def put(self, key, val):
        with self.lmdb_env.begin(write=True) as lmdb_txn:
            try:
                lmdb_txn.put(key, val)
            except TypeError:
                if isinstance(val, str):
                    lmdb_txn.put(key.encode(), val.encode())
                else:
                    lmdb_txn.put(key.encode(), val)
    
    def get(self, key):
        with self.lmdb_env.begin() as lmdb_txn:
            try:
                val = lmdb_txn.get(key)
                return val
            except TypeError:
                val = lmdb_txn.get(key.encode())
                return val

    def get_np(self, key):
        np_string = self.get(key)
        return np.fromstring(np_string)
