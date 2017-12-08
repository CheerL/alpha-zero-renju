import oss2
import os

class OssManager(object):
    AKID = 'LTAIaXXWIaH9B4E1'
    AKSECRET = 'JQHnZ5Ej99kpQeKdw8eRWx2r1eXGZp'
    ENDPOINT = 'oss-cn-shanghai.aliyuncs.com'
    DL_OSS = 'cheer-dl'

    def __init__(self):
        self.auth = oss2.Auth(self.AKID, self.AKSECRET)
        self.bucket = oss2.Bucket(self.auth, self.ENDPOINT, self.DL_OSS, connect_timeout=10)

    def list_dir(self, path, prefix=''):
        return [obj.key for obj in oss2.ObjectIterator(self.bucket, prefix=path + prefix)]

    def download_files(self, files=[], to_path='.'):
        for file_key in files:
            if not file_key.endswith('/'):
                file_name = file_key.split('/')[-1]
                file_path = os.path.join(to_path, file_name)
                self.bucket.get_object_to_file(file_key, file_path)