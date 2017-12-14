import os
import oss2
import utils

class OssManager(object):
    AKID = 'LTAIaXXWIaH9B4E1'
    AKSECRET = 'JQHnZ5Ej99kpQeKdw8eRWx2r1eXGZp'
    ENDPOINT = 'oss-cn-shanghai.aliyuncs.com'
    DL_OSS = 'cheer-dl'

    def __init__(self):
        self.auth = oss2.Auth(self.AKID, self.AKSECRET)
        self.bucket = oss2.Bucket(self.auth, self.ENDPOINT, self.DL_OSS, connect_timeout=60)

    def list_dir(self, path, prefix=''):
        return [obj.key for obj in oss2.ObjectIterator(self.bucket, prefix=path + prefix)]

    def download_file(self, file_key, to_path):
        self.bucket.get_object_to_file(file_key, to_path)

    def download_files(self, files=[], to_path='.'):
        for file_key in files:
            if not file_key.endswith('/'):
                file_name = file_key.split('/')[-1]
                file_path = os.path.join(to_path, file_name)
                self.download_file(file_key, file_path)

    def upload_file(self, from_path, file_key):
        self.bucket.put_object_from_file(from_path, file_key)

    def upload_files(self, form_path, to_path):
        pass

    def copy_file(self, from_path, to_path):
        self.bucket.copy_object(self.DL_OSS, from_path, to_path)

    def read_file(self, path):
        return self.bucket.get_object(path).read()

    def get_win_rate(self, parent_path):
        def get_win_rate_content(win_rate):
            model_num = win_rate.split('-')[1]
            black_win, white_win = map(int, self.read_file(win_rate).decode().split('-'))
            total = black_win + white_win
            return 'Model {}, total {}\nBlack win {} ({:.3f})\nWhite_win {} ({:.3f})'.format(
                model_num, total, black_win, black_win / total, white_win, white_win / total
                )

        win_rates = self.list_dir(parent_path + '/model/record/', 'winrate')
        return [get_win_rate_content(win_rate) for win_rate in win_rates]

    def get_compare(self, parent_path):
        def get_compare_content(compare):
            _, default_model_num, compare_model_num = compare.split('.')[0].split('-')
            compare_win, total = map(int, self.read_file(compare).decode().split('-'))
            return 'Compare model {} with default model {}, total {}\nCompare win {} ({:.3f})'.format(
                compare_model_num, default_model_num, total, compare_win, compare_win / total
                )

        compares = self.list_dir(parent_path + '/model/record/', 'compare')
        return [get_compare_content(compare) for compare in compares]
