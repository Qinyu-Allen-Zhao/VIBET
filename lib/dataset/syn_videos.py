from lib.dataset import Dataset2D
from lib.core.config import SYN_VIDEOS_DIR


class SynVideos(Dataset2D):
    def __init__(self, seq_len, overlap=0.75, folder=None, debug=False):
        db_name = 'syn_videos'
        super(SynVideos, self).__init__(
            seq_len=seq_len,
            folder=SYN_VIDEOS_DIR,
            dataset_name=db_name,
            debug=debug,
            overlap=overlap,
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')
