from lib.dataset import Dataset2D
from lib.core.config import SYN_VIDEOS_DIR


class PennActionErase(Dataset2D):
    """
    The class to leverage the PennAction dataset with erasing as data augmentation
    """

    def __init__(self, seq_len, overlap=0.75, folder=None, debug=False):
        db_name = 'pennaction_erase'
        super(PennActionErase, self).__init__(
            seq_len=seq_len,
            folder=SYN_VIDEOS_DIR,
            dataset_name=db_name,
            debug=debug,
            overlap=overlap,
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')
