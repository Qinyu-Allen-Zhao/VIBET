from lib.dataset import Dataset3D
from lib.core.config import H36M_DIR

class H36M(Dataset3D):
    """
    The class to leverage the Human3.6M dataset
    """

    def __init__(self, set, seq_len, overlap=0.75, debug=False):
        db_name = 'h36m'

        # during testing we don't need data augmentation
        # but we can use it as an ensemble
        is_train = False
        overlap = overlap if is_train else 0.
        print('3DPW Dataset overlap ratio: ', overlap)
        super(H36M, self).__init__(
            set=set,
            folder=H36M_DIR,
            seq_len=seq_len,
            overlap=overlap,
            dataset_name=db_name,
            debug=debug,
        )
        print(f'{db_name} - number of dataset objects {self.__len__()}')