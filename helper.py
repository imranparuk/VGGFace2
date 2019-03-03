import pandas as pd


N_IDENTITY = 9131  # total number of identities in VGG Face2
N_IDENTITY_PRETRAIN = 8631  # the number of identities used in training by Caffe

class VGGFaceDatasetHelper(object):

    def __init__(self, meta_file):

        self.label_dict = self.identity_meta(meta_file)
        self.info_list = []

    def generate_dataset_dict_helper_file(self, file):

        with open(file, 'r') as f:
            for item, _file in enumerate(f):
                _file = _file.strip()
                id = _file.split("/")[1]

                label = self.label_dict[id]

                self.info_list.append({
                    'id': id,
                    'img': _file,
                    'label': label,
                })

    @staticmethod
    def generate_dataset_dict_helper_path(dir):
        pass

    @staticmethod
    def identity_meta(file):

        df = pd.read_csv(file, sep=',\s+', quoting=csv.QUOTE_ALL, encoding="utf-8")

        df["class"] = -1
        df.loc[df["Flag"] == 1, "class"] = range(N_IDENTITY_PRETRAIN)
        df.loc[df["Flag"] == 0, "class"] = range(N_IDENTITY_PRETRAIN, N_IDENTITY)

        key = df["Class_ID"].values
        val = df["class"].values

        return dict(zip(key, val))

    @staticmethod
    def pose_templates(file):
        pass

    @staticmethod
    def age_templates(file):
        pass

    @staticmethod
    def bb_landmarks_loose(file):
        pass

    @staticmethod
    def bb_landmarks(file):
        pass

    def check_dataset(self):
        pass