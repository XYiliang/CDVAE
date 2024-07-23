from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import os
import numpy as np
import shutil
import random
from metrics import *
from folktables import ACSDataSource, ACSEmployment, ACSIncome
import pandas as pd
from randaugment import RandAugment_face as RandAugment
import copy
import joblib
from sklearn.model_selection import train_test_split


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform
        self.transform_strong = copy.deepcopy(transform)
        self.transform_strong.transforms.insert(0, RandAugment(3, 5))

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform_strong(inp)
        return out1, out2


class UTKFaceDataset(Dataset):

    def __init__(self, root, images_name_list, transform=None, transform_strong=None):
        self.root = root
        self.transform = transform
        self.transform_strong = transform_strong
        self.images_name = images_name_list

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, index):
        attrs = self.images_name[index].split('_')

        assert len(attrs) == 4, index

        age = int(attrs[0])
        gender = int(attrs[1])
        race = int(attrs[2])

        # [age] is an integer from 0 to 116, indicating the age
        # [gender] is either 0 (male) or 1 (female)
        # [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others
        # [date&time] is in the format of yyyymmddHHMMSSFFF,
        # showing the date and time an image was collected to UTKFace
        assert age in range(0, 117)
        assert gender in [0, 1]
        assert race in [0, 1, 2, 3, 4]

        # make binary
        if race > 0:
            race = 1

        label = {'age': age, 'gender': gender, 'race': race}

        image_path = os.path.join(self.root, self.images_name[index]).rstrip()
        assert os.path.isfile(image_path)
        image = Image.open(image_path).convert('RGB')

        image_transformed = image
        if self.transform and self.transform_strong:
            image_transformed = (self.transform(image), self.transform_strong(image))
        elif self.transform:
            image_transformed = self.transform(image)

        return {'image': image_transformed, 'label': label}


class FairFaceDataset(Dataset):
    def __init__(self, root, images_file, transform=None, transform_strong=None):
        self.root = root
        self.transform = transform
        self.transform_strong = transform_strong
        self.images_df = pd.read_csv(images_file)

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, index):

        attrs = self.images_df.iloc[index]

        file = attrs["file"]
        age = attrs["age"]
        gender = attrs["gender"]
        race = attrs["race"]

        assert gender in [0, 1]
        assert race in [0, 1]

        label = {'age': age, 'gender': gender, 'race': race}

        image_path = os.path.join(self.root, file).rstrip()
        assert os.path.isfile(image_path)
        image = Image.open(image_path).convert('RGB')

        image_transformed = image

        if self.transform and self.transform_strong:
            image_transformed = (self.transform(image), self.transform_strong(image))
        elif self.transform:
            image_transformed = self.transform(image)

        return {'image': image_transformed, 'label': label}


class SyntheticDataset(Dataset):
    def __init__(self, phase='train', domain='source'):
        source_data = pd.read_csv('datasets/synthetic/syn_source.csv', index_col=0).iloc[:15000, :]
        target_data = pd.read_csv('datasets/synthetic/b1(16)b2(5)xn1(mean45).csv', index_col=0).iloc[:15000, :]
        scaler = StandardScaler()
        total = pd.concat([source_data, target_data], axis=0)
        scaler.fit(total[['xn1', 'xa1', 'xa2']].values)
        if domain == 'source':
            data_train_src, data_test_src = self.process_synthetic(source_data, scaler)
            if phase == 'train':
                self.x, self.a, self.y = data_train_src
                self.x_trans = self.transformation(self.x)
                del data_test_src
            else:
                self.x, self.a, self.y = data_test_src
                self.x_trans = self.transformation(self.x)
                del data_train_src
        else:
            data_train_trg, data_test_trg = self.process_synthetic(target_data, scaler)
            if phase == 'train':
                self.x, self.a, self.y = data_train_trg
                self.x_trans = self.transformation(self.x)
                del data_test_trg
            else:
                self.x, self.a, self.y = data_test_trg
                self.x_trans = self.transformation(self.x)
                del data_train_trg
        del source_data, target_data, scaler

    def process_synthetic(self, data, scaler):
        data[['xn1', 'xa1', 'xa2']] = scaler.transform(data[['xn1', 'xa1', 'xa2']].values)
        x = data[['xn1', 'xn2', 'xa1', 'xa2']].values
        a = data['a'].values.reshape(-1, 1)
        y = data['y'].values.reshape(-1, 1)

        x_train, x_test, a_train, a_test, y_train, y_test = train_test_split(x, a, y, train_size=0.8, random_state=42)
        data_train = x_train, a_train, y_train
        data_test = x_test, a_test, y_test
        return data_train, data_test

    def transformation(self, features):
        # all the features are normalized by z-score
        corrupt_idx = random.sample([0, 1, 2, 3], 2)
        features_copy = copy.deepcopy(features)
        features_copy[:, 0] = np.clip(features[:, 0] + np.random.randint(0, 1), -0.5, 0.5) if 0 in corrupt_idx else features[:, 0]
        features_copy[:, 1] = features_copy[:, 1]
        features_copy[:, 2] = np.clip(features[:, 2] + np.random.randint(0, 1), -0.5, 0.5) if 0 in corrupt_idx else features[:, 2]
        features_copy[:, 2] = np.clip(features[:, 3] + np.random.randint(0, 1), -0.5, 0.5) if 0 in corrupt_idx else features[:, 3]

        return features_copy

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (self.x[index], self.x_trans[index]), self.y[index], self.a[index]


class NewAdultDataset(Dataset):

    def __init__(self, root, states, exclude_states, year, binary_split="non-white",
                 task="income", phase='train',
                 **kwargs):
        self.root = root
        self.all_states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD',
                           'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH',
                           'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                           'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
        if states == 'all':
            self.states = list(set(self.all_states) - set(states))
        else:
            self.states = states

        self.scaler_age = joblib.load('NewAdultStandardscalerAge.pkl')
        self.scaler_wkhp = joblib.load('NewAdultStandardscalerWKHP.pkl')

        self.year = year
        self.binary_split = binary_split
        self.task = task
        self.phase = phase

        self.ACSIncome_features = [
            'AGEP',
            'COW',
            'SCHL',
            'MAR',
            'OCCP',
            'POBP',
            'RELP',
            'WKHP',
            'SEX',
            'RAC1P',
        ]
        # self.fns = {
        #     # 'AGEP': lambda x: self.onehot(np.digitize(x, self.buckets['age']), len(self.buckets['age']) + 1),
        #     'AGEP': lambda x: self.scaler_age.transform(x),
        #     'COW': lambda x: self.onehot(x.astype(int) - 1, 9),
        #     # #'fnlwgt': lambda x: continuous(x),
        #     'SCHL': lambda x: self.onehot(x.astype(int) - 1, 24),
        #     # #'education-num': lambda x: continuous(x),
        #     'MAR': lambda x: self.onehot(x.astype(int) - 1, 5),
        #     'OCCP': lambda x: self.onehot(np.digitize(x, self.buckets['occupation']),
        #                                   len(self.buckets['occupation']) + 1),
        #     # 'OCCP': lambda x: onehot(x, options['occupation']),
        #     'RELP': lambda x: self.onehot(x.astype(int), 18),
        #     'RAC1P': lambda x: self.onehot(x.astype(int) - 1, 9),
        #     'SEX': lambda x: self.onehot(x.astype(int) - 1, 2),
        #     # #'capital-gain': lambda x: continuous(x),
        #     # #'capital-loss': lambda x: continuous(x),
        #     # 'WKHP': lambda x: self.onehot(np.digitize(x, self.buckets['hour']), len(self.buckets['hour']) + 1),
        #     'WKHP': lambda x: self.scaler_wkhp.transform(x),
        #     'POBP': lambda x: self.onehot((x // 100).astype(int), 6),
        #     # #'income': lambda x: onehot(x.strip('.'), options['income']),
        # }

        self.EPS = 1e-8
        # self.buckets = {'age': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
        #                 'hour': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
        #                 'occupation': [441, 961, 1241, 1561, 1981, 2061, 2181, 2556, 2921, 3551, 3656, 3961, 4151, 4256,
        #                                4656, 4966, 5941, 6131, 6951, 7641, 8991, 9431, 9761, 9831]}
        self.buckets = {'age': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
                        'hour': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],
                        'occupation': [441, 751, 961, 1241, 1561, 1981, 2061, 2181, 2556, 2921, 3551, 3656, 3961, 4161,
                                       4256, 4656, 4966, 5941, 6131, 6766, 6951, 7641, 8991, 9761, 9831],
                        'pobp': [79, 170, 255, 400, 470, 528]}
        self.load()

    def load(self):
        assert self.task in ["employment", "income"]
        data_source = ACSDataSource(survey_year=self.year, horizon='1-Year', survey='person', root_dir=self.root)
        acs_data = data_source.get_data(states=self.states, download=False)

        if self.task == "employment":
            features, label, group = ACSEmployment.df_to_numpy(acs_data)
        elif self.task == "income":
            features, label, group = ACSIncome.df_to_numpy(acs_data)
        if self.binary_split == "non-white":
            group[group != 1] = 0
            group = 1 - group
        idx_train = int(len(features) * 0.7)

        if self.phase == "train":
            self.x_trans = self.transformation(features[:idx_train])
            self.x = self.preprocess(features[:idx_train])
            self.x_trans = self.preprocess(self.x_trans)
            self.y = np.where(label[:idx_train] == True, 1, 0)
            self.a = np.where(group[:idx_train] == 'Male', 1, 0)
        elif self.phase == "test":
            self.x_trans = self.transformation(features[idx_train:])
            self.x = self.preprocess(features[idx_train:])
            self.x_trans = self.preprocess(self.x_trans)
            # self.x = self.preprocess(features[idx_train:])
            self.y = np.where(label[idx_train:] == True, 1, 0)
            self.a = np.where(group[idx_train:] == 'Male', 1, 0)

    def preprocess(self, features):
        # [
        #     0'AGEP',
        #     1'COW',
        #     2'SCHL',
        #     3'MAR',
        #     4'OCCP',
        #     5'POBP',
        #     6'RELP',
        #     7'WKHP',
        #     8'SEX',
        #     9'RAC1P',
        # ]
        # a = np.unique(features[:, 5])
        # self.fns = {
        #     # 'AGEP': lambda x: self.onehot(np.digitize(x, self.buckets['age']), len(self.buckets['age']) + 1),
        #     'AGEP': lambda x: self.scaler_age.transform(x),
        #     'COW': lambda x: self.onehot(x.astype(int) - 1, 9),
        #     # #'fnlwgt': lambda x: continuous(x),
        #     'SCHL': lambda x: self.onehot(x.astype(int) - 1, 24),
        #     # #'education-num': lambda x: continuous(x),
        #     'MAR': lambda x: self.onehot(x.astype(int) - 1, 5),
        #     'OCCP': lambda x: self.onehot(np.digitize(x, self.buckets['occupation']),
        #                                   len(self.buckets['occupation']) + 1),
        #     # 'OCCP': lambda x: onehot(x, options['occupation']),
        #     'RELP': lambda x: self.onehot(x.astype(int), 18),
        #     'RAC1P': lambda x: self.onehot(x.astype(int) - 1, 9),
        #     'SEX': lambda x: self.onehot(x.astype(int) - 1, 2),
        #     # #'capital-gain': lambda x: continuous(x),
        #     # #'capital-loss': lambda x: continuous(x),
        #     # 'WKHP': lambda x: self.onehot(np.digitize(x, self.buckets['hour']), len(self.buckets['hour']) + 1),
        #     'WKHP': lambda x: self.scaler_wkhp.transform(x),
        #     'POBP': lambda x: self.onehot((x // 100).astype(int), 6),
        #     # #'income': lambda x: onehot(x.strip('.'), options['income']),
        # }
        return np.concatenate([
            self.scaler_age.transform(features[:, 0].reshape(-1, 1)),  # AGEP

            self.onehot(features[:, 9].astype(int) - 1, 9),  # RAC1P
            self.onehot(np.digitize(features[:, 5].astype(int), self.buckets['pobp']), 7),  # POBP

            self.scaler_wkhp.transform(features[:, 7].reshape(-1, 1)),  # WKHP

            self.onehot((features[:, 2] - 1).astype(int), 24),  # SCHL
            self.onehot(features[:, 1].astype(int) - 1, 9),  # COW
            self.onehot((features[:, 3] - 1).astype(int), 5),  # MAR
            self.onehot(np.digitize(features[:, 4].astype(int), self.buckets['occupation']),
                        len(self.buckets['occupation']) + 1),  # OCCP
            self.onehot(features[:, 6].astype(int), 18),  # RELP
        ],
            axis=1
        )

        # age = self.scaler_age.transform(features[0])
        # cow = self.onehot(features[1].astype(int) - 1, 9)
        # schl = self.onehot(features[2] - 1, 24)
        # mar = self.onehot(features[3] - 1, 5)
        # occp = self.onehot(np.digitize(features[4], self.buckets['occupation']), len(self.buckets['occupation']) + 1)
        # relp = self.onehot(features[5].astype(int), 18)
        # race = self.onehot(features[6].astype(int) - 1, 9)
        # sex = features[7].astype(float)
        # wkhp = self.scaler_wkhp.transform(features[8])
        # pobp = self.onehot((features[0] // 100).astype(int), 6)
        #
        #
        # return one_hot_features

    # def preprocess(self, features):
    #     for i in range(len(features)):
    #         if i == 0:
    #             one_hot_features = self.fns[self.ACSIncome_features[i]](features[i])
    #         else:
    #             a = self.fns[self.ACSIncome_features[i]](features[i])
    #             one_hot_features = np.concatenate((one_hot_features, a))
    #
    #     return one_hot_features

    def onehot(self, a, num_classes):
        return np.squeeze(np.eye(num_classes)[a])

    def whiten(self, X, mn, std):
        mntile = np.tile(mn, (X.shape[0], 1))
        stdtile = np.maximum(np.tile(std, (X.shape[0], 1)), self.EPS)
        X = X - mntile
        X = np.divide(X, stdtile)
        return X

    def transformation(self, features):
        # random corruption (half of the features)
        POBP_list = np.array([0, 100, 200, 300, 400, 500])
        features_copy = copy.deepcopy(features)
        # self.feature_corruption = {
        #     'AGEP': lambda x: max(min(99, x + np.random.randint(-5, 6)), 16),
        #     'COW': lambda x: x,
        #     'SCHL': lambda x: max(min(24, x + np.random.randint(-2, 3)), 0),
        #     'MAR': lambda x: np.random.randint(1, 6),
        #     'OCCP': lambda x: x,
        #     'RELP': lambda x: np.random.randint(0, 18),
        #     'RAC1P': lambda x: np.random.randint(1, 10),
        #     'SEX': lambda x: x,
        #     'WKHP': lambda x: max(min(99, x + np.random.randint(-5, 6)), 1),
        #     'POBP': lambda x: POBP_list[np.random.randint(0, 6)],
        # }
        feature_list = [*range(len(self.ACSIncome_features))]
        corrupt_idx = random.sample(feature_list, len(feature_list) // 2)
        s = features_copy.shape[0]
        features_copy[:, 0] = np.clip(features[:, 0] + np.random.randint(-5, 6, size=s), 16, 99) if 0 in corrupt_idx else features[:, 0]  # AGEP
        features[:, 1] = features[:, 1]  # COW
        features_copy[:, 2] = np.clip(features[:, 2] + np.random.randint(-2, 3, size=s), 0, 24) if 2 in corrupt_idx else features[:, 2]  # SCHL
        features_copy[:, 3] = np.random.randint(1, 6, size=s) if 3 in corrupt_idx else features[:, 3]  # MAR
        features_copy[:, 4] = features_copy[:, 4]  # OCCP
        features_copy[:, 5] = POBP_list[np.random.randint(0, 6, size=s)] if 5 in corrupt_idx else features[:, 5]  # POBP
        features_copy[:, 6] = np.random.randint(0, 18, size=s) if 6 in corrupt_idx else features[:, 6]   #RELP
        features_copy[:, 7] = np.clip(features_copy[:, 7] + np.random.randint(-5, 6, size=s), 1, 99) if 7 in corrupt_idx else features[:, 7]    # WKHP
        features_copy[:, 8] = features_copy[:, 8]   # SEX
        features_copy[:, 9] = np.random.randint(1, 10, size=s) if 9 in corrupt_idx else features[:, 9]  # RAC1P

        return features_copy

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        # data, target, group = self.x[index], self.y[index], self.a[index]
        # data_trans = copy.deepcopy(data)
        # data_trans = self.transformation(data_trans)
        # data = self.preprocess(data), self.preprocess(data_trans)
        # x, x_trans, target, group = self.x[index], self.x_trans[index], self.y[index], self.a[index]
        # data = x, x_trans

        return (self.x[index], self.x_trans[index]), self.y[index], self.a[index]


class ShapesDataset(Dataset):
    def __init__(self, Y, Y_binary, A, A_binary, Y0A0, Y0A1, Y1A0, Y1A1, D, D_dist, data_path, seed=0,
                 batch_size=None, phase='train', transform=None, size=1000, train_proportion=0.7):
        self.FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                                 'orientation']
        self._NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                                       'scale': 8, 'shape': 4, 'orientation': 15}
        self.VALUES_PER_FACTOR = {

            'floor_hue': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'wall_hue': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'object_hue': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'scale': [0.75, 0.82142857, 0.89285714, 0.96428571, 1.03571429,
                      1.10714286, 1.17857143, 1.25],
            'shape': [0., 1., 2., 3.],
            'orientation': [-30., -25.71428571, -21.42857143, -17.14285714,
                            -12.85714286, -8.57142857, -4.28571429, 0.,
                            4.28571429, 8.57142857, 12.85714286, 17.14285714,
                            21.42857143, 25.71428571, 30.]

        }

        self.Y = Y
        self.Y_binary = Y_binary  # indices for [Y_0, Y_1]
        self.A = A
        self.A_binary = A_binary  # indices for [A_0, A_1]
        self.Y0A0 = Y0A0
        self.Y0A1 = Y0A1
        self.Y1A0 = Y1A0
        self.Y1A1 = Y1A1

        self.D = D
        self.D_dist = D_dist
        self.batch_size = batch_size
        self.phase = phase
        self.data_path = data_path
        self.seed = seed
        self.transform = transform
        self.size = size
        self.train_proportion = train_proportion

        self.load()

    def load(self):
        np.random.seed(self.seed)

        split_file_name = str(self.Y) + "_" + str(self.Y_binary) + "_" + str(
            self.A) + "_" + str(self.A_binary) + "_" + str(self.Y0A0) + "_" + str(self.Y0A1) + "_" + str(
            self.Y1A0) + "_" + str(self.Y1A1) + "_" + str(self.D) + "_" + str(self.D_dist) + "_" + str(self.size)

        split_file = os.path.join(self.data_path, split_file_name)

        if not os.path.isfile(split_file + ".npz"):
            indices_D = np.random.multinomial(1, self.D_dist, self.size)
            indices_D = np.argmax(indices_D, axis=1)
            indices_A = np.concatenate((np.zeros(int(self.size * self.Y0A0)), np.ones(int(self.size * self.Y0A1))))
            indices_A_ = np.concatenate((np.zeros(int(self.size * self.Y1A0)), np.ones(int(self.size * self.Y1A1))))
            indices_A = np.concatenate((indices_A, indices_A_))
            indices_Y = np.concatenate((np.zeros(int(self.size * self.Y0A0)), np.zeros(int(self.size * self.Y0A1))))
            indices_Y_ = np.concatenate((np.ones(int(self.size * self.Y1A0)), np.ones(int(self.size * self.Y1A1))))
            indices_Y = np.concatenate((indices_Y, indices_Y_))

            indices_A = np.asarray(
                [self.A_binary[1] if x == 1 else self.A_binary[0] for x in indices_A])  # Map to binary case of A
            indices_Y = np.asarray(
                [self.Y_binary[1] if x == 1 else self.Y_binary[0] for x in indices_Y])  # Map to binary case of Y

            factors = np.zeros([len(self.FACTORS_IN_ORDER), self.size], dtype=np.int32)
            for factor, name in enumerate(self.FACTORS_IN_ORDER):
                num_choices = self._NUM_VALUES_PER_FACTOR[name]
                factors[factor] = np.random.choice(num_choices, self.size)
            factors[self.FACTORS_IN_ORDER.index(self.D)] = indices_D
            factors[self.FACTORS_IN_ORDER.index(self.A)] = indices_A
            factors[self.FACTORS_IN_ORDER.index(self.Y)] = indices_Y
            indices = get_index(factors)

            # split into train, val, test
            n_s = indices.shape[0]
            shuf = np.random.permutation(n_s)
            test_pct = 1 - self.train_proportion
            valid_pct = 0.0
            valid_ct = int(n_s * valid_pct)
            test_ct = int(n_s * test_pct)
            self.train_inds = indices[shuf][valid_ct + test_ct:]
            self.val_inds = indices[shuf][test_ct:valid_ct + test_ct]
            self.test_inds = indices[shuf][:test_ct]

            np.savez(split_file,
                     train_inds=self.train_inds, valid_inds=self.val_inds, test_inds=self.test_inds
                     )

            self.loaded = True
            if self.phase == 'train':
                self.inds = self.train_inds
            elif self.phase == 'valid':
                self.inds = self.val_inds
            elif self.phase == 'test':
                self.inds = self.test_inds
            else:
                raise Exception("invalid phase name")
        else:
            indices = np.load(split_file + ".npz")
            if self.phase == 'train':
                self.inds = indices["train_inds"]
            elif self.phase == 'valid':
                self.inds = indices["valid_inds"]
            elif self.phase == 'test':
                self.inds = indices["test_inds"]
            else:
                raise Exception("invalid phase name")

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, index):
        split_index = self.inds[index]
        image_path = os.path.join(self.data_path, "images", str(split_index) + ".jpg")
        assert os.path.isfile(image_path)
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
            # im_data = np.asarray(image)
            # im = np.moveaxis(im_data, -1, 0) / 255.
            # image = torch.from_numpy(im)
        dict_factor = get_factor(split_index)
        Y_binary = dict_factor[self.Y] == self.Y_binary[1]
        Y_binary = int(Y_binary)
        A_binary = dict_factor[self.A] == self.A_binary[1]
        A_binary = int(A_binary)
        return image, Y_binary, A_binary


def mb_round(t, bs):
    """take array t and batch_size bs and trim t to make it divide evenly by bs"""
    new_length = len(t) // bs * bs
    return t[:new_length, :]


def get_index(factors):
    FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                        'orientation']
    _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                              'scale': 8, 'shape': 4, 'orientation': 15}
    indices = 0
    base = 1
    for factor, name in reversed(list(enumerate(FACTORS_IN_ORDER))):
        indices += factors[factor] * base
        base *= _NUM_VALUES_PER_FACTOR[name]
    return indices


# Given an index, returns a dictionary of value (in interval unit) for each factor
def get_factor(index):
    FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                        'orientation']
    _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                              'scale': 8, 'shape': 4, 'orientation': 15}
    factors = {}
    for facotr, name in list(enumerate(FACTORS_IN_ORDER)):
        start = _NUM_VALUES_PER_FACTOR[name]
        _NUM_VALUES_PER_FACTOR.pop(name)
        while True:
            mult = np.prod(np.array(list(_NUM_VALUES_PER_FACTOR.values())))
            if start * mult <= index:
                break
            else:
                start -= 1
        factors[name] = start
        index -= start * mult
    return factors
