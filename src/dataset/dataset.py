from typing import Tuple, Optional
from attr import define
import numpy as np

@define
class Data:
    x: np.ndarray
    y: np.ndarray

@define
class DataTwoFields:
    x_left: Optional[np.ndarray] = None
    x_right: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None

    def shape(self) -> None:
        print(f"x_left: {self.x_left.shape}, x_right: {self.x_right.shape}, y: {self.y.shape}")

class Dataset:
    def __init__(self, train: Data, test: Data):
        self.train = train
        self.test = test
        self.train_vf = DataTwoFields()
        self.test_vf = DataTwoFields()

    def shuffle(self, data: Data, n: Optional[int] = None):
        original_size = len(data.y)
        if n is None:
            random_indices = np.random.permutation(original_size)
        else:
            random_indices = np.random.choice(np.arange(0, original_size), n)

        return data.x[random_indices], data.y[random_indices]

    def shuffle_vf(self, vf: DataTwoFields, n: Optional[int] = None):
        original_size = len(vf.y)
        if n is None:
            random_indices = np.random.permutation(original_size)
        else:
            random_indices = np.random.choice(np.arange(0, original_size), n)

        return vf.x_left[random_indices], vf.x_right[random_indices], vf.y[random_indices]
    
    def build_vf(self, x_data: np.array, y_data: np.array) -> Tuple[np.array, np.array, np.array]:
        n = len(y_data) * self._final_size
        x_data_left, y_data_left = self.shuffle(Data(x_data, y_data), n)
        x_data_right, y_data_right = self.shuffle(Data(x_data, y_data), n)
        y_data_final = np.zeros(n, dtype=int)
        for i in range(n):
            data_with_cs = np.random.choice(
                [False, True], p=[1 - self._proportion_cs, self._proportion_cs]
            )
            data_with_left_attention = np.random.choice(
                [False, True], p=[1 - self._proportion_left, self._proportion_left]
            )

            # Determines value of attention if dataset entry is CS or SS
            if data_with_cs:
                attention = self._full_attention_value
                no_attention = self._reduced_attention_value
            else:
                attention = self._ss_attention_value
                no_attention = 0

            # Determines which visual field has attention
            if data_with_left_attention:
                x_data_left[i] *= attention
                x_data_right[i] *= no_attention
                y_data_final[i] = y_data_left[i]
            else:
                x_data_left[i] *= no_attention
                x_data_right[i] *= attention
                y_data_final[i] = y_data_right[i]

        return x_data_left, x_data_right, y_data_final
    
    def build_vf_dataset(
        self,
        final_size: float = 1,
        proportion_cs: float = 0.5,
        proportion_left: float = 0.5,
        full_attention_value: float = 1,
        reduced_attention_value: float = 0.5,
        ss_attention_value: float = 0.5,
    ) -> None:
        """
        Builds an entire double visual fields dataset, comprised of two visual fields, left and right, and an array of the corresponding label for both visual fields.

        Input:
        final_size: float: how many times the final dataset is bigger than the input data (default: 1)
        proportion_cs: float: proportion of entries in the final dataset that have CS over SS (default is 0.5)
        proportion_left: float: proportion of entries in the final dataset that have attention on the left visual field (default is 0.5)
        full_attention_value: float: value of the full attention in CS (default is 1)
        reduced_attention_value: float: value of the reduced attention in CS (default is 0.5)
        ss_attention_value: float: value of the attention for SS (default is 0.5)
        """
        
        self._final_size = final_size
        self._proportion_cs = proportion_cs
        self._proportion_left = proportion_left
        self._full_attention_value = full_attention_value
        self._reduced_attention_value = reduced_attention_value
        self._ss_attention_value = ss_attention_value
        
        self.train_vf.x_left, self.train_vf.x_right, self.train_vf.y = self.build_vf(self.train.x, self.train.y)
        self.test_vf.x_left, self.test_vf.x_right, self.test_vf.y = self.build_vf(self.test.x, self.test.y)