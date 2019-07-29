import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from feature_selector import FeatureSelector


class DataPreprocessing():

    def __init__(self, train, test):
        self._df_train = pd.read_csv(train)
        self._df_test = pd.read_csv(test).iloc[:, 1:]  # throw away redundant index column

    def handle_missing_data(self, test=False):
        """Impute missing values.

          Columns of dtype object are imputed with the most frequent value
          in column.

          Columns of other types are imputed with mean of column.

          """
        if test:
            X = pd.DataFrame(self._df_test)
            fill_na_test = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O')
                                                     else X[c].mean() for c in X], index=X.columns)

            self._fill_na_test_dict = {str(k): v for k, v in fill_na_test.items()}
            self._df_test.fillna(self._fill_na_test_dict, inplace=True)
        else:
            X = pd.DataFrame(self._df_train.drop(['label'], axis=1).values)
            fill_na_train = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O')
                                                     else X[c].mean() for c in X], index=X.columns)

            self._fill_na_train_dict = {str(k): v for k, v in fill_na_train.items()}
            self._df_train.fillna(self._fill_na_train_dict, inplace=True)


    def handle_unknown_features(self, test=False):
        """Nullify samples in which df[3] == 'unknown'
        """
        if test:
            self._df_test['3'].replace('unknown', np.NaN, inplace=True)
            self._df_test['8'].replace('?', np.NaN, inplace=True)
        else:
            self._df_train['3'].replace('unknown', np.NaN, inplace=True)
            self._df_train['8'].replace('?', np.NaN, inplace=True)



    def remove_outliers_from_train(self):
        #first find numerical cols
        cols = self._df_train.columns
        num_cols = list(self._df_train._get_numeric_data().columns)[:-1]  # drop label
        self._df_train = self._df_train[(np.abs(stats.zscore(self._df_train.loc[:, num_cols])) < 3).all(axis=1)]

    def encode_categorical(self, test=False):
        if test:
            # find categorial colimns
            cols = self._df_test.columns
            num_cols = self._df_test._get_numeric_data().columns
            categorical_indices = list(set(cols) - set(num_cols))

            # create dummis
            dummies_list = [pd.get_dummies(self._df_test[idx], prefix='{}'.format(idx)) for idx in categorical_indices]
            frames = dummies_list + [self._df_test]
            self._df_test = pd.concat(frames, axis=1)
            self._df_test = self._df_test.drop(columns=categorical_indices)
        else:
            # find categorial columns
            cols = self._df_train.columns
            num_cols = self._df_train._get_numeric_data().columns
            categorical_indices = list(set(cols) - set(num_cols))

            # create dummis
            dummies_list = [pd.get_dummies(self._df_train[idx], prefix='{}'.format(idx)) for idx in categorical_indices]
            frames = dummies_list + [self._df_train]
            self._df_train = pd.concat(frames, axis=1)
            self._df_train = self._df_train.drop(columns=categorical_indices)
        return

    def feature_scaling(self, test=False):
        """First we fit the train set and than the test set
           this way they are fit on the same scale and also the test data isn't leaking
        """
        if not test:
            self.sc = StandardScaler()
            self._df_train.iloc[:, :-1] = self.sc.fit_transform(self._df_train.iloc[:, :-1])
        else:
            self._df_test.iloc[:, :] = self.sc.transform(self._df_test)


