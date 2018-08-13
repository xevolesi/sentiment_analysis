class InvalidFractionError(ValueError):
    """If sum of fractions for splitter greater than 1."""


class DataSplitter:
    """
    Eng:
    ==================================================================================
    Class for splitting the pandas.DataFrame objects. Provides two types of splitting:
        - Basic splitting:
            - train = 0.8;
            - test = 0.1;
            - valid = 0.1.

        - Normal splitting:
            Used list of fractions for splitting.
    ==================================================================================

    Ru:
    ==================================================================================
    Класс для разделения объектов pandas.DataFrame. Предоставляет два типа разделения:
        - Базовое разделение:
            - тренировочный набор = 0.8;
            - тестовый набор = 0.1;
            - валидационный = 0.1.

        - Нормальное разделение:
            Используется список долей для разделения.
    ==================================================================================
    """
    @staticmethod
    def basic_split(df):
        """
        Eng:
        ========================================================================================================
        :param df: pandas.DataFrame for splitting;

        :return: train, test, valid: Tuple of training, testing and validating DFs.

        The data splits with basic fractions: 0.8 - for train, 0.1 - for test and val.
        ========================================================================================================

        Ru:
        ========================================================================================================
        :param df: Объект pandas.DataFrame;

        :return: train, test, valid: Кортеж, содержащий тренировочный тестовый и валидационный pandas.DataFrame.

        Данные разделяются в стандартных пропорциях: 0.8 - для тренировочного набора, 0.1 - для тестового и
        валидационного наборов.
        ========================================================================================================
        """
        # Training = 0.8
        train = df.sample(frac=0.8, random_state=1)

        # Testing = 0.1
        diff = df.drop(train.index)
        test = diff.sample(int(0.1 * len(df)), random_state=1)

        # Validating = 0.1
        valid = diff.drop(test.index)

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        valid = valid.reset_index(drop=True)

        return train, test, valid

    @staticmethod
    def split(df, fractions):
        """
        Eng:
        =======================================================================
        :param df: pandas.DataFrame which should be splitted;

        :param fractions: Fractions for splitting;

        :return: List of DFs.
        =======================================================================

        Ru:
        =======================================================================
        :param df: pandas.DataFrame объект, который следует разделить;

        :param fractions: Список с долями для разделения;

        :return: Список DF'ов.
        =======================================================================
        """
        if abs(1 - sum(fractions)) > 0.005:
            raise InvalidFractionError("Sum of splitting fractions must be less then 1!")
        dfs = []
        t = df
        for fraction in fractions:
            p = t.sample(frac=fraction, random_state=1)
            dfs.append(p.reset_index(drop=True))
            t = t.drop(p.index)

        return dfs
