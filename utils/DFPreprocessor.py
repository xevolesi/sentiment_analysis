import pandas as pd
from joblib import Parallel, delayed
from utils.TextPreprocessor import TextPreprocessor


class ColumnNotFoundError(KeyError):
    """If column not found ..."""


class IncorrectTransformationLabelError(KeyError):
    """If converter dict hasn't got needed keys ..."""


class DFPreprocessor:
    """For parallel using joblib-module is needed."""
    def __init__(self, lang, src_df, converter_dict=None):
        """
        Eng:
        =================================================================================================
        :param src_df: Source DF;

        :param lang: Source language of source texts;

        :param converter_dict: Dictionary which contains mapping of source keys into ["pos", "neg"] keys.

        Example:
            converter_dict = {"ham":"pos", "spam":"neg"}
        =================================================================================================

        Ru:
        =================================================================================================
        :param src_df: Исходный DF;

        :param lang: Язык исходных текстов;

        :param converter_dict: Словарь, содержащий отображение исходных ключей в ключи ["pos", "neg"].

        Например:
            converter_dict = {"ham":"pos", "spam":"neg"}
        =================================================================================================
        """
        self.src = src_df
        self.cd = converter_dict
        self.lang = lang

    def preprocess_text_column(self, columns, puncs=None, regexps=None, n_jobs=8):
        """
        Eng:
        ========================================================================================================
        :param columns: List of columns names for preprocessing;

        :param puncs: String which contains punctuational symbols;

        :param regexps: List of regular expressions for preprocessing;

        :param n_jobs: Number of processors;

        :return: t: New DF containing preprocessed text columns.

        Parallel preprocessing is released by joblib module.
        ========================================================================================================

        Ru:
        ========================================================================================================
        :param columns: Список с именами столбцов, текст в которых нужно предобработать;

        :param puncs: Строка, содержащая пунктуационные символы;

        :param regexps: Список, содержащий регулярные выражение, которые необходимо применить для предобработки;

        :param n_jobs: Число процессов;

        :return: t: Новый DF с предобработанными столбцами из списка columns.

        Реализована параллельная предобработка данных с помощью модуля joblib.
        ========================================================================================================
        """
        for cn in columns:
            if cn not in self.src.keys():
                raise ColumnNotFoundError("Column with name \"{cname}\" not found!".format(cname=cn))

        # No need to be preprocessed
        t = pd.DataFrame()
        for key in self.src.keys():
            if key not in columns:
                t[key] = self.src[key]

        # Parallel preprocessing
        if self.lang == "ru":
            tp = TextPreprocessor(lang="ru", punc=puncs, regexp_lst=regexps)
        else:
            tp = TextPreprocessor(lang="eng", punc=puncs, regexp_lst=regexps)
        for cn in columns:
            t[cn] = Parallel(n_jobs=n_jobs)(delayed(tp.full_preprocess)(text) for text in self.src[cn])

        t.drop_duplicates()
        t = t.sample(frac=1).reset_index(drop=True)
        return t

    def preprocess_label_column(self, lbl_column, old_pos_label, old_neg_label):
        """
        Eng:
        ====================================================================================
        :param dfs: Source DF;

        :param lbl_column: Name of source label column in dfs;

        :param old_pos_label: Name of old positive label;

        :param old_neg_label: Name of old negative label;

        :return: t: New DF with renamed labels in lbl_column.

        Create new DF with new labels. Old labels are replaced by "pos" and "neg" labels.
        ====================================================================================

        Ru:
        ====================================================================================
        :param dfs: Исходный DF;

        :param lbl_column: Имя столбца, содержащего исходные метки в dfs;

        :param old_pos_label: Имя старого положительной метки;

        :param old_neg_label: Имя старой отрицательной метки;

        :return: t: Новый DF с переименнованными метками в lbl_column.

        Создает новый DF с новыми метками. Старые метки заменяется на метки "pos" and "neg".
        ====================================================================================
        """
        if any(label not in self.cd.keys() for label in [old_neg_label, old_pos_label]):
            raise IncorrectTransformationLabelError("Label {opl} or {onl} is incorrect!".format(opl=old_pos_label,
                                                                                                onl=old_neg_label))
        t = pd.DataFrame()

        t[lbl_column] = [self.cd[label] for label in self.src[lbl_column]]

        for key in self.src.keys():
            if key != lbl_column:
                t[key] = self.src[key]

        return t

    def partial_preprocess_text_column(self, columns, puncs=None, methods=None, regexps=None, n_jobs=8):
        """
        Eng:
        ========================================================================================================
        :param columns: List of columns names for preprocessing;

        :param puncs: String which contains punctuational symbols;

        :param methods: List of methods names for partial preprocess;

        :param regexps: List of regular expressions for preprocessing;

        :param n_jobs: Number of processors;

        :return: t: New DF containing preprocessed text columns.

        Parallel preprocessing is released by joblib module.
        ========================================================================================================

        Ru:
        ========================================================================================================
        :param columns: Список с именами столбцов, текст в которых нужно предобработать;

        :param puncs: Строка, содержащая пунктуационные символы;

        :param methods: Список с именами методов для частичной предобработки;

        :param regexps: Список, содержащий регулярные выражение, которые необходимо применить для предобработки;

        :param n_jobs: Число процессов;

        :return: t: Новый DF с предобработанными столбцами из списка columns.

        Реализована параллельная предобработка данных с помощью модуля joblib.
        ========================================================================================================
        """
        for cn in columns:
            if cn not in self.src.keys():
                raise ColumnNotFoundError("Column with name \"{cname}\" not found!".format(cname=cn))

        # No need to be preprocessed
        t = pd.DataFrame()
        for key in self.src.keys():
            if key not in columns:
                t[key] = self.src[key]

        # Parallel preprocessing
        if self.lang == "ru":
            tp = TextPreprocessor(lang="ru", punc=puncs, regexp_lst=regexps, part_methods=methods)
        else:
            tp = TextPreprocessor(lang="eng", punc=puncs, regexp_lst=regexps, part_methods=methods)
        for cn in columns:
            t[cn] = Parallel(n_jobs=n_jobs)(delayed(tp.partial_preprocess)(text) for text in self.src[cn])

        t.drop_duplicates()
        t = t.sample(frac=1).reset_index(drop=True)
        return t
