import json
from collections import Counter
from utils.helpers import get_ngram
from utils.TextPreprocessor import TextPreprocessor


class IncorrectLabelError(KeyError):
    """If label is not "pos" or "neg"."""


class Model:
    def __init__(self, label_column_name=None, text_column_name=None, df=None, n=3, laplace_factor=None):
        """
        Eng:
        ====================================================================================================
        :param label_column_name: Name of column in DF where labels are placed;

        :param text_column_name: Name of column in DF where doc's text is places;

        :param df: Source DF with training set;

        :param n: n-parameter for n-grams;

        :param laplace_factor: Model's Laplace factor for Laplace smoothing.

        Model containing numerical information about training data set using Bag of Words model for n-grams.
        ====================================================================================================

        Ru:
        ====================================================================================================
        :param label_column_name: Название столбца в DF, в котором расположены метки;

        :param text_column_name: Название столбца в DF, в котором расположен текст документов;

        :param df: Исходный DF для тренировочного набора данных;

        :param n: Параметр n для n-грамм;

        :param laplace_factor: Множитель Лапласа для сглаживания.

        Модель содержит численную\количественную информацию о тренировочном наборе данных. Эта информация
        используется для построения модели "Мешок слов" применительно к n-граммам.
        ====================================================================================================
        """
        self.n = n if n is not None else 0
        self.lcn = label_column_name if label_column_name is not None else ""
        self.tcn = text_column_name if text_column_name is not None else ""
        self.lp = laplace_factor if laplace_factor is not None else 0
        self.total_msg_count = len(df) if df is not None else 0

        self.posNgrams = dict(
            Counter(ng for ng in "|".join(
                [get_ngram(text, self.n) for text in [
                    row[1][text_column_name] for row in df[df.label == "pos"].iterrows()]]).split("|"))) \
            if df is not None else None
        self.negNgrams = dict(
            Counter(ng for ng in "|".join(
                [get_ngram(text, self.n) for text in [
                    row[1][text_column_name] for row in df[df.label == "neg"].iterrows()]]).split("|"))) \
            if df is not None else None

        self.pos_label_count = int(df[label_column_name].value_counts().at["pos"]) if df is not None else 0
        self.neg_label_count = int(df[label_column_name].value_counts().at["neg"]) if df is not None else 0

        self.pos_Ngram_count = sum(self.posNgrams.values()) if df is not None else 0
        self.neg_Ngram_count = sum(self.negNgrams.values()) if df is not None else 0

        self.unique_Ngram_count = len(
            set.union(set(self.posNgrams.keys()), set(self.negNgrams.keys()))) if df is not None else 0

    def __repr__(self):
        """
        Eng:
        ==================================================================
        :return: Text representation for numeric characteristics of model.
        ==================================================================

        Ru:
        ==================================================================
        :return: Текстовое представление численных характеристик модели.
        ==================================================================
        """
        return """
           Model for NaiveByes Classifier.
           Using {n}-grams.
           Model contains {total_doc} docs: {neg_doc} negative docs, {pos_doc} positive docs.
           Model's Laplace factor = {lp}.
           Total amount of positive {n}-grams is {pngc}.
           Total amount of negative {n}-grams is {nngc}.
           Amount of unique {n}-grams is {ung}.
           """.format(
            n=self.n,
            total_doc=self.total_msg_count,
            neg_doc=self.neg_label_count,
            pos_doc=self.pos_label_count,
            lp=self.lp,
            pngc=self.pos_Ngram_count,
            nngc=self.neg_Ngram_count,
            ung=self.unique_Ngram_count
        )

    def save_model(self, path):
        """
        Eng:
        ================================================================
        :param path: Path to locate saved model file (as json).
        ================================================================

        Ru:
        ================================================================
        :param path: Пусть, в котором будет размещен файл модели (json).
        ================================================================
        """
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self.__dict__, file)

    @staticmethod
    def read_model(path):
        """
        Eng:
        ==========================================
        :param path: Path to model;

        :return: m: Model() object from json file.
        ==========================================

        Ru:
        ==========================================
        :param path: Путь к модели;

        :return: m: Объект Model() из json файла.
        ==========================================
        """
        with open(path) as file:
            m = Model()
            m.__dict__ = json.load(file)
        return m

    def update(self, msg, label, lang):
        """
        Eng:
        ===============================================================================================================
        :param msg: Message with which the model will be updated;

        :param label: Label to detect what kind of message do we have.

        Methods updates the model by recalculation of it's attributes:
            - self.total_msg_count: total amount of all messages;
            - self.posNgrams: dictionary with ngram as key and amount of that ngram as value;
            - self.negNgrams: same as above but negative;
            - self.pos_label_count: total amount of all positive messages;
            - self.neg_label_count: same as above but negative;
            - self.pos_Ngram_count: total amount of all positive n-gramms;
            - self.neg_Ngram_count: same as above but negative;
            - self.unique_Ngram_count: total amount of all unique n-gramms.
        ===============================================================================================================

        Ru:
        ===============================================================================================================
        :param msg: Сообщение, с помощью которого происходит дообучение модели;

        :param label: Метка, необходимая для определения класса, к которому будет соотнесено данное сообщение.

        Метод дообучает модель посредством перерасчета ее атрибутов:
            - self.total_msg_count: общее число всех сообщений\документов;
            - self.posNgrams: словарь, в котором положительные n-граммы являются ключами, а их количество - значениями;
            - self.negNgrams: то же, что и выше, только для негативных n-грамм;
            - self.pos_label_count: общее количество положительных сообщений\документов;
            - self.neg_label_count: общее количество негативных сообщений\документов;
            - self.pos_Ngram_count: общее количество положительных n-грамм в словаре;
            - self.neg_Ngram_count: общее количество негативных n-грамм в словаре;
            - self.unique_Ngram_count: общее количество уникальных n-грамм в словаре.
        ===============================================================================================================
        """

        if label == "pos" or label == "neg":
            self.total_msg_count += 1
            tp = TextPreprocessor(punc="\\r\\n\\$/#^@'=+_:;*-~`)({}[]|<>.,&%!?\'\"",
                                  regexp_lst=["bSubject", "bsubject"], lang=lang)
            prepmsg = get_ngram(tp.full_preprocess(msg), self.n).split("|")

            lbl_ngrams = "".join([label, "Ngrams"])
            lbl_lbl_count = "".join([label, "_label_count"])
            lbl_lbl_ngram_count = "".join([label, "_Ngram_count"])

            t = self.__getattribute__(lbl_lbl_count)
            t += 1
            for ngram in prepmsg:
                if ngram in self.__getattribute__(lbl_ngrams).keys():
                    self.__getattribute__(lbl_ngrams)[ngram] += 1
                else:
                    self.__getattribute__(lbl_ngrams).update({ngram: 1})
                    self.unique_Ngram_count += 1
                t = self.__getattribute__(lbl_lbl_ngram_count)
                t += 1
        else:
            raise IncorrectLabelError("Label {lbl} is incorrect!"
                                      " Label should be \"spam\" or \"ham\"".format(lbl=label))
