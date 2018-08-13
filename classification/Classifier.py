from math import log
import pandas as pd
from joblib import Parallel, delayed
from utils.helpers import get_ngram, get_count
from utils.TextPreprocessor import TextPreprocessor


class IncorrectPreprocessMethodError(ValueError):
    """If preprocess method is not "full" or "partial"."""


class Classifier:
    def __init__(self, model, lang):
        self.model = model
        self.lang = lang

    def classify_text(self, text, preprocess=None, punc=None, regexp_lst=None, methods=None):
        # if (preprocess != "full") or (preprocess != "partial"):
        #     raise IncorrectPreprocessMethodError(
        #         "Method \"{prep}\" is incorrect. Correct method is \"full\" or \"partial\"")
        prepmsg = ""
        if preprocess == "full":
            tp = TextPreprocessor(punc=punc, regexp_lst=regexp_lst, lang=self.lang)
            prepmsg = get_ngram((tp.full_preprocess(text)), self.model.n)
        if preprocess == "partial":
            tp = TextPreprocessor(punc=punc, regexp_lst=regexp_lst, part_methods=methods, lang=self.lang)
            prepmsg = get_ngram(tp.partial_preprocess(text), self.model.n)
        if preprocess is None:
            prepmsg = get_ngram(text, self.model.n)
        try:
            pos = log(self.model.pos_label_count /
                      (self.model.pos_label_count + self.model.neg_label_count))

            neg = log(self.model.neg_label_count /
                      (self.model.pos_label_count + self.model.neg_label_count))

            for ngram in prepmsg.split("|"):
                pos += log((get_count(self.model.posNgrams, ngram) + self.model.lp) /
                           (self.model.pos_Ngram_count + self.model.lp * self.model.unique_Ngram_count))

                neg += log((get_count(self.model.negNgrams, ngram) + self.model.lp) /
                           (self.model.neg_Ngram_count + self.model.lp * self.model.unique_Ngram_count))

            if pos > neg:
                return "pos"
            else:
                return "neg"
        except ValueError:
            return None

    def batch_classify(self, src_csv_path, text_column, label_column, dst_csv_path=None,
                       preprocess=None, punc=None, regexp_lst=None, methods=None, n_jobs=8):

        df = pd.read_csv(src_csv_path, index_col=0)
        t = pd.DataFrame()

        for key in df.keys():
            if key != text_column:
                t[key] = df[key]

        if preprocess == "full":
            t[label_column] = Parallel(n_jobs=n_jobs)(delayed(
                self.classify_text)(text, "full", punc, regexp_lst) for text in df[text_column])
        if preprocess == "partial":
            t[label_column] = Parallel(n_jobs=n_jobs)(delayed(
                self.classify_text)(text, "partial", punc, regexp_lst, methods) for text in df[text_column])
        if preprocess is None:
            t[label_column] = Parallel(n_jobs=n_jobs)(delayed(
                self.classify_text)(text) for text in df[text_column])

        if dst_csv_path is not None:
            t.to_csv(dst_csv_path)
        return t
