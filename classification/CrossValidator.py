from pandas import read_csv
from classification.Model import Model
from classification.Tester import Tester
from classification.Classifier import Classifier


class CrossValidator:
    def __init__(self, ngrams=None, lpfs=None, lang=None):
        self.lpfs = lpfs
        self.ngrams = ngrams
        self.lang = lang

    def validate(self, train_data_path, validation_data_path):
        cv_errs = []
        val_df = read_csv(validation_data_path, index_col=0)

        models = []
        print("============================================================")
        for ngram in self.ngrams:
            for lp in self.lpfs:
                print("Creating model M({lf}, {n}) ...".format(lf=lp, n=ngram))
                model = Model(
                    "label",
                    "text",
                    read_csv(train_data_path, index_col=0),
                    n=ngram,
                    laplace_factor=lp)
                print("Model successfully created!")

                models.append(model)

                print("Starting validation for M({lf}, {n}) ...".format(lf=lp, n=ngram))
                classifier = Classifier(model, lang=self.lang)
                t = Tester.test(classifier, val_df)
                cv_errs.append((t, lp, ngram))
                print("Validation successfully complete!")

                print("Result of validation: E(M({lf}, {n})) = {ce}".format(lf=lp, n=ngram, ce=t))
                print("============================================================")
                print()
        print("Best model is M({lf}, {n}) = ".format(lf=min(cv_errs)[1], n=min(cv_errs)[2]))
        print(models[cv_errs.index(min(cv_errs))])
        return models[cv_errs.index(min(cv_errs))], min(cv_errs)[1], min(cv_errs)[2]

    def validate_for_stat_with_methods(self, path):
        cv_errs = []
        x = []
        lps = []
        i = 0
        print("============================================================")
        for train, val in path:
            for ngram in self.ngrams:
                for lp in self.lpfs:
                    val_df = read_csv(val, index_col=0)
                    print("Creating model M({lf}, {n}) ...".format(lf=lp, n=ngram))
                    model = Model(
                        "label",
                        "text",
                        read_csv(train, index_col=0),
                        n=ngram,
                        laplace_factor=lp)
                    print("Model successfully created!")
                    x.append((ngram, lp, i))
                    lps.append(lp)
                    print("Starting validation for M({lf}, {n}) ...".format(lf=lp, n=ngram))
                    classifier = Classifier(model, lang=self.lang)
                    t = Tester.test(classifier, val_df)
                    cv_errs.append(t)
                    print("Validation successfully complete!")

                    print("Result of validation: E(M({lf}, {n})) = {ce}".format(lf=lp, n=ngram, ce=t))
                    print("============================================================")
                    print()
            i += 1
        return cv_errs, x, lps
