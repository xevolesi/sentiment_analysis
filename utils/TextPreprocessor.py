from functools import reduce
import re
from pymystem3 import Mystem
from nltk.corpus import stopwords
from nltk.stem.snowball import RussianStemmer
from nltk import PorterStemmer, WordNetLemmatizer


class IncorrectLanguageError(ValueError):
    """If source language is not english or russian."""


class TextPreprocessor:
    """
    Eng:
    ===========================================================================================================
    Class contains methods for text preprocessing.

    Methods:
        - Stop-words removing;
        - Removing punctuation symbols;
        - Replacing all numbers with 1;
        - Stemming;
        - Lemmatizing;
        - Removing special syntax constructions using regular expressions.

    Additional, contains two special methods:
        - Full preprocess method, which using all implemented methods for preprocessing with prefix "prep" in
        it's names;
        - Partial preprocessing method:
            The method signature refines the list with the names of the methods to be applied.
    ===========================================================================================================

    Ru:
    ===========================================================================================================
    Класс содержит методы для предобработки текста.

    Методы:
        - Удаление стоп-слов (слова, часто встречающиеся в языковых конструкциях, не несущие особой смысловой
         нагрузки);
        - Удаление пунктуационных символов;
        - Замена всех вхождений чисел на 1;
        - Стемминг;
        - Лемматизация;
        - Удаление специальных синтаксических конструкций с помощью регулярных выражений.

    Дополнтельно, содержит два метода:
        - Метод полной предобработки full_preprocess, в которой применяются все доступные методы предобработки;
        - Метод множественной частичной предобработки partial_preprocess:
            В параметрах метода указывается список с названиями методов преобработки, которые нужно применить.

    Замечание:
        Если дописать какой-нибудь метод преобработки, дав ему название с префиксом "prep",
        то метод full_preprocess автоматически подхватит его для применения.
    ===========================================================================================================
    """
    def __init__(self, lang, punc=None, regexp_lst=None, part_methods=None):
        """
        Eng:
        =======================================================================================================
        :param lang: Source language of source texts;

        :param punc: String contained bad symbols which should be removed
                     (prep_delete_punctuation_symbols method);

        :param regexp_lst: List of regular expressions for prep_re_sub() method;

        :param part_methods: List with the names of the methods which should be used for
                             partial preprocessing.
        =======================================================================================================

        Ru:
        =======================================================================================================
        :param lang: Язык исходных текстов;

        :param punc: Строка, содержащая символы, которые нужно удалить (Метод prep_delete_punctuation_symbols);

        :param regexp_lst: Список, содержащий регулярные выражения для метода prep_re_sub();

        :param part_methods: Список с именами методов, которые нужно применить для метода множественной
                             частичной предобработки.
        =======================================================================================================
        """
        if lang != "ru" and lang != "eng":
            raise IncorrectLanguageError("Source language must be \"eng\" or \"ru\", not \"{lang}\"".format(lang=lang))

        self.lang = lang
        self.punct_string = punc
        self.methods = part_methods
        self.rel = regexp_lst

    def prep_re_sub(self, text):
        """
        Eng:
        ==================================================================================
        :param text: Text for preprocessing;

        :return: Preprocessed text (without expressions matching with regexp in self.rel).

        For removing expression matching with regexp used re-module.
        ==================================================================================

        Ru:
        ==================================================================================
        :param text: Текст для предобработки;

        :return: Текст без выражений, соответствующих паттернам в self.re.

        Для удаления выражений, подходящих под паттерн, используется модуль re.
        ==================================================================================
        """
        if self.rel is None:
            return text
        t = text
        for regexp in self.rel:
            t = re.sub(regexp, "", t)
        return t

    def prep_delete_stop_words(self, text):
        """
        Eng:
        ==================================================================
        :param text: Text for preprocessing;

        :return: Preprocessed text (without stop-words).

        For removing stop-words used nltk.corpus.stopwords dictionary.
        ==================================================================

        Ru:
        ==================================================================
        :param text: Текст для предобработки;

        :return: Обработанный текст без стоп-слов.

        Для удаления стоп-слов используется nltk.corpus.stopwords словарь.
        ==================================================================
        """
        if isinstance(text, str):
            if self.lang == "ru":
                return " ".join([word for word in text.split() if word not in stopwords.words("russian")])
            return " ".join([word for word in text.split() if word not in stopwords.words("english")])
        else:
            print(type(text))
            raise TypeError("Argument must be str!")

    def prep_delete_punctuation_symbols(self, text):
        """
        Eng:
        =======================================================================
        :param text: Text for preprocessing;

        :return: Preprocessed text (without symbols from self.punct_string).

        Removes symbols from punct_string class atributes.
        =======================================================================

        Ru:
        =======================================================================
        :param text: Текст для предобработки;

        :return: Обработанный текст без символов из атрибута self.punct_string.

        Удаляет "плохие" символы, указанные в атрибуте punct_string.
        =======================================================================
        """
        if isinstance(text, str):
            if self.punct_string is not None:
                return "".join([symbol for symbol in text if symbol not in self.punct_string])
        else:
            print(type(text))
            raise TypeError("Argument must be str!")

    @staticmethod
    def prep_replace_digits(text):
        """
        Eng:
        ============================================================
        :param text: Text for preprocessing;

        :return: Preprocessed text with all numbers replaced with 1.

        Replacing all numbers with 1.
        ============================================================

        Ru:
        ============================================================
        :param text: Текст для предобработки;

        :return: Обработанный текст, в котором все числа заменены 1.

        Заменяет все числа в тексте на 1.
        ============================================================
        """
        if isinstance(text, str):
            return re.sub(r"[0-9]+", "1", text)
        else:
            raise TypeError("Argument must be str!")

    def prep_stem(self, text):
        """
        Eng:
        ============================================================================
        :param text: Text for preprocessing;

        :return: Preprocessed text with all stemmed words.

        Stem all words with Porter stemmer.
        ============================================================================

        Ru:
        ============================================================================
        :param text: Текст для предобработки;

        :return: Обработанный текст, в котором каждое слово подвергнулось стеммингу.

        Стеммингует все слова с помощью стеммера Портера.
        ============================================================================
        """
        if isinstance(text, str):
            if self.lang == "ru":
                return " ".join([RussianStemmer().stem(word) for word in text.split()])
            return " ".join([PorterStemmer().stem(word) for word in text.split()])
        else:
            raise TypeError("Argument must be str!")

    def prep_lemmatize(self, text):
        """
        Eng:
        ===============================================================================
        :param text: Text for preprocessing;

        :return: Preprocessed text with all lemmatized words.

        Lemmatize all words with WordNet Lemmatizer.
        ===============================================================================

        Ru:
        ===============================================================================
        :param text: Текст для предобработки;

        :return: Обработанный текст, в котором каждое слово подвергнулось лемматизации.

        Лемматизирует все слова с помощью WordNet лемматизатора.
        ===============================================================================
        """
        if isinstance(text, str):
            if self.lang == "ru":
                return "".join(Mystem().lemmatize(text))
            return " ".join([WordNetLemmatizer().lemmatize(word) for word in text.split()])
        else:
            raise TypeError("Argument must be str!")

    def partial_preprocess(self, text):
        """
        Eng:
        =======================================================================================
        :param text: Text for preprocessing;

        :return: Preprocessed text with using all methods in self.methods attribute.

        Applies all preprocessing methods with names in self.methods to :param text.
        =======================================================================================

        Ru:
        =======================================================================================
        :param text: Текст для предобработки;

        :return: Обработанный с помощью всех методов, указанных в атрибуте self.methods, текст.

        Применяет все методы предобработки, указанные в в атрибуте self.methods к :param text.
        =======================================================================================
        """
        if isinstance(text, str):
            if all((method in list(dir(self))) and (method.startswith("prep_")) for method in self.methods):
                mtds = [self.__getattribute__(method) for method in self.methods]
                return (lambda y: reduce(lambda f, g: g(f), mtds, y))(text)
            else:
                raise NameError("Incorrect method names")
        else:
            raise TypeError("Argument \"text\" must be str!")

    def full_preprocess(self, text):
        """
        Eng:
        =================================================================
        :param text: Text for preprocessing;

        :return: Preprocessed text using all preprocessing methods.

        Applies all preprocessing methods to :param text.
        =================================================================

        Ru:
        =================================================================
        :param text: Текст для предобработки;

        :return: Обработанный с помощью всех методов предобработки текст.

        Применяет все методы предобработки к :param text.
        =================================================================
        """
        if isinstance(text, str):
            methods = [self.__getattribute__(method)
                       for method in [smethod for smethod in list(dir(self)) if smethod.startswith("prep_")]]
            return (lambda y: reduce(lambda f, g: g(f), methods, y))(text)
        else:
            raise TypeError("Argument must be str!")
