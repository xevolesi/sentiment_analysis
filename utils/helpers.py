def get_ngram(text, n):
    """
    Eng:
    =======================================================================
    :param text: Source text;

    :param n: n parameter for n-gramms;

    :return: ngrams: Str where n-gramms are joined by "|".

    Returns all n-gramms from the source text joined by "|".
    Example:
        Source text: What does the fox say?;
        n parameter: 3;
        Result: What does the|does the fox|the fox say?.
    =======================================================================

    Ru:
    =======================================================================
    :param text: Исходный текст;

    :param n: n параметр для n-грамм;

    :return: ngrams: Строка, в которой все n-граммы разделены символом "|".

    Возвращает все n-граммы, составленные из исходного текста и разделенные
    символом "|".
    Пример:
        Исходный текст: А роза упала на лапу Азора.;
        Параметр n: 3;
        Результат: А роза упала|роза упала на|упала на лапу|на лапу Азора.
    =======================================================================
    """
    ngrams = ""
    if isinstance(text, float):
        return ""
    t = text.split(" ")
    for i in range(len(t) - 2):
        ngrams = "|".join([ngrams, " ".join(t[i:i+n])])
    return ngrams


def get_count(d, key):
    """
    Eng:
    ===============================================================================================
    :param d: Source dictionary;

    :param key: The key on which is needed to find the value in the dictionary d;

    :return: value of d[key] if key in keys of d else 0.

    Returns the value from dictionary key if this key is in keys set of d else returns 0.
    ===============================================================================================

    Ru:
    ===============================================================================================
    :param d: Исходный словарь;

    :param key: Ключ, по которому нужно найти значение в словаре d;

    :return: Значение по исходному ключу из словаря d, или 0, если такого ключа в словаре нет.

    Возвращает значение из исходного словаря d по ключу key, если словарь содержит данный ключ key.
    Иначе, возвращает 0.
    ===============================================================================================
    """
    return d[key] if key in d.keys() else 0


def get_key(d, value):
    """
    Eng:
    ==========================================================================
    :param d: Source dictionary;

    :param value: The value by which to find the key in the dictionary;

    :return: The key of the source dictionary d that corresponds to the value.
    ==========================================================================

    Ru:
    ==========================================================================
    :param d: Исходный словарь;

    :param value: Значение, по которому нужно найти ключ;

    :return: Ключ исходного словаря d, соответствующий значению value.
    ==========================================================================
    """
    for key in d.keys():
        if d[key] == value:
            return key
    return None
