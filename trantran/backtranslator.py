from typing import Sequence, Optional

from trantran import Translator


class BackTranslator:
    """Backtranslate sequences of strings from a language and back.

    This is meant as an augmentation technique for NLP tasks, since backtranslation will rarely
    return the exact same string.
    """

    def __init__(
        self,
        lang_from: str,
        lang_to: str,
        to_multi_prefix: Optional[str] = None,
        from_multi_prefix: Optional[str] = None,
    ):
        """Initialize the BackTranslator. Use :func:`trantran.utils.model_utils.get_all_models` to
        get all available models and language pairs.

        It will search for the language pair on the hub. If it is not found and one of the
        languages is English, it will switch the remaining one to multilingual ('mul'). If it is
        not found and none of the languages is English, both source and destination are switched to
        'mul' and intermediate translation is be used (translating source to English, and English
        to destination).

        The same process is used for the reverse translation, i.e., if `lang_from` is 'es' and
        `lang_to` is 'pt', the reverse translation will be from 'pt' to 'es'.

        `to_multi_prefix` is required when the destination language is multilingual.
        `from_multi_prefix` is required when the source language is multilingual, which will be the
        destination during reverse translation.

        For example, if translating from 'pt' to 'es', since there is no such model in the hub,
        intermediate translation will be used. The full process is
        'pt' -> 'en' -> 'es' -> 'en' -> 'pt', which in actual models will be
        'mul' -> 'en' -> 'mul' -> 'en' -> 'mul', which would require `to_multi_prefix=='>>por<<'`
        and `from_multi_prefix=='>>spa<<'`.

        Parameters
        ----------
        lang_from :
            Source language (e.g. "en", see `examples <https://developers.google.com/admin-sdk/directory/v1/languages>`_).
        lang_to : str
            Destination language (e.g. "es", see `examples <https://developers.google.com/admin-sdk/directory/v1/languages>`_).
        to_multi_prefix :
            In the format '>>lng<<' or None. This prefix instructs the model what the destination
            language should be when the destination is multilingual, which can happen when a model
            for the selected language pair is not available.
        from_multi_prefix :
            In the format '>>lng<<' or None. This prefix instructs the model what the destination
            language should be when the source is multilingual, which can happen when a model
            for the selected language pair is not available.
        """
        self.translator = Translator(
            lang_from=lang_from, lang_to=lang_to, multi_prefix=to_multi_prefix
        )
        self.reverse_translator = Translator(
            lang_from=lang_to, lang_to=lang_from, multi_prefix=from_multi_prefix
        )

    def backtranslate(self, texts: Sequence[str]) -> Sequence[str]:
        """Translate a sequence of strings from source to destination language and back.

        Parameters
        ----------
        texts :
            Sequence of strings in source language.

        Returns
        -------
        Sequence[str]
            Sequence of strings in source language, obtained from translating to destination
            language and back.
        """
        translated = self.translator.translate(texts)
        return self.reverse_translator.translate(translated)
