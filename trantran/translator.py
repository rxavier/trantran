from typing import Sequence, Optional, Tuple
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import HfApi


class Translator:
    """Translate sequences of strings from a source language to a destination language.

    Translator uses Helsinki NLP's Opus MT models available in the HuggingFace hub to translate
    text.
    """

    def __init__(
        self,
        lang_from: str,
        lang_to: str,
        multi_prefix: Optional[str] = None,
        half_precision: bool = True,
    ):
        """Initialize the Translator. Use :func:`trantran.utils.model_utils.get_all_models` to
        get all available models and language pairs.

        It will search for the language pair on the hub. If it is not found and one of the
        languages is English, it will switch the remaining one to multilingual ('mul'). If it is
        not found and none of the languages is English, both source and destination are switched to
        'mul' and intermediate translation is be used (translating source to English, and English
        to destination).

        Parameters
        ----------
        lang_from :
            Source language (e.g. "en", see `examples <https://developers.google.com/admin-sdk/directory/v1/languages>`_).
        lang_to : str
            Destination language (e.g. "es", see `examples <https://developers.google.com/admin-sdk/directory/v1/languages>`_).
        multi_prefix :
            In the format '>>lng<<' or None. This prefix instructs the model what the destination
            language should be when the destination is multilingual, which can happen when a model
            for the selected language pair is not available.
        half_precision :
            Whether to use fp16 to reduce memory usage, default True.
        """
        self.lang_from = lang_from
        self.lang_to = lang_to
        self.multi_prefix = multi_prefix
        self.half_precision = half_precision

        self.device = self._get_device()
        self.BASE_URL = "Helsinki-NLP/opus-mt-"
        self._from, self._to, self.intermediate = self._validate_model_exists()

        if self.intermediate:
            self.first_tokenizer, self.first_model = self._build(self._from, "en")
            self.second_tokenizer, self.second_model = self._build("en", self._to)
        else:
            self.tokenizer, self.model = self._build(self._from, self._to)

    @staticmethod
    def _get_device() -> torch.device:
        """Set `torch.device` to MPS, CUDA or CPU."""
        try:
            if torch.backends.mps.is_available() is True:
                device = "mps"
        except AttributeError:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    def _validate_model_exists(self) -> Tuple[str, str, bool]:
        """Search for the language pair in the HuggingFace hub."""
        api = HfApi()
        model_id = f"opus-mt-{self.lang_from}-{self.lang_to}"
        matches = api.list_models(search=model_id, author="Helsinki-NLP")

        if len(matches) == 1:
            return self.lang_from, self.lang_to, False

        elif len(matches) == 0:
            if self.lang_from == "en":
                warnings.warn(
                    """No model found for language pair. Setting destination language
                       to 'mul'. `multi_prefix` needs to be set to the desired destination
                       language.""",
                    stacklevel=2,
                )
                # TODO: add a method to show available prefixes
                return self.lang_from, "mul", False
            elif self.lang_to == "en":
                warnings.warn(
                    "No model found for language pair. Setting source language to 'mul'.",
                    stacklevel=2,
                )
                return "mul", self.lang_to, False
            else:
                warnings.warn(
                    """"No model found for language pair. Setting source and destination
                              languages to 'mul' and switching to intermediate translation.""",
                    stacklevel=2,
                )
                return "mul", "mul", True
        else:
            raise ValueError("More than one model found for this language pair.")

    def _build(self, lang_from, lang_to) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
        """Build tokenizer and model from a language pair."""
        base_url = "Helsinki-NLP/opus-mt-"
        url = f"{base_url}{lang_from}-{lang_to}"
        tokenizer = AutoTokenizer.from_pretrained(url)
        model = AutoModelForSeq2SeqLM.from_pretrained(url).to(self.device)
        if self.half_precision:
            model = model.half()
        return tokenizer, model

    def _translation_loop(
        self,
        texts: Sequence[str],
        tokenizer: AutoTokenizer,
        model: AutoModelForSeq2SeqLM,
    ) -> Sequence[str]:
        """Tokenize and encode a sequence of texts, translate it and decode back to text."""
        tokens = tokenizer(
            texts, return_tensors="pt", padding=True, max_length=512, truncation=True
        ).to(self.device)
        translated_generated = model.generate(**tokens)
        translated = [
            tokenizer.decode(t, skip_special_tokens=True) for t in translated_generated
        ]
        return translated

    def _translate(
        self,
        texts: Sequence[str],
        tokenizer: AutoTokenizer,
        model: AutoModelForSeq2SeqLM,
    ) -> Sequence[str]:
        """Translate a sequence of texts, adding prefix if necessary."""
        if self._to == "mul":
            texts = [f"{self.multi_prefix} {text}" for text in texts]
        translated = self._translation_loop(texts, tokenizer, model)
        if self.device.cuda.type == "cuda":
            torch.cuda.empty_cache()
        return translated

    def _intermediate_translate(
        self,
        texts: Sequence[str],
        first_tokenizer: AutoTokenizer,
        first_model: AutoModelForSeq2SeqLM,
        second_tokenizer: AutoTokenizer,
        second_model: AutoModelForSeq2SeqLM,
    ) -> Sequence[str]:
        """Translate a sequence of texts using intermediate translation with English,
        adding prefix if necessary."""
        intermediate_translated = self._translation_loop(
            texts, first_tokenizer, first_model
        )
        if self._to == "mul":
            intermediate_translated = [
                f"{self.multi_prefix} {text}" for text in intermediate_translated
            ]
        translated = self._translation_loop(
            intermediate_translated, second_tokenizer, second_model
        )
        if self.device.cuda.type == "cuda":
            torch.cuda.empty_cache()
        return translated

    def translate(self, texts: Sequence[str]) -> Sequence[str]:
        """Translate a sequence of strings from source to destination language.

        Parameters
        ----------
        texts :
            Sequence of strings in source language.

        Returns
        -------
        Sequence[str]
            Sequence of strings in destination language.
        """
        if self.intermediate:
            return self._intermediate_translate(
                texts,
                self.first_tokenizer,
                self.first_model,
                self.second_tokenizer,
                self.second_model,
            )
        else:
            return self._translate(texts, self.tokenizer, self.model)
