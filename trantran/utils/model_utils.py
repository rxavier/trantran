from typing import Optional

from huggingface_hub import HfApi


def get_all_models(
    lang_from: Optional[str] = None, lang_to: Optional[str] = None, only_ids: bool = True
):
    """Search for a Helsinki NLP Opus MT model matching the language pair in the HuggingFace hub.

    Parameters
    ----------
    lang_from :
        Source language or None.
    lang_to : Optional[str], optional
        Destination language or None.
    only_ids : bool, optional
        Whether to return only the model ID or the full `HfApi().list_models()` result.
    """
    api = HfApi()
    matches = api.list_models(search="opus-mt-", author="Helsinki-NLP")
    matches = [match for match in matches if len(match.id.split("-")) == 5]
    if lang_from and lang_to:
        matches = [
            match
            for match in matches
            if match.id.split("-")[-2] == lang_from and match.id.split("-")[-1] == lang_to
        ]
    elif lang_from and not lang_to:
        matches = [match for match in matches if match.id.split("-")[-2] == lang_from]
    elif not lang_from and lang_to:
        matches = [match for match in matches if match.id.split("-")[-1] == lang_to]
    if only_ids:
        matches = [match.id for match in matches]
    return matches
