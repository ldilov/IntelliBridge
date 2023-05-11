import re

import torch
from transformers import PreTrainedTokenizer


class StreamTokenizer:
    def __init__(self, tokenizer):
        super().__init__()

        self.history = ""
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.replacement = chr(0xFFFD)
        self.buffer = []
        self.surrogates = 0
        self.start = 0
        self.end = 0

    def _decode_token_to_text(self, token):
        if not token:
            return ""

        if type(token) == list:
            token = torch.tensor(token).to('cuda')

        if token.dim() > 1:
            while token.dim() > 1:
                token = torch.squeeze(token, dim=-1)

        text = self.tokenizer.decode(token, skip_special_tokens=True)

        return text

    def _is_replacement_present(self, text):
        return self.replacement in text

    def _is_list_of_tensors(self, lst, tensor_class=torch.Tensor):
        return isinstance(lst, list) and all(isinstance(item, tensor_class) for item in lst)

    def _is_nested_list(self, element):
        return isinstance(element, list) and any(isinstance(item, list) for item in element)

    def _flatten(self, lst):
        return [item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])]

    def _parse_token(self, token):
        if isinstance(token, str):
            return self.tokenizer.encode(token, add_special_tokens=False)
        elif isinstance(token, int):
            return [token]
        elif isinstance(token, torch.Tensor):
            if token.ndim == 1:
                return token.tolist()
            else:
                return self._parse_token(token.squeeze(0))

        return token

    def remove_non_printable_and_non_ascii(self, text, replacement=""):
        return re.sub(r'[^\x20-\x7E\u0400-\u04FF]', replacement, text)

    def _handle_replacement_characters(self, token, text):
        negative_surrogates = -self.surrogates if self.surrogates > 0 else len(self.buffer)
        tokens = self.buffer[negative_surrogates:] + [token]
        text = self._decode_token_to_text(tokens)
        text = self.remove_non_printable_and_non_ascii(text, replacement=self.replacement)

        if text and text[-1] != self.replacement:
            text = text.replace(self.replacement, "")
            self.surrogates = 0
        else:
            text = ""
            self.surrogates += 1

        return text

    def _handle_whitespace(self, token, text):
        tokens = self.buffer + [token]
        whole = self._decode_token_to_text(tokens)
        prefix = self._decode_token_to_text(self.buffer)

        if prefix + " " + text == whole:
            text = " " + text

        return text

    def _remove_html_tags(self, text: str) -> str:
        # Define a regular expression pattern to match HTML tags
        html_tag_pattern = re.compile('<.*?>')

        # Use re.sub() to remove HTML tags from the text
        cleaned_text = re.sub(html_tag_pattern, '', text)

        return cleaned_text

    def add_to_history(self, text):
        self.history += text

    def _update_buffer_and_offsets(self, token, text):
        self.buffer = self.buffer[-10:] + [token]
        self.start = self.end
        self.end += len(text)

    def decode(self, token):
        """Decode token to string while handling surrogates and whitespace."""

        text = self._decode_token_to_text(token)

        if self._is_replacement_present(text):
            text = self._handle_replacement_characters(token, text)
        else:
            self.surrogates = 0

        text = self._handle_whitespace(token, text)
        self._update_buffer_and_offsets(token, text)
        text = self._remove_html_tags(text)

        return text
