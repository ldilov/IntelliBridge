class StreamTokenizer:
    """StreamTokenizer wraps around a tokenizer to support stream decoding."""

    def __init__(self, tokenizer, max_surrogate_tokens=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.replacement = chr(0xFFFD)
        self.buffer = []
        self.surrogates = 0
        self.start = 0
        self.end = 0
        self.max_surrogate_tokens = max_surrogate_tokens

    def decode(self, tokens):
        """Decode token to string while handling surrogates and whitespace."""

        if not isinstance(tokens, list):
            tokens = [tokens]

        if not tokens:
            return ""

        # <unk>, <pad> and other special tokens will be decoded into ''.
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)

        self.buffer = self.buffer[-self.max_surrogate_tokens:] + tokens

        if self.replacement in text:
            n = -self.surrogates if self.surrogates > 0 else len(self.buffer)
            tokens = self.buffer[n:] + [tokens]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)

            # Check whether the last grapheme was successfully decoded.
            if text and text[-1] != self.replacement:
                text = text.replace(self.replacement, "")
                self.surrogates = 0
            else:
                text = ""
                self.surrogates += 1
        else:
            self.surrogates = 0

        # Handle whitespace between tokens.
        tokens = self.buffer + [tokens]
        prefix = self.tokenizer.decode(self.buffer, skip_special_tokens=True)
        whole = self.tokenizer.decode(tokens, skip_special_tokens=True)
        if prefix + " " + text == whole:
            text = " " + text

        # Update buffer and offsets.
        self.buffer = self.buffer[-4:] + [tokens]
        self.start = self.end
        self.end += len(text)

        return text

    @property
    def length(self):
        return len(self.current_text)

    def has_reached_eos(self):
        return self.tokenizer.eos_token in self.current_text
