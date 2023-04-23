from typing import List, Optional

from utils.streaming.helpers.models.choice import Choice
from utils.streaming.helpers.models.log_probs import LogProbs


class ChoiceManager:
    @staticmethod
    def map_choice(
            text: str,
            index: int,
            token: Optional[int] = None,
            token_logprob: Optional[float] = None,
            top_logprobs: Optional[List[float]] = None,
            text_offset: Optional[int] = None,
            finish_reason: Optional[str] = None,
    ) -> Choice:
        """Create a choice object from model outputs."""
        if token is not None and token_logprob is not None and top_logprobs is not None and text_offset is not None:
            logprobs = LogProbs(
                tokens=[token],
                token_logprobs=[token_logprob],
                top_logprobs=[top_logprobs],
                text_offset=[text_offset],
            )
        else:
            logprobs = None

        return Choice(text=text, index=index, logprobs=logprobs, finish_reason=finish_reason)

    @staticmethod
    def merge_choices(choices: List[Choice]) -> Choice:
        """Merge a list of choices into a single choice object."""
        buffer = []
        index = 0
        finish_reason = None
        tokens = []
        token_logprobs = []
        top_logprobs = []
        text_offset = []

        for choice in choices:
            buffer.append(choice.text)
            index = choice.index
            finish_reason = choice.finish_reason

            if choice.logprobs is not None:
                tokens += choice.logprobs.tokens
                token_logprobs += choice.logprobs.token_logprobs
                top_logprobs += choice.logprobs.top_logprobs
                text_offset += choice.logprobs.text_offset

        logprobs = LogProbs(tokens=tokens, token_logprobs=token_logprobs, top_logprobs=top_logprobs, text_offset=text_offset) if tokens else None

        return Choice(text="".join(buffer), index=index, logprobs=logprobs, finish_reason=finish_reason)