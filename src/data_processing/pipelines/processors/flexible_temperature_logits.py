from transformers import LogitsProcessor


class FlexibleTemperatureLogitsProcessor(LogitsProcessor):
    def __init__(self, initial_temperature=1.0, final_temperature=1.5, transition_length=50):
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.transition_length = transition_length
        self.gen_length = 0
        assert self.initial_temperature > 0, "initial_temperature should be greater than 0"
        assert self.final_temperature > 0, "final_temperature should be greater than 0"
        assert self.transition_length > 0, "transition_length should be greater than 0"

    def __call__(self, input_ids, scores):
        self.gen_length += 1
        if self.gen_length < self.transition_length:
            temperature = self.initial_temperature
        else:
            temperature = ((self.final_temperature - self.initial_temperature) / (1 - self.transition_length)) * (
                    self.gen_length - 1) + self.initial_temperature

        scores = scores / temperature
        return scores
