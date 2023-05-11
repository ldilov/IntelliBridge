class GPT(object):
    @classmethod
    def clean_response(cls, text):
        if type(text) is str:
            text = text.replace(r'<|endoftext|>', '')
        else:
            text = text.decode('utf-8')

        return text