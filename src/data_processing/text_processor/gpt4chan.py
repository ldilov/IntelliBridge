import re


class Gpt4Chan(object):
    @classmethod
    def clean(cls, text):
        for i in range(10):
            text = re.sub("--- [0-9]*\n>>[0-9]*\n---", "---", text)
            text = re.sub("--- [0-9]*\n *\n---", "---", text)
            text = re.sub("--- [0-9]*\n\n\n---", "---", text)
        return text
