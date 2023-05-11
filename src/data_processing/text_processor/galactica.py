import re


class Galactica(object):
    @classmethod
    def clean(cls, text):
        text = text.replace(r'\[', r'$')
        text = text.replace(r'\]', r'$')
        text = text.replace(r'\(', r'$')
        text = text.replace(r'\)', r'$')
        text = text.replace(r'$$', r'$')
        text = re.sub(r'\n', r'\n\n', text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text