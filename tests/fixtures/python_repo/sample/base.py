from .helpers import normalize


class Base:
    def clean(self, value):
        return normalize(value)
