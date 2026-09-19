from base import Base


def transform(raw):
    return raw.strip().casefold()


class Worker(Base):
    def run(self, raw):
        return transform(raw)
