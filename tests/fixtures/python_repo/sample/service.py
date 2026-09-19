import json
from .base import Base
from . import helpers
from .helpers import normalize as clean


class Service(Base):
    def run(self, value):
        cleaned = clean(value)
        self.clean(cleaned)
        return json.dumps(cleaned)


def pipeline(value):
    def finish(item):
        return helpers.decorate(item)

    return finish(clean(value))
