import worker


def handle_request(raw):
    return worker.transform(raw)


def via_runtime(callback, raw):
    return callback(raw)
