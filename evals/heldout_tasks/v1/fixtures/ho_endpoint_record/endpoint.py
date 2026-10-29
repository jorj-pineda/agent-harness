def parse_endpoint(text):
    host, port = text.rsplit(":", 1)
    return host, int(port)
