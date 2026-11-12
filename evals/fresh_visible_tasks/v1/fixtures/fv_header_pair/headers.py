def parse_header(text):
    name, value = text.split(":", 1)
    return name.strip(), value.strip()
