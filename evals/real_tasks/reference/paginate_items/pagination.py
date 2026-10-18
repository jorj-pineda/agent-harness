def first_page(values: list, size: int) -> list:
    return values[:size]


def page_items(values: list, page: int, page_size: int) -> list:
    if type(page) is not int or page <= 0 or type(page_size) is not int or page_size <= 0:
        raise ValueError("page and page_size must be positive integers")
    start = (page - 1) * page_size
    return values[start : start + page_size]
