def week_key(day):
    week = day.isocalendar().week
    return f"{day.year}-W{week:02d}"
