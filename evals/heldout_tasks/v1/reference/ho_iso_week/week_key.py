def week_key(day):
    year, week, _ = day.isocalendar()
    return f"{year:04d}-W{week:02d}"
