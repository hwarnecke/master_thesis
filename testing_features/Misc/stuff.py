import datetime

ct = datetime.datetime.now()
print(ct)

def format_datetime(dt):
    # Format the datetime object into a string
    formatted = dt.strftime('%Y-%m-%d %H:%M:%S')
    # Replace whitespaces with underscores and colons with dashes
    formatted = formatted.replace(' ', '_').replace(':', '-')
    return formatted

formatted = format_datetime(ct)
print(formatted)

new_dt = ct.strftime('%Y-%m-%d_%H-%M-%S')
print(new_dt)