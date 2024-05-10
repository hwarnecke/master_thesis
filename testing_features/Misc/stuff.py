value = "ashdashd"

lines = value.split("\n")
query_str = lines[0].split(": ")[1]
filters = lines[1].split(": ")[1]

print("Query: " + query_str)
print("filter:" + filters)