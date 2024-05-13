import os

file_path = 'folder/file.txt'

if os.path.exists(file_path):
    print("File exists")
    mode = "a"
else:
    print("File does not exist")
    mode = "w"

with open(file_path, mode) as file:
    file.write("test")