import io

import pexpect
import sys
from io import StringIO

def output():
    print("something")
    answer = input("Do you wish to run the custom code? [y/N]")
    print(answer)

def output_test():
    child = pexpect.spawn('python3 -c "from testing_features.Misc.stuff import output; output()"')
    child.expect(r"Do you wish to run the custom code? \[y/N\]")
    child.sendline("y")
    child.expect("y")
    print("Test passed")

def redirect():
    sys.stdin = StringIO('test\ntest')
    print(input("Give me smth:"))
    print(input("More!"))


class as_stdin:
    def __init__(self, buffer):
        self.buffer = buffer
        self.original_stdin = sys.stdin
    def __enter__(self):
        sys.stdin = self.buffer
    def __exit__(self, *exc):
        sys.stdin = self.original_stdin

def echo():
     print(input())
     #print(input())

def wild():
    with as_stdin(io.StringIO("HEYOO")):
        echo()

if __name__ == "__main__":
    redirect()