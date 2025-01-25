import os
import subprocess
import re

singular_commands = """
LIB "finvar.lib";
ring F = 0, (x, y), dp;
matrix R[4][4] = 0,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0;
matrix D[4][4] = 0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0;
matrix L[4][4] = 0,1,0,0,1,0,0,0,0,0,0,1,0,0,1,0;
print(R);
"""

result = subprocess.run(
    ["C:/cygwin64/bin/bash.exe", "-c", "Singular"],
    input=singular_commands,
    capture_output=True,
    text=True
)

print("Output (stdout):")
print(result.stdout)
print("Error (stderr):")
print(result.stderr)
