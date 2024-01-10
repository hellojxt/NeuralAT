import subprocess
import sys

for i in range(1500):
    subprocess.run(
        ["python", "experiments/neuPAT_audio/generate.py", str(i + int(sys.argv[1]))]
    )
