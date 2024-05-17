import sys
import subprocess

subprocess.run(["python", "experiments/neuPAT_material_scale/test.py"])
subprocess.run(["python", "NeuralSound/tetra2hex.py", "dataset/NeuPAT_new/scale/test"])
subprocess.run(
    ["python", "NeuralSound/test.py", "--dataset", "dataset/NeuPAT_new/scale/test"]
)

from glob import glob

sub_dir_list = glob("dataset/NeuPAT_new/scale/test/*")
import os

for sub_dir in sub_dir_list:
    if os.path.isdir(sub_dir):
        subprocess.run(["python", "experiments/neuPAT_material_scale/plot.py", sub_dir])
