import os
import sys

from launch import run

NAME = "ZeST"
req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "requirements.txt")
print(f"loading {NAME} reqs from {req_file}")
run(f'"{sys.executable}" -m pip install -q -r "{req_file}"',
    f"Checking {NAME} requirements.",
    f"Couldn't install {NAME} requirements.")

print(
    "loading weights from huggingface! if not download weights,please download from https://huggingface.co/h94/IP-Adapter by yourself!!")
run("cd ip_adapter")
run("git clone https://huggingface.co/h94/IP-Adapter")
run("cd ..")

run("cd DPT/weights/")
run("wget https://huggingface.co/lllyasviel/ControlNet/blob/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt")
run("cd ..")
run("cd ..")

