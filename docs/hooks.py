import subprocess


def on_pre_build(config):
    subprocess.run(["doxygen", "Doxyfile"], check=True)
