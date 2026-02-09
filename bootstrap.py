import os
import subprocess
import sys


def setup(repo_name="BE_data_analysis"):
    if not os.path.exists(repo_name):
        subprocess.run(
            ["git", "clone", f"https://github.com/PetitMalo/{repo_name}.git"], check=True
        )

    os.chdir(repo_name)

    # Installation UV + dépendances
    subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True, check=True)
    os.environ["PATH"] += ":" + os.path.expanduser("~/.cargo/bin")
    subprocess.run(["uv", "pip", "install", "--system", "-r", "pyproject.toml"], check=True)

    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())

    print(f"Project {repo_name} is ready.")


if __name__ == "__main__":
    setup()
