import json
import os


def set_env():
    try:
        with open("emecom_with_stack_based_agents/env.json", "r") as f:
            env = json.load(f)

        for key, value in env.items():
            os.environ[key] = value
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            pass
        else:
            print(e)
