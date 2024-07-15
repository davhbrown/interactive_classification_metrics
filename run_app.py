import sys
from importlib import resources
from bokeh.command.bootstrap import main


def wrapper():
    # Use resources.path to get a path-like object pointing to serve.py
    with resources.path(
        "interactive_classification_metrics", "serve.py"
    ) as serve_path:
        main(["bokeh", "serve", "--show", str(serve_path)])


if __name__ == "__main__":
    sys.exit(wrapper())
