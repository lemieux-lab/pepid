if __package__ is None or __package__ == '':
    from main import run
else:
    from .main import run
