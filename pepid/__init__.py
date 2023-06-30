if __package__ is None or __package__ == '':
    import pepid
    from pepid.main import run
else:
    from .main import run
