import importlib

for pkg in ('sentence_transformers', 'sentence_transformers', 'torch'):
    try:
        mod = importlib.import_module(pkg)
        print(pkg, 'OK', getattr(mod, '__version__', ''))
    except Exception as e:
        print(pkg, 'MISSING or error:', type(e).__name__, str(e))
