
try:
    print("Attempting to import ovito...")
    import ovito
    print(f"OVITO imported successfully. Version: {ovito.version}")
    from ovito.io import import_file
    print("ovito.io imported.")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Exception: {e}")

try:
    print("\nCheck PATH environment variable:")
    import os
    print(os.environ.get('PATH'))
except:
    pass
