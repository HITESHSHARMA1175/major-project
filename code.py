import sys

# Standard library module 'code' guard to prevent python naming collision
if __name__ == 'code':
    import importlib.util
    import os
    stdlib_code_path = os.path.join(sys.base_prefix, 'Lib', 'code.py')
    if os.path.exists(stdlib_code_path):
        spec = importlib.util.spec_from_file_location("code", stdlib_code_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["code"] = mod
        spec.loader.exec_module(mod)
else:
    import eye_mouse
    if __name__ == '__main__':
        eye_mouse.main()
