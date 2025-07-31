import json
import os
import subprocess
import sys

# Get the Poetry environment Python path
venv_python = subprocess.check_output(
    ["poetry", "run", "which", "python"], text=True
).strip()
venv_base = os.path.dirname(os.path.dirname(venv_python))

# Find pybind11 include path
pybind11_include = subprocess.check_output(
    ["poetry", "run", "python", "-c", "import pybind11; print(pybind11.get_include())"],
    text=True,
).strip()

# Get Python include directory (where Python.h lives)
python_include = subprocess.check_output(
    [
        "poetry",
        "run",
        "python",
        "-c",
        "import sysconfig; print(sysconfig.get_path('include'))",
    ],
    text=True,
).strip()

# Alternative Python include path (sometimes needed)
python_platinclude = subprocess.check_output(
    [
        "poetry",
        "run",
        "python",
        "-c",
        "import sysconfig; print(sysconfig.get_path('platinclude'))",
    ],
    text=True,
).strip()

# Generic venv includes
venv_include = os.path.join(venv_base, "include")

# System Python includes (fallback)
python_version = subprocess.check_output(
    [
        "poetry",
        "run",
        "python",
        "-c",
        "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
    ],
    text=True,
).strip()
system_python_include = f"/usr/local/include/python{python_version}"

ccp_path = ".vscode/c_cpp_properties.json"
if os.path.exists(ccp_path):
    with open(ccp_path) as f:
        data = json.load(f)
else:
    data = {
        "configurations": [
            {
                "name": "Linux",
                "includePath": [
                    "${workspaceFolder}/src/engine/_core",
                    "${workspaceFolder}/src/engine/_core/**",
                    "/usr/include",
                    "/usr/local/include",
                ],
                "defines": [],
                "compilerPath": "/usr/bin/g++",
                "cStandard": "c11",
                "cppStandard": "c++20",
                "intelliSenseMode": "linux-gcc-x64",
            }
        ],
        "version": 4,
    }

# Add dynamic includes if not already present
ipath = data["configurations"][0]["includePath"]

# All the paths we want to add
needed_paths = [
    pybind11_include,
    python_include,
    python_platinclude,
    venv_include,
    system_python_include,
]

for needed in needed_paths:
    if needed and needed not in ipath and os.path.exists(needed):
        ipath.append(needed)
        print(f"Added include path: {needed}")

with open(ccp_path, "w") as f:
    json.dump(data, f, indent=2)

print("\nPatched .vscode/c_cpp_properties.json successfully!")
print(f"pybind11: {pybind11_include}")
print(f"Python headers: {python_include}")
print(f"Python platform headers: {python_platinclude}")
