# Embedding QUACS in a host application

`quacs_embed` is a small library that lets you run QUACS parameterizations in a host application
write in C or C++, or any langauge that can communicate with C.

## Building

Requirements:
- [`uv`](https://docs.astral.sh/uv/)
- CMake 3.24 or later
- a C and C++17 compiler

Prior to building the test host, from the root QUACS directory, start up a Python virutal
environment and install QUACS:

```sh
uv sync
source .venv/bin/activate
uv run pip install -e .
```

The library can then be built, and the tests run with:

```sh
cmake -S embed -B embed/build -DPython_EXECUTABLE=$PWD/.venv/bin/python
cmake --build embed/build
cmake --test-dir embed/build --output-on-failure
```

You can build and install the library along with its header and a `pkg-config` file
with:

```sh
cmake -S embed -B embed/build -DPython_EXECUTABLE=$PWD/.venv/bin/python \
      -DCMAKE_INSTALL_PREFIX=/path/to/install
cmake --build embed/build
cmake --install embed/build
```

A host application can then be compiled and linked with:

```sh
pkg-config --cflags --libs quacs_embed
```


## Using the library

`quacs_embed.h` includes descriptions for each function in the QUACS embed API. A host
application calls them in this order:

| Step | Function | Purpose |
|---|---|---|
| 1 | `quacs_initialize` | Start the Python interpreter and run the QUACS configuration script. |
| 2 | `quacs_map_name` | Provided a mapping for host variable names to common QUACS variable names. |
| 3 | `quacs_variable_count`, `quacs_variable_info` | Get information for each variable used by the parmaeterizations. |
| 4 | `quacs_set_grid` | Pass the dimensions of the grid to QUACS. |
| 5 | `quacs_bind_variable` | Associate each QUACS variable with the location in memory of the host's data. |
| 6 | `quacs_run_hook` | Run the parameterizations configured for a given hook label. |
| 7 | `quacs_finalize` | Shut down the Python interpreter. |

The host application creates each variable with `QUACS_OWN` permissions, and must already have
the remaining variables.

### The configuration script

The configuration script is a Python file that names the parameterizations for a simulation and
the hooks the run in:

```python
from my_parameterizations import OzoneChemistry, SurfaceDeposition

parameterizations = [OzoneChemsitry, SurfaceDeposition]
hooks = {
    "before_physics": ["ozone_chemistry"],
    "after_physics": ["ozone_chemistry", "surface_deposition"],
}
```

The folder that holds the configuration file is added to Python's module search path,
so parameterizations can be kept alongside it. Setting `compiled = False` in the file
will disable the Numba compilation of the parameterizations.

### Bindin variables

`quacs_bind_variable` takes a pointer to the value for the first column, at the surface
for per-grid-cell and per-interface variables, along with column and level strides in
bytes. The column stride is the distance to the next column in memory, and the level
stride is the distance to the grid cell above the lowest grid cell in memory.

The following table describes the expected layout and conversions from several
ordering patterns.

| The host stores | `data` points to | `column_stride` | `level_stride` |
|---|---|---|---|
| `double a[ncolumns][nlevels]`, surface first | `&a[0][0]` | `nlevels * 8` | `8` |
| `double a[ncolumns][nlevels]`, model top first | `&a[0][nlevels - 1]` | `nlevels * 8` | `-8` |
| Fortran `real(8) :: a(nlevels, ncolumns)`, surface first | `a(1, 1)` | `nlevels * 8` | `8` |
| Fortran `real(8) :: a(ntracers, nlevels, ncolumns)`, tracer `i` | `a(i, 1, 1)` | `ntracers * nlevels * 8` | `ntracers * 8` |

A variable can be bound at any time, and must be bound whenever the variable
data moves to a different memory location.

A `QUACS_BOOL` can be stored as one byte for each value, or as integers. When stored
as integers, the values must be 0 (false) or 1 (true).

### Checking a host appliction integration

QUACS provides a set of parameterizations that can be used to validate a new host
application integration. They include checks to ensure data is operated on
correctly (see `quacs.conformance`). To use them, list them in a configuration file,
as `test_config.py` does, run their hooks, and call `quacs_check_conformance`.
`quacs_report` then can be used to return a full report from the last check.
