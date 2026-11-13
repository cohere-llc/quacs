# Toy Model Example

A small pure-Python "host model" that runs three toy parameterizations through QUACS.

## Files

- `main.py`: the toy model. It sets up the host adapter and the parameterizations, then
  runs five timesteps.
- `host_adapter.py`: `ToyHostAdapter`, the toy model's implementation of the QUACS
  `HostAdapter` protocol. It owns a grid of 4 columns by 3 levels (12 grid cells) and
  two variables of its own, `t_host` and `surface_wind_speed`.
- `parameterizations.py`: three parameterizations that know nothing about the toy model.
  - `HeatingScheme` raises `temperature` in every grid cell.
  - `OzoneChemistry` creates `ozone` and works down each column from the model top.
  - `SurfaceDeposition` creates `surface_ozone_flux` from the ozone nearest the surface
    and the toy model's `surface_wind_speed`.

## Hooks

The toy model hands control to QUACS at two points in each timestep:

| Hook label        | Parameterizations, in order             |
|-------------------|-----------------------------------------|
| `before_dynamics` | `heating`, `ozone_chemistry`            |
| `after_dynamics`  | `ozone_chemistry`, `surface_deposition` |

Between the two hooks the toy model does its own work, a small cooling of `t_host`.

## How parameterizations see the grid

When parameterizations need to access host state data, they pass a function to
`state.for_each_cell` or `state.for_each_column`, along with the names of the variables
needs access to. The function receives one accessor per name, in the same order, and
reads and writes each variable with `get()` and `set(value)`. Column functions receive
the column with a specified selected level. The `step_toward_surface` and
`step_toward_model_top` functions can be used to advance one or steps at a time
vertically. Functions like `at_model_top` or `at_surface` can be used to determine
the current location of the selected level.

Functions passed to `for_each_cell` and `for_each_column` must follow these rules:

- Use only plain numeric code: arithmetic, `math` functions, `if`, and `while`.
- Capture only plain numbers from the enclosing scope. Copy settings such as
  `self._rate` into a local variable first.
- Do not keep state between calls or depend on the order of calls.

Values returned by `get()` are plain numbers. Changing one does not change the host
variable; use `set(value)` for that.

The toy host calls each function once per grid cell or once per column in a Python
loop. The rules above are what would let a host compile the functions instead.

## Running the toy model

You need [uv](https://docs.astral.sh/uv/) and Python 3.13 or later. uv can install
Python for you.

From the repository root:

```sh
uv run python examples/toy-model/main.py
```

The first run creates a virtual environment and installs QUACS and NumPy into it.

The model prints a header and one row per timestep, with the step number and the
domain means of temperature, ozone, and surface ozone flux. You should see:
```sh
step    mean T (K)       mean O3    mean O3 flux
   1      290.2095      0.036737        0.000475
   2      290.4188      0.068639        0.000880
   3      290.6279      0.096380        0.001225
   4      290.8367      0.120528        0.001520
   5      291.0454      0.141568        0.001772
```

## Limitations

- `GridShape.INTERFACE` variables are not supported.
- The variables in one `for_each_cell` call must be either all per grid cell or all per
  column.
- Only `float` variables are supported.