# Toy Model Example

A small pure-Python "host model" that runs three toy parameterizations through QUACS.

## Files

- `main.py`: the toy model. It sets up the host adapter and the parameterizations, then
  runs five timesteps.
- `host_adapter.py`: `ToyHostAdapter`, the toy model's implementation of the QUACS
  `HostAdapter` protocol. It owns a grid of 4 columns by 3 levels (12 grid cells) and
  two variables of its own, `t_host`, `surface_wind_speed`, and `cos_solar_zenith_angle`.
- `parameterizations.py`: three parameterizations that know nothing about the toy model.
  - `HeatingScheme` raises `temperature` in every grid cell, in proportion to the
    sunlight on its column, and records each column's net heating in `column_heating`.
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

A `for_each_cell` function can read a variable that has one value per column in every
grid cell of that column, but cannot set it. Use `for_each_column` to set it.

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
domain means of temperature, ozone, and surface ozone flux, followed by each column's
net heating from the last timestep. You should see:
```sh
step    mean T (K)       mean O3    mean O3 flux
   1      290.0222      0.036713        0.000475
column heating (K): [0.375 0.75  1.125 1.5  ]
   2      290.0444      0.068574        0.000879
column heating (K): [0.375 0.75  1.125 1.5  ]
   3      290.0665      0.096257        0.001224
column heating (K): [0.375 0.75  1.125 1.5  ]
   4      290.0886      0.120335        0.001517
column heating (K): [0.375 0.75  1.125 1.5  ]
   5      290.1107      0.141296        0.001769
column heating (K): [0.375 0.75  1.125 1.5  ]
```

## Limitations

- `GridShape.INTERFACE` variables are not supported.
- The variables in one `for_each_cell` call must be either all per grid cell or all per
  column.
- Only `float` variables are supported.