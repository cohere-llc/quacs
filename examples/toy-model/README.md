# Toy Model Example

A small pure-Python "host model" that runs three toy parameterizations through QUACS.

## Files

- `main.py`: the toy model. It sets up the host adapter and the parameterizations, then
  runs five timesteps.
- `host_adapter.py`: `ToyHostAdapter`, the toy model's host adapter. It builds on
  `quacs.array_host.ArrayHostAdapter`, which provides the operations, so it only has to
  say which variables the toy model has and where they are. The toy model has a grid of
  4 columns by 3 levels (12 grid cells) and four variables of its own: `t_host`,
  `surface_wind_speed`, `cos_solar_zenith_angle`, and `time_step`.
- `parameterizations.py`: three parameterizations that know nothing about the toy model.
  - `HeatingScheme` raises `temperature` in every grid cell, in proportion to the
    sunlight on its column. It records each column's net heating in `column_heating` and
    whether the column is sunlit in `is_daytime`.
  - `OzoneChemistry` calculates the actinic flux at every interface in `actinic_flux`,
    working down each column from the model top, then produces and removes `ozone` in
    every grid cell. It records the flux absorbed in each grid cell in `absorbed_flux`
    and counts its updates to each grid cell in `chemistry_updates`.
  - `SurfaceDeposition` calculates the downward ozone flux at every interface in
    `ozone_flux`: mixing between grid cells, and deposition at the surface using the toy
    model's `surface_wind_speed`. It records the flux at the surface in
    `surface_ozone_flux` and at the model top in `model_top_ozone_flux`.

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
it needs access to. The function receives one accessor per name, in the same order, and
reads and writes each variable with `get()` and `set(value)`. Column functions receive
the column with a specified selected level. The `step_toward_surface` and
`step_toward_model_top` functions can be used to advance one or more steps at a time
vertically. Functions like `at_model_top` or `at_surface` can be used to determine
the current location of the selected level.

Functions passed to `for_each_cell` and `for_each_column` must follow these rules:

- Use only plain numeric code: arithmetic, `math` functions, `if`, and `while`.
- Capture only plain numbers from the enclosing scope, and only numbers that stay the
  same for the whole run. Copy settings such as `self._rate` into a local variable first.
- Do not keep state between calls or depend on the order of calls.

Values returned by `get()` are plain numbers. Changing one does not change the host
variable; use `set(value)` for that.

A `for_each_cell` function can read a variable that has one value per column in every
grid cell of that column, but cannot set it. Use `for_each_column` to set it.

Interfaces can be accessed (read-only) in a `for_each_cell` or `for_each_column`
functions by wrapper the variable with `upper()` or `lower()`. Similarly, adjacent
grid cells can be accessed in `for_each_interface` functions by wrapping the
variable with `above()` or `below()`.

A number given in a `for_each_*` call is passed to the function as it is. So is the
value of a variable that has one value for the whole domain, such as `time_step`. Pass
a number this way if it can change during a run: a captured number is fixed when the
function is compiled.

The toy host can call each function from a Python loop, or compile it with Numba. The
rules above are what make compiling possible.

A variable's values are `float`, `int` or `bool`, as given by the `dtype` it is
registered with. The host chooses the precision it stores `float` and `int` values with.
`set(value)` needs a value of the variable's type; an `int` is also accepted for a
`float` variable.

## Running the toy model

You need [uv](https://docs.astral.sh/uv/) and Python 3.13 or later. uv can install
Python for you.

From the repository root:

```sh
uv run python examples/toy-model/main.py
```
Add `--compiled` to compile the parameterization functions with Numba. The results are
the same, after a pause while the functions are compiled.

The first run creates a virtual environment and installs QUACS and NumPy into it.

The toy model prints out a domain-average time series, final column averages, and
final vertical interface and mid-point averages. You should see:
```sh
QUACS toy model: 4 columns x 3 levels, 5 timesteps

Domain means after each timestep
  step     T (K)        O3  surface O3 flux
  1     290.0222  0.036713         0.000475
  2     290.0444  0.068574         0.000879
  3     290.0665  0.096257         0.001224
  4     290.0886  0.120335         0.001517
  5     290.1107  0.141296         0.001769

Each column after the last timestep
  column  cos(SZA)  daytime  heating (K)  surface O3 flux  model top O3 flux
  1         0.2500     True       0.3750         0.001765           0.000000
  2         0.5000     True       0.7500         0.001768           0.000000
  3         0.7500     True       1.1250         0.001770           0.000000
  4         1.0000     True       1.5000         0.001772           0.000000

Column 1 after the last timestep, from the model top down
                T (K)        O3  absorbed flux  updates  actinic flux   O3 flux
  model top                                                  1.000000  0.000000
  cell 1     289.1760  0.146376       0.065479       10                        
  interface                                                  0.934521  0.000548
  cell 2     289.1760  0.140898       0.059187       10                        
  interface                                                  0.875334  0.000510
  cell 3     289.1760  0.135797       0.053674       10                        
  surface                                                    0.821660  0.001765
```

## Profiling

QUACS includes a simple profiler that parameterization developers can use to compare
the runtime of a parameterization with and without compilation, and to check that its
functions can be compiled by Numba.

To run the profiler for the toy model's ozone parameterization, run:
```sh
uv run python examples/toy-model/profile_ozone.py
```

The report output from the profiler should look something like this:
```sh
Interpreted
Profile of 'ozone_chemistry' over 50 samples on 10000 grid cells. Times are per-grid-cell.
  mean:                   2.995 us
  standard deviation:     27 ns
  fastest sample:         2.871 us
  slowest sample:         3.028 us
  variables, from lowest to highest across the samples:
                       starting value    ending domain mean
    ozone              0.01846 to 0.993  0.01754 to 0.9448
    actinic_flux       0                 0 to 0.7645
    absorbed_flux      0                 0 to 0.05
    temperature        201.5 to 317.7    201.5 to 317.7
    chemistry_updates  0                 1 to 1
    is_daytime         0 to 1            0 to 1
  functions:
    attenuate (for_each_column): compiles with Numba
    react (for_each_cell): compiles with Numba

Compiled
Profile of 'ozone_chemistry' over 50 samples on 10000 grid cells. Times are per-grid-cell.
  mean:                   8 ns
  standard deviation:     1 ns
  fastest sample:         8 ns
  slowest sample:         17 ns
  variables, from lowest to highest across the samples:
                       starting value    ending domain mean
    ozone              0.01846 to 0.993  0.01754 to 0.9448
    actinic_flux       0                 0 to 0.7645
    absorbed_flux      0                 0 to 0.05
    temperature        201.5 to 317.7    201.5 to 317.7
    chemistry_updates  0                 1 to 1
    is_daytime         0 to 1            0 to 1
  functions:
    attenuate (for_each_column): compiles with Numba
    react (for_each_cell): compiles with Numba

The two runs give the same results. Compiled is 356.0 times as fast as interpreted.
```