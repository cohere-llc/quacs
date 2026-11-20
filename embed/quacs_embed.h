/* Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
 * SPDX-License-Identifier: Apache-2.0
 *
 * Runs QUACS parameterizations inside a host application.
 *
 * Every function that returns an int returns 0 on success. After a failure,
 * quacs_last_error() describes what went wrong.
 *
 * The functions are called in this order:
 *
 *   quacs_initialize
 *   quacs_map_name          for each host variable that has a QUACS name
 *   quacs_variable_count
 *   quacs_variable_info     for each variable; the host creates those it does not have
 *   quacs_set_grid
 *   quacs_bind_variable     for each variable, and again whenever one moves in memory
 *   quacs_run_hook          as often as needed
 *   quacs_finalize
 *
 * All calls must come from the thread that called quacs_initialize. There is one grid
 * for each process.
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Max sizes for strings coming from quacs, including the terminating null. */
#define QUACS_NAME_LENGTH 128
#define QUACS_DESCRIPTION_LENGTH 512

/* Where a variable has values. Integer values match `_SHAPE_CODES` in `embed.py`*/
enum quacs_shape {
  QUACS_CELL = 1,           /* every grid cell */
  QUACS_INTERFACE = 2,      /* every interface between grid cells, and the two ends */
  QUACS_SURFACE_CELL = 3,   /* every column, for the grid cell nearest the surface */
  QUACS_MODEL_TOP_CELL = 4, /* every column, for the grid cell nearest the model top */
  QUACS_COLUMN = 5,         /* every column, for the column as a whole */
  QUACS_DOMAIN = 6          /* one value for the whole domain */
};

/* Variable data types, as parameterizations see them. Integer values match
 * `_DTYPE_CODES` in `embed.py`. */
enum quacs_dtype { QUACS_FLOAT = 1, QUACS_INT = 2, QUACS_BOOL = 3 };

/* Parameterization permissions for host variables. The host application should create
 * each variable that is owned, and must already have the remaining variables.
 * Integer values match `_PERMISSION_CODES` in `embed.py`. */
enum quacs_permissions { QUACS_READ_ONLY = 1, QUACS_UPDATE = 2, QUACS_OWN = 3 };

/* Host data types. Precision for int and float is determined by the host. A QUACS_BOOL
 * can be stored as QUACS_STORAGE_BOOL, or an integer that is 0 (False) or 1 (True).
 * Integer enum values match `Storage` in `embed.py`. */
enum quacs_storage {
    QUACS_STORAGE_FLOAT32 = 1,
    QUACS_STORAGE_FLOAT64 = 2,
    QUACS_STORAGE_INT32 = 3,
    QUACS_STORAGE_INT64 = 4,
    QUACS_STORAGE_BOOL = 5
};

/* A variable used by QUACS parameterizations */
typedef struct quacs_variable {
    char name[QUACS_NAME_LENGTH]; /* the host's name for the variable */
    char units[QUACS_NAME_LENGTH];
    char description[QUACS_DESCRIPTION_LENGTH]; /* trimmed to fit if needed */
    int shape;       /* a `quacs_shape` */
    int dtype;       /* a `quacs_dtype` */
    int permissions; /* a `quacs_permissions` */
    int transported; /* 1 if the host application should transport the variables
                      * with horizontal air flow*/
} quacs_variable;

/* Starts Python and runs the configuration script, which sets the parameterizations,
 * the hooks and which parameterizations run in them in order, and whether to
 * jit-compile the parameterizations with Numba. This can be called once in the
 * life of a process. */
int quacs_initialize(const char *config_path);

/* Returns the host application's name for a QUACS variable. */
int quacs_map_name(const char *name, const char *host_name);

/* Counts the variables that the parameterizations use. Names can not be mapped after
 * calling this function. */
int quacs_variable_count(int *count);

/* Describes a variable. `index` starts at 0. The host chooses the order of variables. */
int quacs_variable_info(int index, quacs_variable *variable);

/* Sets the number of columns and the number of grid cells in each column. */
int quacs_set_grid(int ncolumns, int nlevels);

/* Binds the memory location and ordering of a host variable to a QUACS variable.
 *
 * QUACS parameterizations operate on host data in-place (no copies). If the host
 * array changes memory location, the QUACS variable must be bound again.
 * 
 * `data` points to the value for the first column, at the surface for a variable
 * with per-grid-cell or per-interface data.
 * 
 * `column_stride` and `level_stride` are the distances in bytes from `data` to
 * the next column and next level moving upward, respectively. Strides can be
 * negative. `level_stride` is ignored for variables without levels, and both are
 * ignored for a `QUACS_DOMAIN` variable. */
int quacs_bind_variable(const char *name, void *data, int storage, int64_t column_stride,
                        int64_t level_stride);

/* Runs the parameterizations configured for a specific hook. */
int quacs_run_hook(const char *label);

/* Compares the variables of any conformance-check parameterizations included in the
 * configuration with the values they are expected to have. Sets `failures` to the
 * number of failed checks. `quacs_report()` can then be called to get the result of
 * every check. */
int quacs_check_conformance(int *failures);

/* Returns the results of a conformance check. `quacs_check_performance()` must be
 * called first. */
const char *quacs_report(void);

/* Returns details about the last QUACS error. */
const char *quacs_last_error(void);

/* Stops Python. Nothing else can be called after calling this. */
void quacs_finalize(void);

#ifdef __cplusplus
}
#endif
