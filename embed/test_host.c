/* Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
 * SPDX-License-Identifier: Apache-2.0
 *
 * A host application written in C that runs the QUACS conformance checks.
 * 
 * This can be used during a new QUACS integration to test the build configuration for QUACS
 * embedding on the target platform. It can also act as a template for creating a real
 * integration.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "quacs_embed.h"

#define NCOLUMNS 5
#define NLEVELS 4
#define NTRACERS 3
#define QUACS_TRACER 1 /* the position of the QUACS tracer among the host's tracers */
#define STEPS 3
#define MAX_VARIABLES 64

typedef struct {
    quacs_variable info;
    void *data;  /* values for each column, and each vertical level from the model top down */
    int levels;  /* the number of values for each column */
    size_t size; /* the number of bytes in each value */
    int storage; /* a quacs_storage type */
} variable;

static variable variables[MAX_VARIABLES];
static int count = 0;
static int failures = 0;

/* The host's tracers: [column][level from the model top][tracer] */
static double tracers[NCOLUMNS][NLEVELS][NTRACERS];

static void report(const char *label, int passed) {
    printf("%s  %s\n", passed ? "PASS" : "FAIL", label);
    if (!passed) {
        ++failures;
    }
}

/* Stops the program if a call that should work does not. */
static void must(int status) {
    if (status != 0) {
        printf("FAIL  %s\n", quacs_last_error());
        exit(1);
    }
}

static variable *find(const char *name) {
    for (int i = 0; i < count; ++i) {
        if (strcmp(variables[i].info.name, name) == 0) {
            return &variables[i];
        }
    }
    printf("FAIL  QUACS did not ask for the variable %s\n", name);
    exit(1);
}

/* Creates a variable, set to zero. A transported variable is one of the host's tracers */
static void create(variable *v) {
    int shape = v->info.shape;
    int is_float = v->info.dtype == QUACS_FLOAT;
    v->levels = shape == QUACS_CELL ? NLEVELS : shape == QUACS_INTERFACE ? NLEVELS + 1 : 1;
    v->storage = is_float ? QUACS_STORAGE_FLOAT64 : QUACS_STORAGE_INT32;
    v->size = is_float ? sizeof(double) : sizeof(int32_t);
    size_t columns = shape == QUACS_DOMAIN ? 1 : NCOLUMNS;
    v->data = v->info.transported ? NULL : calloc(columns * (size_t)v->levels, v->size);
}

/* Binds a host variable to a QUACS variable. For a variable with levels, the value at the
 * surface is the last one of the first column. */
static int bind(const variable *v) {
    if (v->info.transported) {
        return quacs_bind_variable(v->info.name, &tracers[0][NLEVELS - 1][QUACS_TRACER], v->storage,
                                    sizeof(tracers[0]), -(int64_t)sizeof(tracers[0][0]));
    }
    if (v->info.shape == QUACS_CELL || v->info.shape == QUACS_INTERFACE) {
        char *surface = (char *)v->data + (size_t)(v->levels - 1) * v->size;
        return quacs_bind_variable(v->info.name, surface, v->storage,
                                    (int64_t)((size_t)v->levels * v->size), -(int64_t)v->size);
    }
    return quacs_bind_variable(v->info.name, v->data, v->storage, (int64_t)v->size, 0);
}

/* Returns whether every value of an integer variable is `at_top + step * (levels below the top)`. */
static int holds(const variable *v, int at_top, int step) {
  const int32_t *values = v->data;
  size_t columns = v->info.shape == QUACS_DOMAIN ? 1 : NCOLUMNS;
  for (size_t column = 0; column < columns; column++) {
    for (int level = 0; level < v->levels; level++) {
      if (values[column * (size_t)v->levels + (size_t)level] != at_top + step * level) {
        return 0;
      }
    }
  }
  return 1;
}

/* Returns whether the QUACS tracer is `value` everywhere and the host's other tracers are untouched. */
static int tracers_hold(double value) {
    for (int column = 0; column < NCOLUMNS; ++column) {
        for (int level = 0; level < NLEVELS; ++level) {
            for (int tracer = 0; tracer < NTRACERS; ++tracer) {
                if (tracers[column][level][tracer] != (tracer == QUACS_TRACER ? value : -7.0)) {
                    return 0;
                }
            }
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <configuration file>\n", argv[0]);
        return 1;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);

    /* Find out which variables the parameterizations use, and create them. */
    must(quacs_initialize(argv[1]));
    must(quacs_map_name("height", "z"));
    must(quacs_variable_count(&count));
    if (count > MAX_VARIABLES) {
        printf("FAIL  QUACS asked for %d variables\n", count);
        return 1;
    }
    for (int i = 0; i < count; ++i) {
        must(quacs_variable_info(i, &variables[i].info));
        create(&variables[i]);
    }
    variable *z = find("z");
    variable *tracer = find("quacs_check_tracer");
    report("the host's owned variable is only read", z->info.permissions == QUACS_READ_ONLY);
    report("the tracer is owned and transported",
            tracer->info.permissions == QUACS_OWN && tracer->info.transported == 1);

    for (int column = 0; column < NCOLUMNS; ++column) {
        for (int level = 0; level < NLEVELS; ++level) {
            ((double*)z->data)[column * NLEVELS + level] = 100.0 * (NLEVELS - level);
            for (int other = 0; other < NTRACERS; other++) {
                tracers[column][level][other] = other == QUACS_TRACER ? 0.0 : -7.0;
            }
        }
    }

    /* Bind the variables */
    report("a hook cannot run before the grid is set", quacs_run_hook("step") != 0);
    must(quacs_set_grid(NCOLUMNS, NLEVELS));
    for (int i = 1; i < count; ++i) {
        must(bind(&variables[i]));
    }
    report("a hook cannot be run before every variable is bound",
            quacs_run_hook("step") != 0 &&
                strstr(quacs_last_error(), variables[0].info.name) != NULL);
    must(bind(&variables[0]));
    report("a float variable cannot be bound as integers",
            quacs_bind_variable("z", z->data, QUACS_STORAGE_INT32, 0, 0) != 0);
    must(bind(z));

    int failed = -1;
    must(quacs_check_conformance(&failed));
    report("every variable starts at zero", failed == 0);

    /* Run the parameterizations and checks. The memory moves after the first step
     * and is rebound. */
    variable *visits = find("quacs_check_cell_visits");
    for (int step = 0; step < STEPS; ++step) {
        must(quacs_run_hook("step"));
        if (step == 0) {
            size_t bytes = NCOLUMNS * (size_t)visits->levels * visits->size;
            void *moved = malloc(bytes);
            memcpy(moved, visits->data, bytes);
            free(visits->data);
            visits->data = moved;
            must(bind(visits));
        }
    }

    must(quacs_check_conformance(&failed));
    printf("%s\n", quacs_report());
    report("the conformance checks pass", failed == 0);

    /* Display the results in the hosts native layout. */
    report("levels are numbered from the surface, which the host stores last",
           holds(find("quacs_check_level"), NLEVELS, -1));
    report("the variable that moved was updated in its new place", holds(visits, STEPS, 0));
    report("a bool stored as integers is exactly 1 where it is true",
           holds(find("quacs_check_column_swept"), 1, 0));
    report("only the QUACS tracer was set among the host's tracers", tracers_hold(1.0));

    report("an unknown hook is reported", quacs_run_hook("missing") != 0);

    quacs_finalize();
    report("QUACS cannot be started a second time", quacs_initialize(argv[1]) != 0);

    for (int i = 0; i < count; i++) {
      free(variables[i].data);
    }
    printf("%s\n", failures == 0 ? "The embedded host passed." : "The embedded host FAILED.");
    return failures == 0 ? 0 : 1;   
}
