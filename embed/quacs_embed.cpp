// Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
// SPDX-License-Identifier: Apache-2.0
//
// The C interface to QUACS. This starts Python and passes each call on to a
// `quacs.embed.EmbeddedRuntime`, which does all of the work.
#include "quacs_embed.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

#include <pybind11/embed.h>

namespace py = pybind11;

namespace {
    std::unique_ptr<py::scoped_interpreter> interpreter;
    py::object runtime;
    bool finalized = false;
    std::string last_error;
    std::string report;

    // Runs `action`, turning any exception into an error code and a message. A Python
    // exception arrives here as a C++ exception whose message includes the traceback.
    template<typename Action>
    int guarded(Action action) {
        try {
            action();
            return 0;
        } catch (const std::exception &error) {
            last_error = error.what();
        } catch (...) {
            last_error = "unknown_error";
        }
        return 1;
    }

    py::object &started() {
        if (!runtime) {
            throw std::runtime_error("QUACS is not running. Call quacs_initialize() first.");
        }
        return runtime;
    }

    // Copies `text` into a fixed-length buffer. Text that does not fit is cut short
    // if `trim` is set, and is an error otherwise.
    void copy(const std::string &text, char *buffer, std::size_t length, bool trim = false) {
        if (text.size() >= length && !trim) {
            throw std::length_error("'" + text + "' is longer than " + std::to_string(length - 1) + 
                                    " characters.");
        }
        std::size_t count = std::min(text.size(), length - 1);
        std::memcpy(buffer, text.data(), count);
        buffer[count] = '\0';
    }
}

int quacs_initialize(const char *config_path) {
    return guarded([=] {
        if (interpreter || finalized) {
            throw std::runtime_error("QUACS can only be initialized once in a process.");
        }
        const char *python = std::getenv("QUACS_PYTHON");
        PyConfig config;
        PyConfig_InitPythonConfig(&config);
        config.install_signal_handlers = 0;
        PyStatus status = PyConfig_SetBytesString(&config, &config.executable,
                                                  python ? python : QUACS_PYTHON_EXECUTABLE);
        if (PyStatus_Exception(status)) {
            PyConfig_Clear(&config);
            throw std::runtime_error("The path of the Python executable could not be set.");
        }
        interpreter = std::make_unique<py::scoped_interpreter>(&config);
        runtime = py::module_::import("quacs.embed").attr("EmbeddedRuntime")(config_path);
    });
}

int quacs_map_name(const char *name, const char *host_name) {
    return guarded([=] { started().attr("map_name")(name, host_name); });
}

int quacs_variable_count(int *count) {
    return guarded([=] { *count = static_cast<int>(py::len(started().attr("variables")())); });
}

int quacs_variable_info(int index, quacs_variable *variable) {
    return guarded([=] {
        py::tuple info = started().attr("variable_info")(index);
        copy(info[0].cast<std::string>(), variable->name, QUACS_NAME_LENGTH);
        copy(info[1].cast<std::string>(), variable->units, QUACS_NAME_LENGTH, true);
        copy(info[2].cast<std::string>(), variable->description, QUACS_DESCRIPTION_LENGTH, true);
        variable->shape = info[3].cast<int>();
        variable->dtype = info[4].cast<int>();
        variable->permissions = info[5].cast<int>();
        variable->transported = info[6].cast<bool>() ? 1 : 0;
    });
}

int quacs_set_grid(int ncolumns, int nlevels) {
    return guarded([=] { started().attr("set_grid")(ncolumns, nlevels); });
}

int quacs_bind_variable(const char *name, void *data, int storage, int64_t column_stride,
                        int64_t level_stride) {
    return guarded([=] {
        started().attr("bind_variable")(name, reinterpret_cast<std::uintptr_t>(data), storage,
                                        column_stride, level_stride);
    });
}

int quacs_run_hook(const char *label) {
    return guarded([=] { started().attr("run_hook")(label); });
}

int quacs_check_conformance(int *failures) {
    return guarded([=] {
        py::object result = started().attr("check_conformance")();
        report = py::str(result).cast<std::string>();
        int count = 0;
        for (py::handle check : result.attr("results")) {
            if (!check.attr("passed").cast<bool>()) {
                ++count;
            }
        }
        *failures = count;
    });
}

const char *quacs_report(void) { return report.c_str(); }

const char *quacs_last_error(void) { return last_error.c_str(); }

void quacs_finalize(void) {
    if (!interpreter) {
        return;
    }
    runtime = py::object();
    interpreter.reset();
    finalized = true;
}
