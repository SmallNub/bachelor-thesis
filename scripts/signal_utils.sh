#!/bin/bash

log INFO "Loading signal utilities..."

# Global flag to stop waiting only on SIGINT or SIGTERM
__stop_waiting=false
__child_pid=

# Cleanup function to terminate child process
__cleanup_child() {
    if [[ -n "$__child_pid" ]] && kill -0 "$__child_pid" 2>/dev/null; then
        log WARNING "Attempting to gracefully terminate child process $__child_pid"
        kill "$__child_pid"

        # Wait up to 5 seconds for graceful termination
        for i in {1..5}; do
            sleep 1
            if ! kill -0 "$__child_pid" 2>/dev/null; then
                log INFO "Child process $__child_pid terminated gracefully"
                return
            fi
        done

        log ERROR "Child process $__child_pid did not exit, forcing termination"
        kill -9 "$__child_pid"
    else
        log INFO "No active child process to clean up"
    fi
}

# Trap handlers
trap '__stop_waiting=true; log WARNING "SIGINT received"; __on_signal SIGINT; __cleanup_child' INT
trap '__stop_waiting=true; log WARNING "SIGTERM received"; __on_signal SIGTERM; __cleanup_child' TERM
trap 'log WARNING "SIGUSR1 received"; __on_signal SIGUSR1' USR1

# Overridable signal handler
__on_signal() {
    local signal=$1
    log WARNING "Unhandled signal $signal"
}

# Function: wait_with_signals <pid>
# Waits for a background process and stops only on SIGINT or SIGTERM
wait_with_signals() {
    local pid=$1
    local status
    __child_pid=$pid

    log INFO "Waiting for process $pid"

    while true; do
        wait "$pid"
        status=$?

        if [[ $__stop_waiting == true ]]; then
            log WARNING "Interrupted by SIGINT or SIGTERM, not re-waiting"
            break
        fi

        if [[ $status -eq 0 ]]; then
            log INFO "Process $pid finished successfully"
            break
        elif [[ $status -gt 128 ]]; then
            log WARNING "Wait interrupted by a signal, re-waiting..."
            continue
        else
            log ERROR "Process $pid exited with error status $status"
            break
        fi
    done

    return $status
}

log INFO "Signal utilities loaded"
