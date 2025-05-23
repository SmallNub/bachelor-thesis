#!/bin/bash

log INFO "Loading signal utilities..."

# Global flag to check if signal was received
__interrupted=false

# Trap handlers
trap '__interrupted=true; log WARNING "SIGUSR1 received"; __on_signal SIGUSR1' USR1
trap '__interrupted=true; log WARNING "SIGINT received"; __on_signal SIGINT' INT
trap '__interrupted=true; log WARNING "SIGTERM received"; __on_signal SIGTERM' TERM

# Overridable signal handler
__on_signal() {
    local signal=$1
    log WARNING "Unhandled signal $signal"
}

# Function: wait_with_signals <pid>
# Waits for a background process and continues handling signals
wait_with_signals() {
    local pid=$1
    local status

    log INFO "Waiting "

    while true; do
        wait "$pid"
        status=$?
        if [[ $status -eq 0 ]]; then
            log INFO "Process $pid finished successfully"
            break
        elif [[ $status -gt 128 ]]; then
            log WARNING "Wait was interrupted by signal, re-waiting..."
            continue
        else
            log ERROR "Process $pid exited with error status $status"
            break
        fi
    done

    return $status
}

log INFO "Signal utilities loaded"
