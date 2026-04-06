#!/bin/bash -l

# -- Virtual environment (change as needed) --
VENV="/Users/rstiskalek/Projects/CANDEL/venv_candel"
source "${VENV}/bin/activate"

# -- Number of parallel jobs (set to 1 for serial) --
NJOBS=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/output"

mkdir -p "${OUTPUT_DIR}"

INPUT="$(realpath "${1:?Usage: $0 /path/to/chains_dir_or_single_chain}")"

# If it contains a log.param, it's a single chain directory
if [ -f "${INPUT}/log.param" ]; then
    chain_dirs=("${INPUT}")
else
    chain_dirs=("${INPUT}"/desi_*/)
fi

if [ "${#chain_dirs[@]}" -eq 1 ]; then
    # Single chain: run in foreground with live output
    name=$(basename "${chain_dirs[0]}")
    chain_output="${OUTPUT_DIR}/${name}"
    mkdir -p "${chain_output}"
    echo "Processing: ${name}"
    python "${SCRIPT_DIR}/compute_evidence.py" "${chain_dirs[0]}" \
        --output-dir "${chain_output}" \
        2>&1 | tee "${chain_output}/log.txt"
else
    # Multiple chains: run in parallel batches
    pids=()
    for chain_dir in "${chain_dirs[@]}"; do
        name=$(basename "${chain_dir}")
        chain_output="${OUTPUT_DIR}/${name}"
        mkdir -p "${chain_output}"
        echo "Launching: ${name}"

        python "${SCRIPT_DIR}/compute_evidence.py" "${chain_dir}" \
            --output-dir "${chain_output}" \
            > "${chain_output}/log.txt" 2>&1 &

        pids+=($!)

        # Throttle: wait for all if we hit NJOBS
        if [ "${#pids[@]}" -ge "${NJOBS}" ]; then
            wait "${pids[@]}"
            pids=()
        fi
    done

    wait
fi
echo "All jobs finished."

# Generate report only for multiple chains
if [ "${#chain_dirs[@]}" -gt 1 ]; then
    python "${SCRIPT_DIR}/make_report.py" "${OUTPUT_DIR}"
fi

echo "All done. Results in ${OUTPUT_DIR}/"
