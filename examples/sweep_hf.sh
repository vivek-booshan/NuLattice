#!/bin/bash
#SBATCH -A gen006
#SBATCH -J hf_sweep
#SBATCH -o slurm/%x_%j.out
#SBATCH -e slurm/%x_%j.err
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56          # Adjust based on your CPU parallelization needs
##SBATCH --mem=64G                   # Memory baseline to support larger L sizes (e.g., L=20)
#SBATCH -t 02:00:00                 # Time window for running iterations sequentially

OUTPUT_CSV="data/hf_sweep_results.csv"
TEMP_LOG="temp_hf_output.log"

# ONLY write the header if the CSV file does not already exist
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "Element,Lattice,Convergence,Energy_MeV,Time_Sec,Peak_Memory_Mb,Iterations" > "$OUTPUT_CSV"
    echo "Created new output file: $OUTPUT_CSV"
else
    echo "Found existing output file: $OUTPUT_CSV. Appending new data..."
fi

# Define the grids to sweep across
ELEMENTS=(he4)
LATTICES=(16 18 20)

echo "Starting sequential Hartree-Fock loop sweep..."

# Nested loop through parameters
start=$SECONDS
for ELEMENT in "${ELEMENTS[@]}"; do
    for L in "${LATTICES[@]}"; do
        echo "Processing: Element=$ELEMENT, Lattice L=$L"

        # Execute the command directly and dump stdout/stderr to a temporary log file
        uv run python Example_HF.py --backend cpu --L "$L" --element "$ELEMENT" > "$TEMP_LOG" 2>&1
        
        # Capture the exit status code of the execution
        STATUS=$?

        # Handle hard script crashes or out-of-memory killing events
        if [ $STATUS -ne 0 ]; then
            echo "⚠️ Run crashed or timed out for Element=$ELEMENT, L=$L"
            echo "$ELEMENT,$L,CRASHED,,,,0" >> "$OUTPUT_CSV"
            continue
        fi

        # Extract target metrics via grep and pull specific whitespace-separated columns via awk
        CONV=$(grep "HF Convergence:" "$TEMP_LOG" | awk '{print $3}')
        ENERGY=$(grep "Final HF Energy:" "$TEMP_LOG" | awk '{print $4}')
        
        TIME=$(grep "Time:" "$TEMP_LOG" | awk '{print $2}')
        MEM=$(grep "Peak Memory," "$TEMP_LOG" | awk '{print $3}')

        # Count lines starting with an index number followed by 'E=' (e.g., "0 E=", "31 E=")
        ITERATIONS=$(grep -c -E '^[0-9]+ E=' "$TEMP_LOG")

        # Safely handle empty extractions if the log files parsed weirdly
        CONV=${CONV:-"FAILED_TO_PARSE"}
        ITERATIONS=${ITERATIONS:-"0"}

        # Write data row straight to the master CSV file
        echo "$ELEMENT,$L,$CONV,$ENERGY,$TIME,$MEM,$ITERATIONS" >> "$OUTPUT_CSV"
    done
done
elapsed=$(( SECONDS - start ))
printf "took %02d:%02d:%02d\n" "$((elapsed/3600))" "$((elapsed%3600/60))" "$((elapsed%60))"

# Clean up the temporary tracking log
if [ -f "$TEMP_LOG" ]; then
    rm "$TEMP_LOG"
fi

echo "✅ Sweep completely finished! Output captured inside: $OUTPUT_CSV"
