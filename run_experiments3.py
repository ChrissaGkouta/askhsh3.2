import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sys

# --- ΡΥΘΜΙΣΕΙΣ ΠΕΙΡΑΜΑΤΩΝ ---
# Μείωσα λίγο τα runs για να τελειώνει πιο γρήγορα το demo
SIZES = [1024, 2048, 4096]
SPARSITIES = [0.50, 0.95, 0.99] 
ITERATIONS = 10
PROCS = [1, 2, 4, 8]
REPEATS = 3

# Ρυθμίσεις εμφάνισης γραφημάτων
sns.set_theme(style="whitegrid")
# Χρησιμοποιούμε 'Qt5Agg' ή 'TkAgg' για interactive backend αν χρειαστεί, 
# συνήθως το matplotlib το βρίσκει αυτόματα.

def compile_code():
    """Κάνει compile τον C κώδικα χρησιμοποιώντας το make."""
    print("Compiling C code...")
    try:
        # Καθαρισμός παλιών binaries και compile
        subprocess.run(["make", "clean"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["make"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Compilation successful.\n")
    except subprocess.CalledProcessError:
        print("Error: Compilation failed. Check your Makefile and C code.")
        sys.exit(1)

def parse_output(output_str):
    """Regex parsing της εξόδου του C προγράμματος."""
    data = {}
    patterns = {
        'Time_CSR_Build': r"Time_CSR_Build:\s+([0-9\.]+)",
        'Time_CSR_Comm': r"Time_CSR_Comm:\s+([0-9\.]+)",
        'Time_CSR_Calc': r"Time_CSR_Calc:\s+([0-9\.]+)",
        'Time_Total_CSR': r"Time_Total_CSR:\s+([0-9\.]+)",
        'Time_Total_Dense': r"Time_Total_Dense:\s+([0-9\.]+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output_str)
        if match:
            data[key] = float(match.group(1))
        else:
            data[key] = None
    return data

def run_experiments():
    """Τρέχει τα πειράματα και επιστρέφει DataFrame."""
    results_list = []
    total_experiments = len(SIZES) * len(SPARSITIES) * len(PROCS) * REPEATS
    counter = 0

    print(f"Starting experiments (Total runs: {total_experiments})...")

    for n in SIZES:
        for sp in SPARSITIES:
            for p in PROCS:
                for r in range(REPEATS):
                    cmd = ["mpirun", "-np", str(p), "./mpi_spmv", str(n), str(sp), str(ITERATIONS)]
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        times = parse_output(result.stdout)
                        if None in times.values():
                            continue
                        entry = {'N': n, 'Sparsity': sp, 'Iterations': ITERATIONS, 'Procs': p, 'Run': r + 1, **times}
                        results_list.append(entry)
                    except subprocess.CalledProcessError as e:
                        print(f"Error running MPI: {e}")
                    
                    counter += 1
                    print(f"\rProgress: {counter}/{total_experiments}", end="")
    print("\nDone collecting data.")
    
    df = pd.DataFrame(results_list)
    if df.empty: return df
    # Μέσος όρος των repeats
    df_avg = df.groupby(['N', 'Sparsity', 'Iterations', 'Procs']).mean().reset_index()
    return df_avg

# --- ΣΥΝΑΡΤΗΣΕΙΣ ΓΡΑΦΗΜΑΤΩΝ (ΜΕ plt.show() ΑΝΤΙ ΓΙΑ savefig) ---

def plot_scalability(df):
    plt.figure(figsize=(10, 6))
    target_sparsity = 0.95
    subset = df[df['Sparsity'] == target_sparsity].copy()
    if subset.empty: return

    baseline = subset[subset['Procs'] == 1][['N', 'Time_CSR_Calc']].rename(columns={'Time_CSR_Calc': 'Base_Time'})
    subset = pd.merge(subset, baseline, on='N')
    subset['Speedup'] = subset['Base_Time'] / subset['Time_CSR_Calc']

    sns.lineplot(data=subset, x='Procs', y='Speedup', hue='N', style='N', markers=True, dashes=False, palette="viridis")
    plt.plot([1, subset['Procs'].max()], [1, subset['Procs'].max()], '--', color='gray', label='Ideal')
    
    plt.title(f'CSR Scalability (Sparsity {target_sparsity})')
    plt.ylabel('Speedup')
    plt.tight_layout()
    print("Displaying Scalability plot... (Close window to continue)")
    plt.show() # <--- Εμφάνιση στην οθόνη

def plot_csr_vs_dense(df):
    plt.figure(figsize=(10, 6))
    max_n = df['N'].max()
    max_procs = df['Procs'].max()
    subset = df[(df['N'] == max_n) & (df['Procs'] == max_procs)].copy()
    if subset.empty: return
    
    melted = subset.melt(id_vars=['Sparsity'], value_vars=['Time_CSR_Calc', 'Time_Total_Dense'], var_name='Method', value_name='Time')
    melted['Method'] = melted['Method'].replace({'Time_CSR_Calc': 'CSR', 'Time_Total_Dense': 'Dense'})

    sns.barplot(data=melted, x='Sparsity', y='Time', hue='Method', palette="muted")
    plt.title(f'CSR vs Dense Performance (N={max_n}, Procs={max_procs})')
    plt.yscale('log')
    plt.ylabel('Time (log scale)')
    plt.tight_layout()
    print("Displaying CSR vs Dense plot... (Close window to continue)")
    plt.show() # <--- Εμφάνιση στην οθόνη

def plot_time_breakdown(df):
    plt.figure(figsize=(10, 6))
    target_n = df['N'].max()
    subset = df[(df['N'] == target_n) & (df['Sparsity'] == 0.95)].copy()
    if subset.empty: return
    
    plt.bar(subset['Procs'], subset['Time_CSR_Build'], label='Build (Serial)', color='#e74c3c')
    plt.bar(subset['Procs'], subset['Time_CSR_Comm'], bottom=subset['Time_CSR_Build'], label='Communication', color='#3498db')
    plt.bar(subset['Procs'], subset['Time_CSR_Calc'], bottom=subset['Time_CSR_Build']+subset['Time_CSR_Comm'], label='Calculation', color='#2ecc71')
    
    plt.title(f'CSR Time Breakdown (N={target_n}, Sp=0.95)')
    plt.xlabel('Processes')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.tight_layout()
    print("Displaying Time Breakdown plot... (Close window to finish)")
    plt.show() # <--- Εμφάνιση στην οθόνη

# --- MAIN ---

if __name__ == "__main__":
    compile_code()
    
    df_results = run_experiments()
    
    if not df_results.empty:
        print("\nExperiments finished. Preparing plots...")
        # Το script θα σταματάει σε κάθε plot μέχρι να κλείσεις το παράθυρο
        plot_scalability(df_results)
        plot_csr_vs_dense(df_results)
        plot_time_breakdown(df_results)
        print("\nAll plots displayed.")
    else:
        print("No results collected. Check compilation or MPI execution.")