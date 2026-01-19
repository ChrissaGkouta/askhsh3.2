import subprocess
#import matplotlib.pyplot as plt
import re
import sys
from collections import defaultdict

# --- ΡΥΘΜΙΣΕΙΣ ---
SIZES = [1024, 2048, 4096]
SPARSITIES = [0.50, 0.95, 0.99]
ITERATIONS = 10
PROCS = [1, 2, 4, 8]
REPEATS = 3  # Πόσες φορές τρέχει για μέσο όρο

def compile_code():
    print("Compiling C code...")
    try:
        subprocess.run(["make", "clean"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["make"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Compilation successful.\n")
    except subprocess.CalledProcessError:
        print("Error: Compilation failed.")
        sys.exit(1)

def parse_output(output_str):
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
            return None # Αν λείπει κάτι, ακύρωση
    return data

def run_experiments():
    # Λεξικό για να μαζέψουμε τα δεδομένα: Key -> List of results
    raw_data = defaultdict(list)
    
    total = len(SIZES) * len(SPARSITIES) * len(PROCS) * REPEATS
    count = 0
    print(f"Starting {total} runs...")

    for n in SIZES:
        for sp in SPARSITIES:
            for p in PROCS:
                for r in range(REPEATS):
                    cmd = ["mpirun", "-np", str(p), "./mpi_spmv", str(n), str(sp), str(ITERATIONS)]
                    try:
                        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        parsed = parse_output(res.stdout)
                        if parsed:
                            # Αποθηκεύουμε τα αποτελέσματα με κλειδί τις παραμέτρους
                            key = (n, sp, p)
                            raw_data[key].append(parsed)
                    except subprocess.CalledProcessError:
                        print(f"\nError running N={n} P={p}")
                    
                    count += 1
                    print(f"\rProgress: {count}/{total}", end="")
    print("\nProcessing data...")
    
    # Υπολογισμός Μέσων Όρων
    averaged_data = []
    for key, runs in raw_data.items():
        n, sp, p = key
        avg_entry = {'N': n, 'Sparsity': sp, 'Procs': p}
        
        # Μέσος όρος για κάθε μετρική
        for metric in ['Time_CSR_Build', 'Time_CSR_Comm', 'Time_CSR_Calc', 'Time_Total_CSR', 'Time_Total_Dense']:
            values = [r[metric] for r in runs]
            avg_entry[metric] = sum(values) / len(values)
        
        averaged_data.append(avg_entry)
        
    return averaged_data

# --- PLOTTING (Pure Matplotlib) ---

def plot_scalability(data):
    # Φιλτράρισμα για Sparsity = 0.95
    target_sp = 0.95
    subset = [d for d in data if d['Sparsity'] == target_sp]
    
    if not subset: return

    plt.figure(figsize=(10, 6))
    
    # Βρες τα μοναδικά N
    unique_ns = sorted(list(set(d['N'] for d in subset)))
    
    for n in unique_ns:
        # Πάρε τα δεδομένα για αυτό το N, ταξινομημένα κατά Procs
        points = sorted([d for d in subset if d['N'] == n], key=lambda x: x['Procs'])
        
        procs = [p['Procs'] for p in points]
        times = [p['Time_CSR_Calc'] for p in points]
        
        # Speedup = T(1) / T(P)
        base_time = times[0] # Ο χρόνος με 1 proc (είναι sorted)
        speedup = [base_time / t for t in times]
        
        plt.plot(procs, speedup, marker='o', label=f'N={n}')

    # Ιδανική γραμμή
    max_p = max(d['Procs'] for d in subset)
    plt.plot([1, max_p], [1, max_p], '--', color='gray', label='Ideal')
    
    plt.title(f'Scalability (Sparsity {target_sp})')
    plt.xlabel('Processes')
    plt.ylabel('Speedup')
    plt.legend()
    plt.grid(True)
    print("Showing Scalability Plot (Close window to continue)...")
    plt.show()

def plot_csr_vs_dense(data):
    # Φιλτράρισμα για Max N και Max Procs
    max_n = max(d['N'] for d in data)
    max_p = max(d['Procs'] for d in data)
    
    subset = sorted([d for d in data if d['N'] == max_n and d['Procs'] == max_p], key=lambda x: x['Sparsity'])
    
    if not subset: return

    sparsities = [d['Sparsity'] for d in subset]
    t_csr = [d['Time_CSR_Calc'] for d in subset]
    t_dense = [d['Time_Total_Dense'] for d in subset]
    
    x = range(len(sparsities))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], t_csr, width, label='CSR')
    plt.bar([i + width/2 for i in x], t_dense, width, label='Dense')
    
    plt.xticks(x, sparsities)
    plt.title(f'CSR vs Dense (N={max_n}, Procs={max_p})')
    plt.xlabel('Sparsity')
    plt.ylabel('Time (s) - Log Scale')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, axis='y')
    print("Showing CSR vs Dense Plot (Close window to continue)...")
    plt.show()

def plot_breakdown(data):
    # Φιλτράρισμα για Max N και Sparsity 0.95
    max_n = max(d['N'] for d in data)
    target_sp = 0.95
    
    subset = sorted([d for d in data if d['N'] == max_n and d['Sparsity'] == target_sp], key=lambda x: x['Procs'])
    
    if not subset: return
    
    procs = [str(d['Procs']) for d in subset] # String για να είναι κατηγορικός άξονας
    t_build = [d['Time_CSR_Build'] for d in subset]
    t_comm = [d['Time_CSR_Comm'] for d in subset]
    t_calc = [d['Time_CSR_Calc'] for d in subset]
    
    plt.figure(figsize=(10, 6))
    
    # Stacked Bars
    p1 = plt.bar(procs, t_build, label='Build')
    p2 = plt.bar(procs, t_comm, bottom=t_build, label='Comm')
    # Το bottom του 3ου είναι το άθροισμα των 2 πρώτων
    bottom_3 = [t_build[i] + t_comm[i] for i in range(len(t_build))]
    p3 = plt.bar(procs, t_calc, bottom=bottom_3, label='Calc')
    
    plt.title(f'Time Breakdown (N={max_n}, Sp={target_sp})')
    plt.xlabel('Processes')
    plt.ylabel('Time (s)')
    plt.legend()
    print("Showing Breakdown Plot...")
    plt.show()

if __name__ == "__main__":
    compile_code()
    results = run_experiments()
    
    if results:
        plot_scalability(results)
        plot_csr_vs_dense(results)
        plot_breakdown(results)
    else:
        print("No results found.")