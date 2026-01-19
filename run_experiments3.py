import subprocess
import re
import sys
import time

# --- ΡΥΘΜΙΣΕΙΣ ---
# Μπορείς να αλλάξεις τις τιμές εδώ ανάλογα με τι θέλεις να τρέξεις
SIZES = [1024, 2048, 4096]          # Μέγεθος Πίνακα (N)
SPARSITIES = [0.50, 0.80, 0.95]     # Ποσοστό Μηδενικών
ITERATIONS = 10                     # Επαναλήψεις στον C κώδικα
PROCS = [1, 2, 4, 8]                # Αριθμός Διεργασιών MPI
REPEATS = 4                         # Πόσες φορές θα τρέξει για να βγει ο μέσος όρος

def compile_code():
    print("[-] Compiling C code (make clean && make)...")
    try:
        subprocess.run(["make", "clean"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["make"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[+] Compilation successful.\n")
    except subprocess.CalledProcessError:
        print("[!] Error: Compilation failed. Check your Makefile.")
        sys.exit(1)

def parse_output(output_str):
    """Εξάγει τους χρόνους από το output του C προγράμματος."""
    data = {}
    # Regex για να πιάσουμε τα νούμερα μετά το :
    patterns = {
        'build': r"Time_CSR_Build:\s+([0-9\.]+)",
        'comm':  r"Time_CSR_Comm:\s+([0-9\.]+)",
        'calc':  r"Time_CSR_Calc:\s+([0-9\.]+)",
        't_csr': r"Time_Total_CSR:\s+([0-9\.]+)",
        't_dns': r"Time_Total_Dense:\s+([0-9\.]+)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output_str)
        if match:
            data[key] = float(match.group(1))
        else:
            return None # Κάτι πήγε στραβά
    return data

def run_all():
    # Επικεφαλίδες Πίνακα
    header = f"{'N':<6} | {'Sp%':<5} | {'P':<3} | {'Build(s)':<9} | {'Comm(s)':<9} | {'Calc(s)':<9} | {'TotCSR(s)':<10} | {'TotDense(s)':<11} | {'Speedup':<7}"
    separator = "-" * len(header)
    
    print(separator)
    print(header)
    print(separator)

    # Αποθήκευση του χρόνου Calc για P=1 ώστε να υπολογίζουμε το Speedup
    baseline_calc = {} 

    for n in SIZES:
        for sp in SPARSITIES:
            for p in PROCS:
                
                # Αθροιστές για τον υπολογισμό μέσου όρου
                sums = {'build': 0.0, 'comm': 0.0, 'calc': 0.0, 't_csr': 0.0, 't_dns': 0.0}
                successful_runs = 0
                
                # Τρέξιμο REPEATS φορές
                for r in range(REPEATS):
                    cmd = ["mpirun", "--oversubscribe", "-np", str(p), "./mpi_spmv", str(n), str(sp), str(ITERATIONS)]
                    try:
                        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        data = parse_output(res.stdout)
                        
                        if data:
                            for k in sums:
                                sums[k] += data[k]
                            successful_runs += 1
                    except subprocess.CalledProcessError:
                        pass # Αν σκάσει μια φορά, συνεχίζουμε
                
                # Αν δεν πέτυχε καμία εκτέλεση, προσπερνάμε
                if successful_runs == 0:
                    print(f"{n:<6} | {sp:<5} | {p:<3} | {'FAILED':^50}")
                    continue

                # Υπολογισμός Μέσων Όρων
                avgs = {k: v / successful_runs for k, v in sums.items()}
                
                # Υπολογισμός Speedup (T_serial / T_parallel) για το Calculation
                speedup_str = "1.00x"
                if p == 1:
                    baseline_calc[(n, sp)] = avgs['calc']
                else:
                    base = baseline_calc.get((n, sp))
                    if base and avgs['calc'] > 0:
                        s = base / avgs['calc']
                        speedup_str = f"{s:.2f}x"
                    else:
                        speedup_str = "-"

                # Εκτύπωση γραμμής αποτελεσμάτων
                print(f"{n:<6} | {sp:<5} | {p:<3} | "
                      f"{avgs['build']:.6f}  | {avgs['comm']:.6f}  | {avgs['calc']:.6f}  | "
                      f"{avgs['t_csr']:.6f}   | {avgs['t_dns']:.6f}    | {speedup_str:<7}")
                
                # Κάνουμε flush για να εμφανίζονται αμέσως
                sys.stdout.flush()

    print(separator)
    print("\n[+] Experiments completed.")

if __name__ == "__main__":
    compile_code()
    run_all()