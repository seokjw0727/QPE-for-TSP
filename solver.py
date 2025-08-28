import time, os, psutil, warnings, gc, threading, argparse, itertools
from math import ceil, log2
from typing import Dict
from collections import defaultdict
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFTGate, DiagonalGate

warnings.filterwarnings("ignore", category=DeprecationWarning)

def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Benchmark a TSP solver for N runs with random problems.")
    parser.add_argument("-n", "--runs", type=int, default=1, help="Number of simulation runs to perform. (Default: 1)")
    parser.add_argument("-c", "--cities", type=int, default=4, help="Number of cities for the random TSP problems. (Default: 4)")
    return parser.parse_args()

# ===============================================
# 2. 공통 유틸리티 함수
# ===============================================
def generate_random_tsp(num_cities: int, min_dist: int = 10, max_dist: int = 100) -> np.ndarray:
    dist_matrix = np.zeros((num_cities, num_cities), dtype=int)
    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            random_dist = np.random.randint(min_dist, max_dist)
            dist_matrix[i, j] = random_dist
            dist_matrix[j, i] = random_dist
    return dist_matrix

class PeakMemSampler:
    def __init__(self, interval: float = 0.005):
        self.interval = interval; self._stop = False; self._t = None; self.peak_rss_mb = 0.0
    def _run(self):
        proc = psutil.Process(os.getpid())
        while not self._stop:
            rss_mb = proc.memory_info().rss / (1024*1024)
            if rss_mb > self.peak_rss_mb: self.peak_rss_mb = rss_mb
            time.sleep(self.interval)
    def __enter__(self):
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start()
        return self
    def __exit__(self, exc_type, exc, tb):
        self._stop = True
        if self._t: self._t.join()

# ===============================================
# 3. 알고리즘별 핵심 로직 (QPE)
# ===============================================
def get_tours_and_costs(dist_matrix):
    num_cities = len(dist_matrix)
    other_cities = list(range(1, num_cities))
    base_permutations = list(itertools.permutations(other_cities))
    permutations = [(0,) + p for p in base_permutations]
    tour_map = {i: tour for i, tour in enumerate(permutations)}
    costs = [sum(dist_matrix[tour[i], tour[i+1]] for i in range(num_cities - 1)) + dist_matrix[tour[-1], tour[0]] for tour in permutations]
    return costs, tour_map

def create_qpe_search_circuit(n_counting, n_system, phases):
    counting_qr = QuantumRegister(n_counting, name='q_counting')
    system_qr = QuantumRegister(n_system, name='q_system')
    counting_cr = ClassicalRegister(n_counting, name='c_cost')
    system_cr = ClassicalRegister(n_system, name='c_path')
    
    qc = QuantumCircuit(counting_qr, system_qr, counting_cr, system_cr)
    qc.h(system_qr); qc.barrier(); qc.h(counting_qr); qc.barrier()

    diag_elements = [np.exp(1j * p) for p in phases]
    if len(diag_elements) < 2**n_system:
        diag_elements.extend([1] * (2**n_system - len(diag_elements)))
    unitary_op = DiagonalGate(diag_elements)

    for i in range(n_counting):
        repetitions = 2**i
        controlled_unitary = unitary_op.control(1).power(repetitions)
        qc.append(controlled_unitary, [counting_qr[i]] + system_qr[:])
    qc.barrier()
    
    qft_gate = QFTGate(num_qubits=n_counting)
    qft_inv = qft_gate.inverse()
    qft_inv.label = "IQFT"
    qc.append(qft_inv, counting_qr)
    qc.barrier()

    qc.measure(counting_qr, counting_cr)
    qc.measure(system_qr, system_cr)
    return qc

def solve_qpe(W: np.ndarray) -> Dict:
    all_costs, tour_map = get_tours_and_costs(W)
    if not all_costs: return {'cost': float('inf'), 'path': None}

    max_cost = max(all_costs)
    theta = (2 * np.pi) / (max_cost * 1.05) if max_cost > 0 else 0
    phases = [cost * theta for cost in all_costs]
    
    n_counting_qubits = 10
    n_system_qubits = ceil(log2(len(all_costs)))

    qpe_circuit = create_qpe_search_circuit(n_counting=n_counting_qubits, n_system=n_system_qubits, phases=phases)
    simulator = AerSimulator()
    transpiled_circuit = transpile(qpe_circuit, simulator)
    result = simulator.run(transpiled_circuit, shots=4096).result()
    counts = result.get_counts()
    
    results_dict = defaultdict(list)
    for bitstring, count in counts.items():
        path_bits, cost_bits = bitstring.split()
        path_idx = int(path_bits, 2)
        if path_idx in tour_map:
            measured_int = int(cost_bits, 2)
            measured_phase = (measured_int / (2**n_counting_qubits)) * 2 * np.pi
            estimated_cost = measured_phase / theta if theta > 0 else 0
            results_dict[path_idx].append((estimated_cost, count))

    final_results = []
    for path_idx, cost_counts in results_dict.items():
        most_likely_cost = max(cost_counts, key=lambda item: item[1])[0]
        final_results.append({'path_idx': path_idx, 'tour': tour_map[path_idx], 'cost': most_likely_cost})
        
    final_results = sorted(final_results, key=lambda x: x['cost'])
    if not final_results: return {'cost': float('inf'), 'path': None}
        
    best_solution = final_results[0]
    return {'cost': best_solution['cost'], 'path': best_solution['tour']}

# ===============================================
# 4. 표준 벤치마크 실행 함수
# ===============================================
def run_single_benchmark(num_cities: int) -> Dict[str, float]:
    W = generate_random_tsp(num_cities)
    solution, elapsed_time, peak_memory = None, 0, 0
    with PeakMemSampler() as sampler:
        t0 = time.perf_counter()
        solution = solve_qpe(W)
        elapsed_time = time.perf_counter() - t0
        peak_memory = sampler.peak_rss_mb
    return {"cost": solution['cost'], "time": elapsed_time, "mem_peak_rss": peak_memory}

# ===============================================
# 5. 완전 동일한 메인 실행부
# ===============================================
def summarize_results(records: list, algorithm_name: str):
    num_runs = len(records)
    g = lambda k: [r.get(k, float('nan')) for r in records]
    c = g("cost"); t = g("time"); r = g("mem_peak_rss")
    valid_costs = [x for x in c if x != float('inf')]
    avg_cost = np.mean(valid_costs) if valid_costs else float('inf')
    std_cost = np.std(valid_costs) if valid_costs else float('nan')
    
    print("\n" + "="*40)
    print(f"📊 {algorithm_name} Benchmark Results (after {num_runs} runs)")
    print("="*40)
    print("🎯 Shortest Distance Found:")
    print(f"   - Average: {avg_cost:.4f} (±{std_cost:.4f})")
    print("-"*40)
    print("⏱️  Execution Time:")
    print(f"   - Average: {np.mean(t):.4f} seconds (±{np.std(t):.4f})")
    print("-"*40)
    print("💾 Peak Memory Usage:")
    print(f"   - Average: {np.mean(r):.4f} MB (±{np.std(r):.4f})")
    print("="*40)

def main(num_runs: int, num_cities: int):
    if num_cities > 10:
        print(f"Warning: N={num_cities} is large for QPE/Grover simulation and may cause OOM error or take a very long time.")
    records = []
    for i in range(num_runs):
        print(f"Running simulation {i+1}/{num_runs}...")
        result_dict = run_single_benchmark(num_cities)
        records.append(result_dict)
    summarize_results(records, "QPE")

if __name__ == "__main__":
    args = setup_arg_parser()
    gc.collect()
    print(" === Configuration ==="); print(f"-   Algorithm: QPE"); print(f"-   N Runs = {args.runs}"); print(f"-   N Cities = {args.cities}"); print("")
    main(num_runs=args.runs, num_cities=args.cities)