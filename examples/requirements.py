import math
import argparse

def calculate_requirements(L, n_occ, max_nodes=1000, dtype_bytes=8):
    n_stat = (L**3) * 4
    n_virt = n_stat - n_occ
    gb_size = (1024 ** 3)
    f_pp_gb = (n_virt**2 * dtype_bytes) / gb_size
    v_mmhh_gb = (n_stat**2 * n_occ**2 * dtype_bytes) / gb_size
    v_pphh_gb = (n_virt**2 * n_occ**2 * dtype_bytes) / gb_size
    v_pph_gb = (n_virt**2 * n_occ * dtype_bytes) / gb_size
    v_phhh_gb = (n_virt * n_occ**3 * dtype_bytes) / gb_size
    W_gb = (n_stat * n_stat) / gb_size
    P_gb = (n_virt * n_stat) / gb_size
    H_gb = (n_occ * n_stat) / gb_size

    print(f"--- Lattice Report: L={L} ---")
    print(f"N_virt: {n_virt} | N_occ: {n_occ}")
    print(f"Dense f_pp Block: {f_pp_gb:.2f} GiB")
    print(f"Dense V_mmhh Block: {v_mmhh_gb:.2f} GiB")
    print(f"Dense V_pphh Block: {v_pphh_gb:.2f} GiB")
    print(f"Dense V_pph Block: {v_pph_gb:.2f} GiB")
    print(f"Dense V_phhh Block: {v_phhh_gb:.2f} GiB")
    # print(f"Dense V_hhhh Block: {v_hhhh_gb:.2f} GiB")
    print(f"THC W tensor: {W_gb:.2f} GiB")
    print(f"THC P tensor: {P_gb:.2f} GiB")
    print(f"THC H tensor: {H_gb:.2f} GiB")
    print("-" * 50)
    print(f"{'Nodes':>5} | {'GPN':>3} | {'Total GPUs':>10} | {'Shard Size':>10} | {'Vpphh Size':>10}")
    print("-" * 50)

    found_any = False
    for nodes in range(1, max_nodes + 1):
        for gpn in range(1,9):
            total_gpus = nodes * gpn
            if int(math.sqrt(total_gpus))**2 != total_gpus:
                continue
            gpu_sqrt = int(math.sqrt(total_gpus))
            if n_virt % gpu_sqrt == 0:
                shard_len = n_virt // total_gpus
                shard_mem = v_pphh_gb / total_gpus
                print(f"{nodes:5d} | {gpn:3d} | {total_gpus:10d} | {shard_len:10d} | {shard_mem:10f}")
                found_any = True

    if not found_any:
        print(f"No configurations found. N_virt={n_virt} is prime(-ish).")
        print("You must add padding to allow sharding.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=6)
    parser.add_argument("--occ", type=int, default=16)
    parser.add_argument("--nodes", type=int, default=1000)
    args = parser.parse_args()
    calculate_requirements(args.L, args.occ, args.nodes)
