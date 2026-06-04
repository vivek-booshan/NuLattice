def main():
    import jax
    import os

    coordinator_addr = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "localhost") + ":5000"
    jax.distributed.initialize(
        coordinator_address=coordinator_addr,
        num_processes=int(os.environ["SLURM_NNODES"]), # 2 nodes = 2 processes total
        process_id=int(os.environ["SLURM_NODEID"]),    # Global node rank (0, 1)
        cluster_detection_method="deactivate"          # Prevents JAX from enforcing 1 GPU per process
    )

    total_devices = jax.device_count()
    local_devices = jax.local_device_count()
    # process_id = jax.process_index()
    total_processes = jax.process_count()
    print("--- Cluster Report ---")
    print(f"Total Processes (Nodes/Tasks): {total_processes}")
    print(f"Total GCDs across cluster: {total_devices}")
    print(f"Local GCDs per Node: {local_devices}")
    print(f"Devices: {jax.devices()}")
    print("----------------------")

if __name__ == "__main__":
    main()
