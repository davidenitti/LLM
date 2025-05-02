import subprocess
import time
import random
import traceback

def wait_until_gpu_free(gpu_id=0, threshold=20, check_interval=60):
    """
    Waits until the GPU usage drops below the specified threshold.

    :param gpu_id: The ID of the GPU to monitor.
    :param threshold: The utilization percentage below which the GPU is considered free.
    :param check_interval: The time interval (in seconds) between checks.
    """
    while True:
        try:
            # Run nvidia-smi command and get the GPU utilization
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True
            )
            utilization = int(result.stdout.split("\n")[gpu_id].strip())

            # If utilization is below the threshold, exit the loop
            if utilization < threshold:
                print(f"GPU {gpu_id} is free (Utilization: {utilization}%)")
                break

            print(f"GPU {gpu_id} busy (Utilization: {utilization}%), waiting...")
            time.sleep(check_interval//2 + random.randint(1,check_interval//2))
        
        except Exception as e:
            print(f"Error checking GPU status: {e}")
            traceback.print_exc()
            print(result.stdout)
            time.sleep(check_interval//2 + random.randint(1,check_interval//2))


def wait_until_gpu_memory_free(gpu_id=0, min_free_memory=4000, check_interval=60):
    """
    Waits until the GPU has at least the specified amount of free memory.

    :param gpu_id: The ID of the GPU to monitor.
    :param min_free_memory: The minimum free memory (in MB) required.
    :param check_interval: The time interval (in seconds) between checks.
    """
    while True:
        try:
            # Run nvidia-smi command to get free memory
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True
            )
            free_memory = int(result.stdout.split("\n")[gpu_id].strip())

            # If free memory is greater than or equal to the threshold, exit the loop
            if free_memory >= min_free_memory:
                print(f"GPU {gpu_id} is free (Free Memory: {free_memory} MB)")
                break

            print(f"GPU {gpu_id} busy (Free Memory: {free_memory} MB), needed {min_free_memory} MB waiting...")
            time.sleep(check_interval//2 + random.randint(1,check_interval//2))

        except Exception as e:
            print(f"Error checking GPU memory status: {e}")
            traceback.print_exc()
            print(result.stdout)
            time.sleep(check_interval//2 + random.randint(1,check_interval//2))

if __name__ == "__main__":
    wait_until_gpu_free()
    wait_until_gpu_memory_free(min_free_memory=6*1200)