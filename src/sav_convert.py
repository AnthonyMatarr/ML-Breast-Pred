import pyreadstat
import pandas as pd
from pathlib import Path
from config import BASE_PATH
import subprocess
import multiprocessing as mp
import time


def convert_single_threaded(args):
    """Convert one file - designed for parallel execution"""
    sav_file, output_dir = args
    process_name = mp.current_process().name

    print(f"\n[{process_name}] Converting: {sav_file.name}")
    print(f"[{process_name}] Started: {time.strftime('%I:%M:%S %p')}")

    start_time = time.time()

    try:
        df, _ = pyreadstat.read_sav(str(sav_file))

        output = output_dir / f"{sav_file.stem}.parquet"
        df.to_parquet(output)

        elapsed = time.time() - start_time
        print(f"[{process_name}] ✓ Done: {output.name}")
        print(f"[{process_name}] {df.shape[0]:,} rows, {df.shape[1]} cols")
        print(f"[{process_name}] Time: {elapsed/60:.1f} minutes")
        return (sav_file.name, True)

    except Exception as e:
        print(f"[{process_name}] ❌ Error: {e}")
        return (sav_file.name, False)


if __name__ == "__main__":
    # Start caffeinate to prevent sleep
    print("Starting caffeinate to prevent sleep...")
    caffeinate_process = subprocess.Popen(
        ["caffeinate", "-i", "-w", str(subprocess.os.getpid())],  # type: ignore
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        data_dir = BASE_PATH / "data" / "raw"
        output_dir = BASE_PATH / "data" / "raw" / "parquet"
        output_dir.mkdir(exist_ok=True)

        files_to_convert = [
            "NSQIP_15.sav",
            "NSQIP_16.sav",
            "NSQIP_17.sav",
            "NSQIP_18.sav",
        ]

        # Remove already-converted files
        sav_files = []
        for filename in files_to_convert:
            sav_file = data_dir / filename
            output = output_dir / f"{sav_file.stem}.parquet"
            if output.exists():
                print(f"⚠️  Skipping {filename} - already exists")
            else:
                sav_files.append(sav_file)

        if not sav_files:
            print("All files already converted!")
            exit(0)

        print(f"\n{'='*60}")
        print(f"PARALLEL CONVERSION - {len(sav_files)} files")
        print(f"Using {min(len(sav_files), mp.cpu_count())} CPU cores")
        print(f"Sleep prevention: ENABLED")
        print(f"{'='*60}\n")

        # Create jobs
        jobs = [(sav_file, output_dir) for sav_file in sav_files]

        # Process in parallel
        num_processes = min(len(sav_files), mp.cpu_count())
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(convert_single_threaded, jobs)

        # Summary
        successful = sum(1 for _, success in results if success)
        failed = len(results) - successful

        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        print(f"✓ Successful: {successful} files")
        if failed > 0:
            print(f"❌ Failed: {failed} files")
        print(f"{'='*60}")

    finally:
        caffeinate_process.terminate()
        print("\nCaffeinate stopped.")
