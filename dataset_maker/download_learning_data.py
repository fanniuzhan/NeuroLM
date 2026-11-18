"""Download small, public NeuroLM data sources; standard library and curl only.

Run from any directory. Existing files are checked against PhysioNet SHA256;
interrupted downloads use .part files and can be resumed by rerunning.
No signal processing or model training is performed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "eegmat-1.0.0": "https://physionet.org/files/eegmat/1.0.0/",
    "hmc-1.1": "https://physionet.org/files/hmc-sleep-staging/1.1/",
    "eegmmidb-1.0.0": "https://physionet.org/files/eegmmidb/1.0.0/",
}


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    # Restart curl between attempts so -C - recalculates the saved offset.
    # curl's internal retries can restart a large transfer at its original offset.
    for attempt in range(4):
        result = subprocess.run([
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--connect-timeout", "20", "--max-time", "1800",
            "--speed-time", "60", "--speed-limit", "1024",
            "--continue-at", "-", "--output", str(part), url,
        ])
        if result.returncode == 0:
            break
        if attempt == 3:
            result.check_returncode()
    part.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def edf_info(path):
    """Read only the EDF header; check declared length without loading EEG."""
    with path.open("rb") as f:
        header = f.read(256)
        n = int(header[252:256])
        header_bytes = int(header[184:192])
        records = int(header[236:244])
        seconds = float(header[244:252])
        fields = f.read(256 * n)
    def field(offset, width):
        return [fields[offset + width*i:offset + width*(i+1)].decode("ascii").strip()
                for i in range(n)]
    samples = [int(s) for s in field(216*n, 8)]
    expected_bytes = header_bytes + records * 2 * sum(samples)
    if records < 0 or path.stat().st_size != expected_bytes:
        raise RuntimeError(f"EDF length mismatch: {path}")
    return {"duration_seconds": records * seconds if seconds else None,
            "signal_labels": field(0, 16), "physical_units": field(96*n, 8),
            "sampling_rates_hz": [s / seconds if seconds else None for s in samples]}


def selected(dataset, name):
    if dataset == "eegmat-1.0.0":
        return True  # Entire 36-subject Workload dataset, including metadata.
    if dataset == "hmc-1.1":
        return "/" not in name or any(
            name.startswith("recordings/" + subject + ".") or
            name.startswith("recordings/" + subject + "_")
            for subject in ("SN001", "SN103", "SN128")
        )
    return "/" not in name or any(
        name in (f"S{s:03d}/S{s:03d}R{r:02d}.edf",
                 f"S{s:03d}/S{s:03d}R{r:02d}.edf.event")
        for s in (1, 2, 3) for r in (1, 2, 4)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    tasks = []
    for dataset, base in SOURCES.items():
        folder = args.data_dir / "raw" / dataset
        checksums = folder / "SHA256SUMS.txt"
        if not checksums.exists() and not args.verify_only:
            fetch(base + "SHA256SUMS.txt", checksums)
        for line in checksums.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            name = name.lstrip("*")
            if selected(dataset, name):
                tasks.append((dataset, name, digest, base, folder))

    def download(task):
        dataset, name, expected, base, folder = task
        path = folder / name
        if not path.exists() or sha256(path) != expected:
            if args.verify_only:
                raise RuntimeError(f"Missing or corrupt: {path}")
            fetch(base + name, path)
            if sha256(path) != expected:
                raise RuntimeError(f"SHA256 mismatch: {path}")
        print(f"OK {dataset}/{name} ({path.stat().st_size} bytes)", flush=True)
        result = {"dataset": dataset, "file": name, "url": base + name,
                "bytes": path.stat().st_size, "sha256": expected,
                "verified_against": "PhysioNet SHA256SUMS.txt"}
        if path.suffix == ".edf":
            result["edf_header"] = edf_info(path)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(download, tasks))
    report = args.data_dir / "download_manifest.json"
    report.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Verified {len(records)} files, {sum(r['bytes'] for r in records):,} bytes")
    print(f"Manifest: {report}")


if __name__ == "__main__":
    main()
