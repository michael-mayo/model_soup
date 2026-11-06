# Soup

## Setup

Use the `soup` conda environment:

```bash
conda activate soup
```

Install tracked Python dependencies when needed:

```bash
python -m pip install -r requirements.txt
```

Edit `config.yaml` to configure the program. See `SYSTEM.md` for the system overview.

Run the application from this directory with:

```bash
conda run -n soup python main.py
```

All application messages use the format `[module_name] message`.

Prepare all three datasets in Ultralytics format with:

```bash
conda run -n soup python dataset_fracatlas.py
conda run -n soup python dataset_kvasir.py
conda run -n soup python dataset_bccd.py
```

Archives are stored in `scratch/downloads/`. Each dataset has its own directory
under `scratch/datasets/`, with raw files in `raw/` and Ultralytics files in
`yolo/`. Use `--force` to redownload and rebuild a dataset.

If SSL verification fails on the local network, use the explicit insecure
variant for the affected dataset:

```bash
conda run -n soup python dataset_fracatlas.py --insecure
conda run -n soup python dataset_kvasir.py --insecure
conda run -n soup python dataset_bccd.py --insecure
```
