# grpc-supersonic

A lightweight model runtime pipeline for loading and executing PyTorch models from multiple formats using a Rust backend.

The goal of this project is to build a flexible inference pipeline capable of supporting multiple PyTorch model formats while keeping the runtime fast and minimal.

Currently supported formats:

* **TorchScript (`.pt`)**

---

## Project Structure

```
grpc_supersonic/
│
├─ backend/
│   ├─ model_init/
│   │   ├─ src/          # Rust runtime code
│   │   └─ models/       # generated models (ignored by git)
│   │
│   └─ mnist_init.py     # model export pipeline
│
├─ requirements-runtime.txt
├─ requirements-export.txt
└─ README.md
```

---

## Model Formats

| Format | Description             | Runtime                       |
| ------ | ----------------------- | ----------------------------- |
| `.pt`  | TorchScript model       | Loaded via `tch-rs`           |
| `.pt2` | PyTorch ExportedProgram | Supported via export pipeline (in progress)|

TorchScript models can be executed directly in the Rust runtime.

ExportedProgram models (`.pt2`) are produced to support newer PyTorch compilation pipelines.

---

## Setup

This project uses **two Python environments**:

* `sonic-runtime`
* `sonic-export`

### Runtime environment

Used for compatibility with the Rust runtime.

```
pip install -r requirements-runtime.txt
```

---

### Export environment

Used to generate model artifacts.

```
pip install -r requirements-export.txt
```

---

## Generating Models

Models are not stored in the repository due to size constraints.

To generate them:

```
python backend/mnist_init.py
```

This will produce:

```
backend/model_init/models/
    mnist.pt
    mnist.pt2
```

---

## Running the Rust Runtime

Build the runtime:

```
cargo build
```

Run the model loader:

```
cargo run
```

The runtime detects model format automatically and loads the appropriate backend.

---

## Notes

Large model artifacts are ignored via `.gitignore`:

```
*.pt
*.pt2
*.pte
```

Models should be regenerated locally instead of committed.

---

## License

MIT
