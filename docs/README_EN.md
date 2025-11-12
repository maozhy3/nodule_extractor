# Lung Nodule Size Extraction Tool

[中文](../README.md) | English

An automated lung nodule size extraction tool based on llama.cpp and Qwen medical model, supporting intelligent extraction of maximum lesion diameter from imaging reports.

> **Platform Support**: Currently only supports Windows platform. Linux/Mac users can run from source code, but the packaged exe version is only available for Windows.

## Features

- 🖥️ Both GUI and command-line interfaces
- 🔍 Intelligent text preprocessing and numerical extraction
- 🚀 Single-process/multi-process parallel inference
- 💾 Checkpoint resume (continue processing after interruption)
- 📊 Batch processing of Excel data
- ⚙️ Flexible configuration management
- 📦 Package as standalone exe

## Interface Preview

![GUI Interface](images/gui-screenshot.png)

---

## Quick Start

### 1. Install VC++ Runtime (Windows Required)

Download and install: https://aka.ms/vs/17/release/vc_redist.x64.exe

Or place it in the `_vcredist/` directory, the program will install it automatically.

### 2. Install Python Dependencies

```bash
# Create virtual environment (recommended)
conda create -n lung-nodule python=3.10
conda activate lung-nodule

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Model

> **Note**: The current Release version does not include model files, they need to be downloaded separately.

```bash
python download_model.py
```

Recommended models:
- **Qwen2.5-3B** (~2GB) - Fast, suitable for testing
- **Qwen2.5-7B** (~4.7GB) - Better performance, recommended

### 4. Run the Program

**GUI (Recommended)**:
```bash
python gui.py
```

**Command Line**:
```bash
python main.py
```

---

## Configuration

### Configuration File Priority

The project supports multi-layer configuration override mechanism, priority from low to high:

```
config.py (default configuration)
    ↓ override
config_private.py (private configuration) ⭐ Recommended
    ↓ override
external config.py (after packaging)
```

### Configuration Files

#### 1. config.py (Default Configuration)
- **Location**: Project root directory
- **Purpose**: Default project configuration, committed to git
- **Use Case**: Team-shared default settings

#### 2. config_private.py (Private Configuration) ⭐ Recommended
- **Location**: Project root directory
- **Purpose**: Personal private configuration, **not committed to git**
- **Use Cases**:
  - Personal model paths
  - Personal hardware configuration (CPU threads, GPU layers)
  - Special test configurations

#### 3. External config.py (After Packaging)
- **Location**: Same directory as exe file
- **Purpose**: External configuration after packaging
- **Use Case**: Users can modify configuration without repackaging

### Quick Start: Create Private Configuration

**Recommended to use `config_private.py` for personal configuration** to avoid git conflicts from modifying `config.py`.

1. **Copy example file**:
   ```bash
   copy config_private.py.example config_private.py
   ```

2. **Edit config_private.py** (only write the configuration items you want to override):
   ```python
   # Enable feature extraction
   ENABLE_FEATURE_EXTRACTION = True
   
   # Use your model path
   MODEL_PATHS = [
       str(_ROOT / "models" / "qwen-medical-lora-251106-q4_k_m.gguf")
   ]
   
   # Your hardware configuration
   LLAMA_N_THREADS = 16
   LLAMA_N_GPU_LAYERS = 33
   ```

3. **Run the program**:
   ```bash
   python main.py
   ```
   
   The program will automatically load your private configuration and display:
   ```
   ✓ Loaded private configuration: config_private.py
   ```

### Main Configuration Items

```python
# Input/Output
EXCEL_PATH = "tests/test.xlsx"           # Input file
OUTPUT_PATH = "tests/test_results.xlsx"  # Output file

# Model path
MODEL_PATHS = ["models/qwen2.5-7b-instruct-q4_k_m.gguf"]

# Performance settings
LLAMA_N_THREADS = 8               # CPU threads
LLAMA_N_GPU_LAYERS = 0            # GPU layers (0=CPU only)
PROCESS_POOL_MAX_WORKERS = 1      # Number of processes (1=single process)
CHECKPOINT_SAVE_INTERVAL = 5000   # Checkpoint save interval

# Feature extraction configuration (new)
ENABLE_FEATURE_EXTRACTION = False  # Enable feature extraction
FEATURE_EXTRACTION_MODEL_PATH = None  # Model for feature extraction
```

### Performance Recommendations

- Low-end (4 cores, 8GB): `PROCESS_POOL_MAX_WORKERS = 1, LLAMA_N_THREADS = 4`
- Mid-range (8 cores, 16GB): `PROCESS_POOL_MAX_WORKERS = 2, LLAMA_N_THREADS = 4`
- High-end (16 cores, 32GB): `PROCESS_POOL_MAX_WORKERS = 4, LLAMA_N_THREADS = 4`

### Configuration Examples

#### Example 1: Developer's Personal Configuration
```python
# config_private.py

# My test data
EXCEL_PATH = _ROOT / "my_test.xlsx"

# My GPU configuration
LLAMA_N_THREADS = 4
LLAMA_N_GPU_LAYERS = 33

# Enable feature extraction
ENABLE_FEATURE_EXTRACTION = True
```

#### Example 2: Quick Test Configuration
```python
# config_private.py

# Small dataset for testing
EXCEL_PATH = _ROOT / "test_5_samples.xlsx"

# Quick checkpoint
CHECKPOINT_SAVE_INTERVAL = 5

# Verbose logging
LLAMA_VERBOSE = True
```

### Best Practices

✅ **Recommended**:
- Use `config_private.py` for personal configuration
- Only write configuration items you want to override in `config_private.py`
- Keep `config_private.py.example` updated as a template

❌ **Not Recommended**:
- Directly modify `config.py` for personal configuration (affects git commits)
- Commit `config_private.py` to git (already excluded in .gitignore)

---

## Usage

### Data Format

Excel file must contain `yxbx` column (imaging findings):

| yxbx |
|------|
| Nodule in right upper lobe, size about 8mm |
| Ground-glass opacity in left lower lobe, diameter 1.2cm |

### Output Results

New column `pred_model_name` added:

| yxbx | pred_qwen2.5-7b |
|------|-----------------|
| Nodule in right upper lobe, size about 8mm | 8 |
| Ground-glass opacity in left lower lobe, diameter 1.2cm | 12 |

### Checkpoint Resume

- Press `Ctrl+C` to interrupt, checkpoint automatically saved to `checkpoints/`
- Run again to continue from last interruption
- Checkpoint automatically cleared after completion

---

## Packaging and Deployment

> **Note**: Packaging is only supported on Windows platform.

### Prerequisites

1. **Windows Operating System** (packaging only supported on Windows)
2. **Conda Environment**: `node_extractor` environment installed and activated
3. **PyInstaller**: Installed in the environment
4. **VC++ Runtime**: `_vcredist/vc_redist.x64.exe` (auto-installed on first run)

### Packaging Methods

#### Method 1: GUI Version (Recommended)

GUI version provides graphical interface, suitable for most users.

```bash
# Run build script
build_gui.bat
```

**Output Directory**: `dist/医疗影像报告预测工具/`

**Included Files**:
- `医疗影像报告预测工具.exe` - Main program (GUI)
- `_internal/` - Dependencies
- `models/` - Model files directory (copy models manually)
- `_vcredist/` - VC++ runtime
- `test.xlsx` - Sample data file
- `config_example.py` - Configuration example
- `config_private.py.example` - Private configuration template

#### Method 2: Command-Line Version

Command-line version suitable for automation scripts or server environments.

```bash
# Run build script
build_main.bat
```

**Output Directory**: `dist/医疗影像报告预测工具-CLI/`

### Packaging Configuration Files

#### gui.spec

PyInstaller configuration file for GUI version.

**Key Settings**:
- `console=False` - No console window
- Includes `config.py`, `config_loader.py`, `core.py` and other core modules
- Auto-collects llama_cpp DLL files

#### main.spec

PyInstaller configuration file for command-line version.

**Key Settings**:
- `console=True` - Show console window
- Includes all core modules and dependencies

### Deploy to Offline Computer

> **Important**:
> - Release version does not include model files, need to download manually and place in `models/` directory
> - Windows platform only

1. Copy the entire `dist/医疗影像报告预测工具/` folder to target Windows computer
2. **Manually download model files** and place in `models/` directory (not included in Release)
3. Ensure `_vcredist/vc_redist.x64.exe` exists
4. Double-click `医疗影像报告预测工具.exe` to run

### External Configuration

Create `config.py` in the same directory as exe to override default settings without repackaging.

Packaged program supports three-layer configuration override:

```
Built-in config.py (included during packaging)
    ↓ override
config_private.py (if exists)
    ↓ override
External config.py (same directory as exe)
```

### Post-Packaging Testing

#### Test Checklist

- [ ] Program starts normally
- [ ] Can load model files
- [ ] Can read Excel files
- [ ] Can predict and save results normally
- [ ] Stop function works properly
- [ ] Checkpoint function works (resume after interruption)
- [ ] Feature extraction works (if enabled)
- [ ] External configuration file loads properly

#### Test Steps

1. **Basic Function Test**:
   - Double-click to run exe
   - Select input file (test.xlsx)
   - Select model file
   - Click "Start Prediction"
   - Wait for completion, check output file

2. **Stop and Resume Test**:
   - Start prediction
   - Click "Stop" button
   - Check if checkpoint is saved
   - Run again, confirm resume from checkpoint

3. **External Configuration Test**:
   - Create config.py in same directory as exe
   - Modify configuration (e.g., thread count)
   - Run program, confirm configuration takes effect

### Common Packaging Issues

#### 1. Packaging Failed: Module Not Found

**Solution**: Check if `.spec` file's `hiddenimports` includes all necessary modules.

Required modules: `pandas`, `openpyxl`, `llama_cpp`, `tqdm`, `tkinter`, `config_loader`, `core`

#### 2. Runtime DLL Not Found

**Solution**:
- Ensure llama_cpp DLL files are correctly collected
- Check `binaries` configuration in `.spec` file
- Ensure VC++ runtime is installed

#### 3. Configuration File Not Working

**Solution**:
- Check if configuration file name is correct (`config.py` or `config_private.py`)
- Check configuration file location (same directory as exe)
- View program startup log to confirm if configuration is loaded

#### 4. Model Loading Failed

**Solution**:
- Ensure model files are in `models/` directory
- Check model file path configuration
- Confirm model file format is correct (.gguf)

---

## Development Guide

### Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```


### Project Structure

```
.
├── main.py              # Command-line entry
├── gui.py               # GUI entry
├── core.py              # Core logic
├── config.py            # Configuration file
├── config_loader.py     # Configuration loader
├── download_model.py    # Model downloader
├── requirements.txt     # Core dependencies
├── requirements-dev.txt # Development dependencies
├── tests/               # Test files (unit tests, test data, test scripts)
├── models/              # Model directory
└── checkpoints/         # Checkpoint directory
```


### Run Tests

```bash
# Windows
cd tests
run_tests.bat

# Linux/Mac
cd tests
./run_tests.sh

# Or use pytest directly
pytest tests/ -v
```

### Code Formatting

```bash
# Auto format
format_code.bat

# Or manually
black *.py tests/
ruff check . --fix
```

---

## FAQ

**Q: Failed to install llama-cpp-python?**  
A: Make sure VC++ runtime is installed

**Q: Out of memory?**  
A: Set `PROCESS_POOL_MAX_WORKERS = 1`

**Q: How to use GPU?**  
A: Install CUDA version of llama-cpp-python, set `LLAMA_N_GPU_LAYERS = 35`

**Q: Can I delete checkpoint files?**  
A: They are automatically cleared after task completion, or you can manually delete the `checkpoints/` directory

**Q: How to verify installation?**  
A: Run `python verify_setup.py`

---

## Changelog

### v1.2.0 (2024-11-12)
- ✅ Optimized packaging process: no longer automatically copies all model files during packaging
- ✅ Improved user experience: users can manually select models to deploy as needed
- ✅ Reduced release package size: avoids packaging unnecessary large model files

### v1.1.0 (2024-11-10)
- ✅ Added standard dependency management (requirements.txt)
- ✅ Standardized code style and type annotations
- ✅ Refactored configuration loading logic
- ✅ Added comprehensive unit tests (52 test cases)
- ✅ Improved documentation and development tools

### v1.0.0 (2024-11-06)
- ✅ Initial release
- ✅ GUI and command-line interface
- ✅ Batch prediction and checkpoint resume
- ✅ Multi-process parallel processing

---

## License

This project is licensed under the [MIT License](LICENSE).

**Disclaimer**: This tool is for learning and research purposes only and should not be used for clinical diagnosis. Users assume all risks and responsibilities associated with using this tool.
