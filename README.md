# 🎨 Mondriaan Detector

Een machine learning project voor het detecteren en classificeren van Mondriaan schilderijen.

## 📋 Vereisten

- Python 3.12 of hoger
- pip (Python package manager)

## 🚀 Installatie

### 1. Virtual Environment Aanmaken

```bash
# Maak een virtual environment aan
python -m venv venv

# Activeer de virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Windows (CMD):
.\venv\Scripts\activate.bat

# macOS/Linux:
source venv/bin/activate
```

### 2. Dependencies Installeren

```bash
pip install -r requirements.txt
```

### 3. Settings Configureren

Maak een bestand genaamd `settings.py` in de `src` map met de volgende inhoud:

```python
from pathlib import Path

# Pad naar het model
MODEL_PATH = "mondriaan_svm_model.joblib"

# Optioneel: pad naar test afbeelding
TEST_IMAGE_PATH = Path("test_set") / "example_image.jpg"
```

## 💻 Gebruik

### Hoofdprogramma

```bash
python src/main.py
```

Dit opent een GUI waar je kunt kiezen tussen:
- 📷 **Camera**: Maak direct een foto met je webcam
- 📁 **Bestand**: Selecteer een afbeelding van je computer

### Test Algoritme

```bash
python src/test_algoritme.py
```

Test het algoritme op meerdere afbeeldingen in een geselecteerde map.

## 📁 Project Structuur

```
mondriaan-detector/
├── src/                    # Source code
│   ├── main.py            # Hoofdprogramma
│   ├── main_helpers.py    # Helper functies en GUI
│   ├── test_algoritme.py  # Test script
│   └── ...
├── data/                  # Data bestanden
├── test_set/             # Test afbeeldingen
├── requirements.txt      # Python dependencies
└── README.md            # Deze file
```
