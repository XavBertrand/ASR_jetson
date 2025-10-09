import sys
from pathlib import Path

# On définit la racine projet comme point de référence
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Variable globale pratique
PROJECT_ROOT = ROOT
