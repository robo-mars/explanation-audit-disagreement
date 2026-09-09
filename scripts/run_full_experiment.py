from pathlib import Path
from src.pipeline import Paths

if __name__ == "__main__":
    paths = Paths.from_base(Path(__file__).resolve().parents[1])
    paths.ensure()
    (paths.results / 'placeholder.txt').write_text('Full experiment driver placeholder\n')
    print(f'Run from: {paths.base}')
