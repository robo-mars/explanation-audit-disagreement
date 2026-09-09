from pathlib import Path
from src.pipeline import Paths

if __name__ == "__main__":
    paths = Paths.from_base(Path(__file__).resolve().parents[1])
    paths.ensure()
    (paths.logs / 'pilot_day1.txt').write_text('Pilot scaffold initialized\n')
    print('Pilot scaffold ready')
