"""The junk model's record facts for a rescue corpus (search/build_quality.py, pointed at it), into corpus/quality/.

    search/.venv/bin/python pipeline/rescue/quality.py RESCUE_DIR

Without them the search service treats rescued papers as unknown records and ranks them lower (the daily updates
measured a median P(junk) of 0.10 instead of 0.02 for comparable articles).
"""
import os, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "search"))
import build_quality as bq  # noqa: E402

corpus = Path(sys.argv[1]) / "corpus"
bq.CORPUS = corpus
tmp = corpus / "quality-build"
shutil.rmtree(tmp, ignore_errors=True)
bq.build(tmp, threads=8, memory="24GB", passes=4, raw_glob=str(bq.RAW / "updated_date=*" / "*.parquet"))
shutil.rmtree(corpus / "quality", ignore_errors=True)
os.replace(tmp, corpus / "quality")
print("quality facts in", corpus / "quality")
