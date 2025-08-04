from pathlib import Path
import time, requests, re
from huggingface_hub import HfApi, snapshot_download

try:                                      # ← NEW
    # ≥ 0.20
    from huggingface_hub.utils._errors import HfHubHTTPError
except ModuleNotFoundError:               # fallback for older versions
    from huggingface_hub.utils import HfHubHTTPError

REPO_ID   = "mission-impossible-lms/deterministic-shuffle-s21-gpt2"
OUT_ROOT  = Path("/home/s2587034/pretrained_checkpoints/babylm_models_en/babylm_shuffle_deterministic21_100M_randinit/runs/babylm_shuffle_deterministic21_100M_randinit_seed0")

MAX_RETRIES  = 5        # network retries
BACKOFF_BASE = 2        # seconds → 2,4,8…

api = HfApi()
refs = api.list_repo_refs(REPO_ID)         # grab *all* branches & tags
ckpt_refs = sorted(
    (r.name for r in refs.branches + refs.tags if re.match(r"checkpoint-\d+", r.name)),
    key=lambda x: int(x.split("-")[1]),
)                                           # e.g. ['checkpoint-100', ..., 'checkpoint-3000']

OUT_ROOT.mkdir(parents=True, exist_ok=True)


def download_with_retry(revision: str, target_dir: Path):
    """snapshot_download that retries on transient network errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            snapshot_download(
                repo_id=REPO_ID,
                revision=revision,          # correct remote ref ✅
                local_dir=target_dir,
                resume_download=True,
                local_dir_use_symlinks=False,   # make each folder self-contained
            )
            return                          # success
        except (requests.exceptions.ConnectTimeout,
                requests.exceptions.ConnectionError,
                HfHubHTTPError) as err:
            # 404? stop immediately – ref truly missing
            if isinstance(err, HfHubHTTPError) and err.response.status_code == 404:
                print(f"✗ {revision} not found – skipping")
                return
            if attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_BASE ** attempt
            print(f"[{revision}] hiccup ({err.__class__.__name__}) – retry {attempt}/{MAX_RETRIES} in {wait}s")
            time.sleep(wait)


for ref in ckpt_refs:
    step = ref.split("-")[1]
    dest = OUT_ROOT / f"checkpoint-{step}"
    if dest.exists():
        continue
    download_with_retry(revision=ref, target_dir=dest)
    print(f"✓ saved {ref} → {dest}")
