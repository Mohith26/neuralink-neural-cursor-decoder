#!/usr/bin/env bash
# Download ONE session of the O'Doherty et al. 2017 NHP reaching dataset
# (Zenodo record 583331, DOI 10.5281/zenodo.583331) into data/.
# The dataset is NOT committed to git (see .gitignore); this fetches it on demand.
set -euo pipefail

RECORD="583331"
FILE="indy_20161005_06.mat"          # smallest session (84 MB); sorted units + cursor kinematics
EXPECTED_MD5="5ea300952642e0fc54245144499db9bb"
URL="https://zenodo.org/api/records/${RECORD}/files/${FILE}/content"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data"
mkdir -p "$DIR"
DEST="$DIR/$FILE"

if [[ -f "$DEST" ]]; then
  echo "Already present: $DEST"
else
  echo "Downloading $FILE (~84 MB) from Zenodo record $RECORD ..."
  curl -L --fail --retry 3 -o "$DEST" "$URL"
fi

echo "Verifying MD5 ..."
if command -v md5sum >/dev/null 2>&1; then
  ACTUAL=$(md5sum "$DEST" | awk '{print $1}')
else
  ACTUAL=$(md5 -q "$DEST")
fi

if [[ "$ACTUAL" != "$EXPECTED_MD5" ]]; then
  echo "MD5 MISMATCH: got $ACTUAL expected $EXPECTED_MD5" >&2
  exit 1
fi
echo "OK: $DEST (md5 $ACTUAL)"
