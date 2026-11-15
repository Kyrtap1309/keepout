#!/bin/sh
# The init service of compose.yaml: give the data volume the demo's store
# (built into the image by tools/dashboard_seed.py --store) unless it has a
# store already. The copy is made under temporary names and renamed into
# place, the database last, so an interrupted copy leaves no store that
# looks whole. KEEPOUT_SEED_DIR and KEEPOUT_DATA_DIR override the paths
# (for the tests).
set -eu

seed="${KEEPOUT_SEED_DIR:-/opt/keepout/seed}"
data="${KEEPOUT_DATA_DIR:-/data}"
store="$data/keepout.db"

if [ -e "$store" ]; then
    echo "store kept: $store"
    exit 0
fi

partial="$data/.init-keepout.db"
rm -rf "$partial" "$partial.catalogs" "$store.catalogs"
cp "$seed/keepout.db" "$partial"
cp -R "$seed/keepout.db.catalogs" "$partial.catalogs"
mv "$partial.catalogs" "$store.catalogs"
mv "$partial" "$store"
echo "demo store copied to $store"
