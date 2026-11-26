#!/bin/sh
# The init service of compose.yaml: give the data volume the demo's store
# (built into the image by tools/dashboard_seed.py --store) unless it has a
# store already, and check that a store it has is at this keepout's schema,
# so that docker compose up fails here, with the fix, rather than leave the
# dashboard and the worker restarting. The copy is made under temporary
# names and renamed into place, the database last, so an interrupted copy
# leaves no store that looks whole.
#
# Usage: init-store.sh [SEED_DIR [DATA_DIR]]
set -eu

seed="${1:-/opt/keepout/seed}"
data="${2:-/data}"
store="$data/keepout.db"

if [ -e "$store" ]; then
    # Opens the store read-only; fails on another schema, or on a file that
    # is no store. Only a store of an older keepout gets the upgrade hint.
    if ! error="$(keepout jobs list --store "$store" 2>&1 >/dev/null)"; then
        echo "$error" >&2
        case "$error" in
        *"made by an older keepout"*)
            echo "upgrade it: docker compose stop dashboard worker &&" \
                "docker compose run --rm --no-deps worker keepout store upgrade" \
                "--store $store" >&2
            ;;
        esac
        exit 1
    fi
    echo "store kept: $store"
    exit 0
fi

if [ -e "$store.catalogs" ]; then
    echo "refusing to replace $store.catalogs, found without $store:" \
        "move it away or restore $store" >&2
    exit 1
fi
partial="$data/.init-keepout.db"
rm -rf "$partial" "$partial.catalogs"
cp "$seed/keepout.db" "$partial"
cp -R "$seed/keepout.db.catalogs" "$partial.catalogs"
mv "$partial.catalogs" "$store.catalogs"
mv "$partial" "$store"
echo "demo store copied to $store"
