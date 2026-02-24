#!/usr/bin/env bash

set -euo pipefail

FRONTEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$FRONTEND_DIR"

echo "[frontend-tests] compile test bundle"
rm -rf tests-dist
npx tsc -p tsconfig.tests.json --noEmit
npx esbuild tests/test_api_base.ts --bundle --platform=node --format=esm --outfile=tests-dist/test_api_base.mjs >/dev/null
npx esbuild tests/test_query_client.ts --bundle --platform=node --format=esm --outfile=tests-dist/test_query_client.mjs >/dev/null
npx esbuild tests/test_route_guards.ts --bundle --platform=node --format=esm --outfile=tests-dist/test_route_guards.mjs >/dev/null

echo "[frontend-tests] run test_api_base"
node tests-dist/test_api_base.mjs

echo "[frontend-tests] run test_query_client"
node tests-dist/test_query_client.mjs

echo "[frontend-tests] run test_route_guards"
node tests-dist/test_route_guards.mjs

echo "[frontend-tests] all tests passed"
