#!/bin/bash
# Double-click this to try accent colours on the real site.
# Close this window when you're done — that stops the server.

cd "$(dirname "$0")" || exit 1

PORT=8910
# if that port is busy, walk up until we find a free one
while lsof -i ":$PORT" >/dev/null 2>&1; do
    PORT=$((PORT + 1))
done

echo ""
echo "  Theme preview running at:  http://localhost:$PORT/theme-preview.html"
echo ""
echo "  Click the colour circles across the top. Your real page is underneath."
echo "  Nothing is saved to the site — it's preview only."
echo ""
echo "  CLOSE THIS WINDOW when you're finished."
echo ""

python3 -m http.server "$PORT" >/dev/null 2>&1 &
SERVER_PID=$!
# stop the server if this window is closed
trap 'kill $SERVER_PID 2>/dev/null' EXIT

sleep 1
open "http://localhost:$PORT/theme-preview.html"

wait $SERVER_PID
