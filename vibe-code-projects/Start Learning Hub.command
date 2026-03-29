#!/bin/bash
cd "$(dirname "$0")"
echo ""
echo "  Starting Learning Hub..."
echo ""
open "http://localhost:3000"
node image-server.js
