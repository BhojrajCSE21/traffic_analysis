#!/bin/bash
# Traffic Analytics Platform - Run Script

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Traffic Analytics Platform...${NC}"

# Navigate to platform directory
cd "$(dirname "$0")"

# Check if venv exists in parent project
if [ -d "../venv" ]; then
    echo -e "${GREEN}✓ Using existing virtual environment${NC}"
    source ../venv/bin/activate
else
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
fi

# Install requirements
echo -e "${BLUE}Installing dependencies...${NC}"
pip install -q -r requirements.txt

# Create necessary directories
mkdir -p uploads results

# Start the server
echo -e "${GREEN}✓ Starting server at http://localhost:8000${NC}"
echo -e "${GREEN}✓ API docs at http://localhost:8000/docs${NC}"
echo ""

cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
