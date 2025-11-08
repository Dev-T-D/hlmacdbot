#!/bin/bash
# Setup script for Hyperliquid credentials

echo "🔐 Setting up Hyperliquid credentials..."
echo ""

# Check if .env exists
if [ -f .env ]; then
    echo "⚠️  .env file already exists!"
    read -p "Overwrite? (y/N): " overwrite
    if [ "$overwrite" != "y" ]; then
        echo "Cancelled."
        exit 0
    fi
fi

# Create .env file
cat > .env << 'ENVEOF'
# Hyperliquid Credentials
# NEVER commit this file to git!

HYPERLIQUID_PRIVATE_KEY=0x0a4f7d6f4f17aac74bcb3f50eaa1528e97678b48c7c380bd45d7e484b1ae3028
HYPERLIQUID_WALLET_ADDRESS=0xd36cd831e00cf42c505a9d705475f74fc8c492ad
HYPERLIQUID_TESTNET=true
ENVEOF

echo "✅ Created .env file with your credentials"
echo ""
echo "📝 To use these credentials, run:"
echo "   source .env"
echo "   # or"
echo "   export HYPERLIQUID_PRIVATE_KEY=0x..."
echo "   export HYPERLIQUID_WALLET_ADDRESS=0x..."
echo ""
echo "🔒 Security:"
echo "   - .env is in .gitignore (won't be committed)"
echo "   - Credentials are loaded from environment variables"
echo "   - For production, use system keyring: pip install keyring"
