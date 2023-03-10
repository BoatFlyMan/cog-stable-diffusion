sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
cog login --token-stdin <<< 3cc13fd5-c8a1-43e3-8030-205bddc74e67