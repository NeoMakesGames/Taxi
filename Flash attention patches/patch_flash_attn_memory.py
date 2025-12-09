import os

file_path = "/home/ubuntu/.cache/huggingface/modules/transformers_modules/zhihan1996/DNABERT-2-117M/7bce263b15377fc15361f52cfab88f8b586abda0/flash_attn_triton.py"

with open(file_path, 'r') as f:
    content = f.read()

old_config = """        triton.Config({
            'BLOCK_M': 128,
            'BLOCK_N': 128
        },
                      num_warps=8,
                      num_stages=1),"""

new_config = """        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 64
        },
                      num_warps=4,
                      num_stages=1),"""

if old_config in content:
    new_content = content.replace(old_config, new_config)
    with open(file_path, 'w') as f:
        f.write(new_content)
    print("File patched successfully (memory fix).")
else:
    print("Could not find the config string to replace.")
    # Let's try a more robust replacement if exact match fails due to whitespace invisible chars
    import re
    # Regex to match the config block
    pattern = re.compile(r"triton\.Config\(\{\s*'BLOCK_M': 128,\s*'BLOCK_N': 128\s*\},\s*num_warps=8,\s*num_stages=1\),", re.DOTALL)
    
    if pattern.search(content):
        print("Found with regex, replacing...")
        new_content = pattern.sub(new_config, content)
        with open(file_path, 'w') as f:
            f.write(new_content)
        print("File patched successfully with regex.")
    else:
        print("Regex also failed to find the config.")

