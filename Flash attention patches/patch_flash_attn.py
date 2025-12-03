import os

file_path = "/home/ubuntu/.cache/huggingface/modules/transformers_modules/zhihan1996/DNABERT-2-117M/7bce263b15377fc15361f52cfab88f8b586abda0/flash_attn_triton.py"

with open(file_path, 'r') as f:
    content = f.read()

new_content = content.replace("tl.dot(q, k, trans_b=True)", "tl.dot(q, tl.trans(k))")

if content == new_content:
    print("No changes made. String not found?")
else:
    with open(file_path, 'w') as f:
        f.write(new_content)
    print("File patched successfully.")
