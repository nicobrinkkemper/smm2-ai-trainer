import os
import glob
import json
import re
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO, 'data', 'v3.0.3', 'functions.csv')

# Load all O functions
matched_funcs = []
with open(CSV_PATH, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) >= 4 and parts[1] == 'O':
            matched_funcs.append({
                'addr': parts[0],
                'size': parts[2],
                'name': parts[3]
            })

def find_func_in_cpp(func_name):
    # Only search src/game to get juicy custom logic
    for root, _, files in os.walk(os.path.join(REPO, 'src', 'game')):
        for file in files:
            if not file.endswith('.cpp'): continue
            path = os.path.join(root, file)
            with open(path, 'r', errors='ignore') as f:
                content = f.read()
                # Basic string match to speed up
                if func_name in content:
                    return path, content
    return None, None

def extract_function(source: str, func_name: str) -> str:
    lines = source.splitlines()
    result = []
    in_func = False
    brace_depth = 0
    
    # We look for the function name followed by (
    for i, line in enumerate(lines):
        if not in_func:
            if func_name in line and '(' in line and not line.strip().startswith('//'):
                in_func = True
                brace_depth = line.count('{') - line.count('}')
                result.append(line)
                if '{' not in line:
                    # check next line
                    for j in range(i+1, min(i+5, len(lines))):
                        if '{' in lines[j]:
                            brace_depth += lines[j].count('{') - lines[j].count('}')
                            break
        else:
            result.append(line)
            brace_depth += line.count('{') - line.count('}')
            if brace_depth <= 0:
                break
    
    if in_func and len(result) > 0:
        return "\n".join(result)
    return None

def get_assembly(obj_file, func_name):
    cmd = ["llvm-objdump", "-d", obj_file]
    try:
        out = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.DEVNULL)
    except:
        return None
        
    lines = out.split('\n')
    in_func = False
    asm = []
    for line in lines:
        if f"<{func_name}>:" in line:
            in_func = True
            continue
        if in_func:
            if not line.strip() or line.startswith("00"):
                break
            # Remove the raw hex bytes to just leave the assembly instructions
            parts = line.split('\t')
            if len(parts) >= 3:
                asm.append(("\t".join(parts[2:])).strip())
            elif len(parts) >= 2:
                asm.append(parts[1].strip())
    if asm:
        return "\n".join(asm)
    return None

def main():
    out_file = os.path.join(REPO, 'scratch', 'gemma_finetune_dataset.jsonl')
    count = 0
    with open(out_file, 'w') as out:
        for func in matched_funcs:
            # We want to keep the dataset size manageable, maybe 500 perfect examples to start
            if count >= 500:
                break
                
            path, source = find_func_in_cpp(func['name'])
            if not path: continue
            
            cpp_code = extract_function(source, func['name'])
            if not cpp_code or len(cpp_code.splitlines()) < 5:
                continue
                
            # Build object path from cpp path
            rel_path = os.path.relpath(path, REPO)
            obj_path = os.path.join(REPO, 'build', 'CMakeFiles', 'Slope.dir', f"{rel_path}.obj")
            
            if not os.path.exists(obj_path):
                continue
                
            asm_code = get_assembly(obj_path, func['name'])
            if not asm_code:
                continue
                
            # Create standard chat template for fine-tuning
            item = {
                "messages": [
                    {"role": "system", "content": "You are an expert C++ decompilation assistant. Translate the following AArch64 assembly into clean Clang 8 C++ code that perfectly byte-matches."},
                    {"role": "user", "content": f"AArch64 Assembly:\n```asm\n{asm_code}\n```\n\nDecompile this into C++."},
                    {"role": "assistant", "content": f"```cpp\n{cpp_code}\n```"}
                ]
            }
            
            out.write(json.dumps(item) + '\n')
            count += 1
            if count % 50 == 0:
                print(f"Extracted {count} functions...")

    print(f"Done! Extracted {count} functions to {out_file}")

if __name__ == '__main__':
    main()
