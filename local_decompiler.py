import os
import json
import subprocess
import urllib.request
import urllib.parse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO, 'smm2-decomp', 'data', 'v3.0.3', 'functions.csv')

def get_assembly(obj_file, func_name):
    cmd = ["llvm-objdump", "-d", obj_file]
    try:
        out = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.DEVNULL)
    except Exception as e:
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
            parts = line.split('\t')
            if len(parts) >= 3:
                asm.append(("\t".join(parts[2:])).strip())
            elif len(parts) >= 2:
                asm.append(parts[1].strip())
    if asm:
        return "\n".join(asm)
    return None

def call_ollama(prompt):
    url = "http://127.0.0.1:11434/api/generate"
    data = {
        "model": "gemma2:27b",
        "prompt": prompt,
        "stream": False,
        "system": "You are an expert C++ AArch64 decompilation AI. Only return the C++ code block, do not explain.",
        "options": {"temperature": 0.1}
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode('utf-8'))
            return res.get("response", "")
    except Exception as e:
        return f"Error connecting to Ollama: {e}"

def main():
    print("Starting Local Gemma 2 Decompiler...")
    # Find the first W function in the CSV
    target_func = None
    with open(CSV_PATH, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4 and parts[1] == 'W':
                target_func = parts[3]
                break
                
    if not target_func:
        print("No 'W' functions found in CSV!")
        return

    print(f"🎯 Target Acquired: {target_func}")
    # Run permuter or a basic dump to get the target object
    print("Extracting assembly...")
    
    # We use a dummy hardcoded path for now just to prove it works
    # A real implementation would parse the SRC tree to find where it belongs
    asm_code = "add x0, x1, x2\nret" # Placeholder for assembly
    
    prompt = f"Decompile this AArch64 assembly into clean C++ for the SMM2 project:\n```asm\n{asm_code}\n```"
    
    print(f"🧠 Asking Gemma2:27b (via Ollama API) to decompile...")
    cpp_output = call_ollama(prompt)
    
    # Extract code from Markdown blocks
    if "```cpp" in cpp_output:
        cpp_output = cpp_output.split("```cpp")[1].split("```")[0].strip()
    elif "```c" in cpp_output:
        cpp_output = cpp_output.split("```c")[1].split("```")[0].strip()
    elif "```" in cpp_output:
        cpp_output = cpp_output.split("```")[1].strip()
        
    out_file = os.path.join(REPO, 'smm2-decomp', 'scratch', f"{target_func}_gemma.cpp")
    with open(out_file, 'w') as f:
        f.write(cpp_output)
        
    print(f"✅ Generated C++ saved to: {out_file}")
    print("Review the code and steer the model for future runs!")

if __name__ == '__main__':
    main()
