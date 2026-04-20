# SMM2 Auto-Decompilation Program

This is the system prompt and execution ruleset for the autonomous C++ decompilation loop of Super Mario Maker 2.

## The Philosophy: Readability Over Perfection
Our ultimate goal is to produce clean, human-readable, logically accurate C++ source code that documents how the Nintendo Switch SMM2 physics engine operates.

**DO NOT sacrifice readability for a 100% byte match.**
If achieving an `OK` byte match requires you to spam `*(volatile uint32_t*)` casts, write inline assembly blocks, or fundamentally contort the logical flow of the code, **do not do it**. 

## The Core Loop

1. **Target Selection:** Pick the next target function from `data/v3.0.3/functions.csv` that is currently marked as `W` (WIP) or `A` (Auto-matched).
2. **Disassembly:** Run `llvm-objdump -d` on the corresponding compiled `.obj` file in `build/` to see the target AArch64 assembly.
3. **Drafting:** Write a clean, human-readable C++ stub matching the signature in the target `.cpp` file in `src/`.
4. **Compilation & Checking:** Run the project's build and verification script:
   ```bash
   ninja -C build && tools/check sub_XXXXXXX
   ```
5. **Evaluation & Escalation:**
   - If it says `OK`: You won! Commit the result to the repository with a descriptive commit message (`git commit -am "decomp: match sub_XXXXXXX"`) and move to the next target.
   - If it outputs a mismatch diff: Analyze the AArch64 diff provided by Viking.
   - If the mismatch score goes **down** from your previous attempt without compromising code readability: Keep the changes.
   - **The "Step Back" Rule (Fuzz Testing):** If the mismatch score gets stuck due to instruction scheduling (e.g., swapping of independent register loads) or register allocation differences (`x8` vs `x9`), do not endlessly push through the mountain.
     - **Stop brute-forcing.** 
     - Do NOT use volatile hacks or inline assembly.
     - Instead, write a property-based Unicorn fuzz test in `tests/bridge/test_sub_XXXXXXX.py` comparing your compiled object against the original binary.
     - If the memory states and register outputs are functionally identical across 10,000 iterations, the function is perfectly accurate. Mark it as `E` (Equivalent) in the CSV, commit it, and move on.
   - If a pattern of mismatch is recurring due to a specific compiler behavior (like NEON merging), step back, pause the loop, and suggest a global compiler patch or macro to the human orchestrator.

## The Rules
- **Never guess blindly:** Always base your next C++ edit on the explicit output of the `tools/check` diff or the Unicorn fuzzing results.
- **Understand the Compiler:** We use a patched Clang 8.0.0 (`-mllvm -enable-post-misched=false -mllvm -disable-store-merging -fno-slp-vectorize`). It is heavily optimized, but it inlines aggressively. If a function is massive, look for inline helper functions in the SMM1 decompilation references (`ref/smr/`).
- **Structs over Primitives:** Nintendo extensively uses bitfields and structs. If an instruction uses `ldr w8, [x19, #0x492]`, you are likely dealing with a struct field, not raw pointer math. Search the `src/` headers for matching offsets.
- **Ask for Help:** If you encounter a fundamental failure in the build system or you get totally stuck, do not spin in an endless loop apologizing. Pause the loop immediately and ask the human operator for help.
