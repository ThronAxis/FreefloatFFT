# FreefloatFFT Documentation

This directory contains the project documentation as specified in the
Design Document (section 6).

## Documents

- **[PRD.md](../FreefloatFFT_PRD.md)** -- Product Requirements Document
  - Executive summary, problem statement, scope, architecture
  - Technical specifications, performance requirements
  - API contract, testing strategy, milestones
  - Risk register, antigravity application context

- **[DESIGN.md](../FreefloatFFT_DESIGN.md)** -- Technical Design & Implementation Reference
  - Mathematical foundation (DFT, Cooley-Tukey, bit-reversal)
  - CUDA execution model deep-dive
  - Twiddle factor implementation
  - Butterfly network (full implementation)
  - Shared memory bank conflict analysis & fix
  - Complete kernel source layout
  - Kaggle integration walkthrough
  - Profiling guide (nvprof, nsight)
  - Benchmark results & analysis
  - Antigravity signal processing integration
  - Advanced optimizations roadmap

## Quick Links

| Topic | Document | Section |
|-------|----------|---------|
| API Reference | PRD | Section 10 |
| Performance Targets | PRD | Section 6 |
| Bank Conflict Proof | DESIGN | Section 5 |
| Kernel Architecture | DESIGN | Section 4 |
| Kaggle Setup | DESIGN | Section 7 |
| Profiling Guide | DESIGN | Section 8 |
