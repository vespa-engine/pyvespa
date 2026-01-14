---
title: Nanobeir Hybrid Evaluation
emoji: ⚡
colorFrom: green
colorTo: purple
sdk: static
pinned: false
header: mini
license: apache-2.0
short_description: Overview of a selection of embedding model across dimensions
---

Please see https://github.com/vespa-engine/pyvespa/blob/master/vespa/evaluation/_mteb.py for example that can be adapted to run your own benchmarks on selected model(s).

## Deploying to Hugging Face Spaces

To deploy only the required files (`index.html`, `NanoBEIR/models.js`, and `README.md`) to the Hugging Face Space, make sure you are in the same directory as this README, then run:

```bash
hf upload vespa-engine/nanobeir-hybrid-evaluation index.html index.html --repo-type space 
hf upload vespa-engine/nanobeir-hybrid-evaluation NanoBEIR/models.js NanoBEIR/models.js --repo-type space
hf upload vespa-engine/nanobeir-hybrid-evaluation README.md README.md --repo-type space
```

This ensures no other files are deployed to the space.