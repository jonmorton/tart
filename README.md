**T**raining **A**uto**R**egresive **T**ransformers

More capable than nanoGPT, just as much fun! This is my library for experimenting with transformers. I am particularly interested in exploring byte level, tokenizer-free, heirarchical autoregressive models.

Uses flash attention v2 for maximum speeed.

Included model types:

 * **gpt2**: Vanilla gpt2 architecture
 * **ibt** (improved baseline transformer): achieves lower validation loss than vanilla gpt2 with similar compute and memory requirements, by incorporating some of the latest tricks like rotary embeddings, time shifting, geglu, improved initialization, rmsnorm, and sliding window attention.
 * **hourglass** (WIP): hourglass transformers for efficient character level modeling with long context window. Still working to achieve similar perf/compute/memory as above models.
