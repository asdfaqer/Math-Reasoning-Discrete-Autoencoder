# Math-Reasoning-Discrete-Autoencoder
LLMs are very verbose and dataset for fine-tuning LLMs for math can take up to terabytes of storage. With this lightweight discrete autoencoder for text, it enable lossless compresion through arthmetic coding better then the standard gzip by a factor of 2. 

# Performance


# How it works
Raw text is feed through a BPE tokenizer from Roberta Base and passed through one encoder/decoder transformer and then a decoder only transformer. The first encoders the text into 64 codebook tokens, an indexer picks the top M (arbitrary) tokens to pass to the second transformer. The second takes in the M tokens as input and outputs what it thinks is the raw text encoded. 

The training is done in three phases. In phase 1, a non discrete base is trained. In phase 2, we initialize codebook through k-means clustering then train only the half after the codebook (this phase prevents codebook collapse until the decoder learns enough to understand encoded tokens). In phase 3, we finally train jointly with both halves.

# How to Run


