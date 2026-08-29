https://claude.ai/code/artifact/f8923c09-9d09-49ca-abb8-3f2fd6063a30

according to this, we are throwing away >80% of the 2048-point FFT we performed. thats a LOT of wasted work, since this FFT is performed for all 511 frames! similar for the 1024-point fft (38%)

TODO compute exact wasted FLOPs,