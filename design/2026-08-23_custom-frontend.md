# Custom Frontend
1.6s (~30% of remaining time) is spent on the spectrogram computation. There are two major sources of inefficiency.

## Custom STFT Implementation
A Short-Time-Fourier-Transform (STFT) operates by sliding a window of fixed length `L` across the 1D signal vector, with some overlap:
```
sample : 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19
frame0 : [===============]
frame1 :             [===============]
frame2 :                         [===============]
frame3 :                                     [===============]
```
Conceptually, the existing tflite implements this by creating a dense index matrix:

$$
Index = hop*Range(0, N_Windows)^T + Range(L)
$$

Which produces a matrix of shape `[N_Windows,L]` that looks something like:
```
[
    0,  1, 2, ... L-1
    hop, hop+1, ... hop+(L-1)
    ...
    (N_Windows-1)hop ... (N_Windows-1)hop + (L-1)
]
```
The model then performs `gather(samples, Index, axis=0)`. The result is a matrix of shape `[N_Windows,L]` where each entry of `Index` is replaced by its corresponding linear index into `samples`.

In BirdNET, there are two 'branches' of STFT: `L=2048` and `L=1024`. `N_Windows = 511` for both. The formula for `hop` given a fixed `N_Windows` is:
$$
hop_L = \left \lfloor {\frac{N_{Samples}-L}{N_{Windows}-1}} \right \rfloor
$$

Thus:
$$
    hop_{2048} = \left \lfloor {\frac{144000-2048}{511-1}} \right \rfloor = 278 \\
    hop_{1024} = \left \lfloor {\frac{144000-1024}{511-1}} \right \rfloor = 280
$$

### GCD Optimization
There is a trick to reduce the amount of indices we need to represent the windows above. The observation comes from the fact that every window can be 'tiled' into chunks of size
$$
g = \text{gcd}(hop_L, L)
$$

This is because (1) each window starts at some multiple of $hop_L$ (by definition), and (2) each window is $L$ units long. Thus, instead of using an index for every single sample, we can write our indices in terms of $g$-sized chunks ($hop_L = hg$ for some $h$ and $L=lg$ for some l).

Conceptually, the model does this by 'folding the g dimension'. For example, though the input is of shape `[144000]`, for the L=1024 window we can see:

$$
    g = \text{gcd}(hop_{1024}, 1024) = \text{gcd}(280, 1024) = 8
$$

So we can instead reshape the input to `[144000/8, 8] = [18000,8]`. The Index vector can be written in terms of samples as:

$$
    Index' = \frac{hop_L}{g}*Range(0, N_Windows)^T + Range(\frac{L}{g})
$$

Which is of shape `[N_Windows, L/g]`. We can then gather along the tile axis: `gather([18000, 8], [511, 128], axis=0)`, which gives us a `[511,128,8]` tensor which we can statically reinterpret as our originally desired `[511,1024]`.

### The Fourier Transform
Now that we have a window matrix (the `[511,2048]` or `[511, 1024]` matrices), we perform an `RFFT2D` operation on it, which performs a row-wise FFT. The result is a `[511,L/2 + 1]` matrix containing _complex values_.

### Inefficiencies on PSP
Even with the GCD optimization, the intermediate tensors store duplicate data from the input samples in a large tensor (`[511,2048]` and `[511, 1024]` respectively, which is > 1.5M elements from a 144k-element input).

Instead, to reduce memory pressure, we can construct strided views into the input sample tensor:
```rust
    // let l = 2048;
    // let n_windows = 511;
    fn stft_branch(samples, n_windows, l) -> Tensor<n_windows, l> {
        let n_samples = samples.len();
        let hop = compute_hop(n_samples, n_windows, l);
        let t_view = StridedView(base=samples, stride=hop, len=l)
        let stft2d = allocate_tensor(n_windows, l);
        fft(&stft2d, t_view);
        
        stft2d
    }

    ...
    fn fft(result: &mut Tensor<n_windows, l>, t_view: StridedTensorView) {
        for let i = range(0, n_windows) {
            let window_fft = vfpu_fft(t_view.get());
            result[i,:] = window_fft;
        }
    }
```

To do this, we introduce a new `PspOp` called `StridedViewSTFT` that lowers to something like the above.

To use it, we expose a new interface to the compiler - a custom PspOp-level subgraph builder:

```rust
// device/build.rs
    pub fn buildSTFTFrontend() {
        let graph = PspModelBuilder::new();
        let samples = graph->alloc_tensor([144000]);
        let stft_1024 = PspOp::StridedViewSTFT(samples, 1024, 511);
        let stft_2048 = PspOp::StridedViewSTFT(samples, 2048, 511);
        graph->add_op(stft_1024);
        graph->add_op(stft_2048);

        psp_tc::compile(graph, "birdnet_stft");
    }
...
// device/src/main.rs
    pub fn main() {
        let samples = record_samples();
        let outputs = birdnet_stft::forward(&samples);

        assert!(len(outputs) == 2);
        assert!(outputs[0].shape() == [511, 513]);
        assert!(outputs[1].shape() == [511, 1025]);

        ...
        // generated was compiled from a tflite
        let result = generated::forward(outputs[0]);
    }
```

This will also require a way to provide a unique identifier for the generated module as above.

## Custom Mel Implementation
- the mel filterbanks are dense GEMMs: [511, 1025] @ [1025, 96] and [511, 513] @ [513, 96]
    - this is a lot of FLOPs even though the mel matrices (second matrix) is very sparse (~80%)
- We are currently passing 44.1kHz samples to a frontend that expects 48kHz - this reduces accuracy in some cases
    - Need to resample (or adjust mel filterbanks accordingly)
- Based on reversing the constants, the mel banks are:
    - HTK formula for mels: 2595.0 * log10(1.0 + f / 700.0)
    - For the 2048-len window: Fmin,Fmax = [0, 3000]
    - For the 1024-len window: Fmin,Fmax = [500, 15000]

### Mel Filterbank
The function make_mel returns a mel filterbank matrix and takes a few parameters

n_banks
fmin
fmax
n_freqs
sampling_rate (=44.1kHz for PSP)

the result is a tensor of shape [n_freqs, n_banks] stored in Compressed Sparse Column format (see below for details)

conceptually, the matrix divides the range fmin,fmax into sections of equal mel spaces. that is, each resulting bank represents

deltaM = {mel(fmax)-mel(fmin)}/(n_banks+1)

mels, where mel(f) is computed w the htk formula 2595log(1+f/700).

in principle, each bank i needs to select frequencies imel(deltaM\*i)+1 to imel(deltaM\*(i+1))

where imel(m) is the inverse of mel(f). we could do this by simply placing 1s in the rows corresponding to those frequency ranges. The drawbacks are that this doesnt do any smoothing and allows for no overlap (todo why does this matter). instead we use a *triangle* function for each bank.

let:
m_min = mel(fmin)
m_i = m_min + i * deltaM
f_i = imel(m_i)

then each triangle is centered at m_i and decreases linearly to 0 at m_i-1 and m_(i+1) (i in [1, B] where B is the number of banks to use). applying the imel to these 3 points gives us the frequencies of interest.

since the sampling rate is given, the frequency of bin k is given by:

`freq_k = k*sampling_rate/(2*(n_freqs-1))`
thus 
`k = (2*(n_freqs-1))*freq / sampling_rate`

(note that n_fft = 2*(n_freqs-1)) gives the closest index to a given frequency.

Thus the algorithm works as follows.
let:
`L_fft`: the length of the FFT window used. It is related to `n_freqs` by `n_freqs = L_fft/2 + 1`
`F_s`: the sampling rate. 48kHz for BirdNET
`f_min, f_max`: the min and max frequencies that the filterbanks should represent
`B`: the number of mel filterbanks

we define the following functions:
`mel(f) = 2595log(1+f/700)`: convert a frequency to a mel value
`f(k) = k * F_s/L_fft`: the frequency represented by bin `k`
`m(k) = mel(f(k))`: the mel represented by bin `k`
`k(m) = floor(if(imel(m)))`: the maximum bin `k` whose mel is less than or equal to `m` (where if is inverse f(k) and imel is inverse mel(f))

we begin by computing
`Δm = (mel(f_max) - mel(f_min)) / B`
`m_bank[i] = mel(f_min) + iΔm` for `i ∈ [0, B+1]`

then we compute the `B` banks. for `b ∈ [1, B]`, we compute the endpoints:
`kl_b = k(m_bank[b-1])`
`kr_b = k(m_bank[b+1])+1`
the filter function is then computed over the range `k ∈ [kl_b, kr_b]`:
`filter_b[k] = 1 - abs((m(k) - m_bank[b])/(m_bank[b] - m_bank[b-1]))`

Using this, create a ColumnBand datastructure containing:
start:kl_b
len: kr_b - kl_b + 1
data: filter_b[k - kl_b]

Collect all these into a new ColumnBandMatrix object that contains:
num_columns
Band[]

(note that for now, the assumption is that all columns have a CSC structure)

this looks like:
```rust
struct ColumnBand {
  start_idx: uint64
  len: uint64
  buf: [float32; len]
}

struct CBMatrix {
 columns : [ColumnBand]
 n_cols : uint64
}

...

fn make_mel(n_freqs, n_banks, fmin, fmax, sampling_rate) -> CBMatrix<n_freqs, n_banks> {
}
```

### FullyConnected CB PspOp
We extend FullyConnected to accept a CB variant. when this variant is selected, the constant tensor data is stored as a CBMatrix.

### Mel Spectrogram Subgraph Builder
Using the new subgraph builder interface, we design a function:
```rust
fn mel_spectrogram(
  input_fft: Tensor<windows, n_freqs>, 
  sampling_rate,
  fmin,fmax,
  nbanks,
  pow){
    let mel_mat = make_mel(n_freqs, nbanks, fmin, fbank, sampling_rate)
   let matmul = PspOp::FC::new_csc(input_fft, mel_mat)
   
   // note that we are receiving the real part of the spectrum; so we need to square before the fractional pow.
   // we eagerly fuse this into a single 2+pow op.
    let pow = PspOp::ElementWise::new_pow(matmul->result(), 2 + pow)

   graph->add_op(matmul)
   graph->add_op(pow)
}

...

```

### FullyConnected CB Kernel
we design a CB kernel that is vfpu accelerated.

suppose we are doing M,N x N,K where the CB matrix is on the rhs. each column of the CB "slices" the input matrix to a [M,C] subset where C is the num of nonzeros of the current column. Thus we are doing a `[MxC] x [Cx1]` dense matmul.

As a first baseline, we implement the kernel as `K` dense mat-vec multiplications.

### Putting it All Together

now we can design a custom mel frontend benchmark. similar to the FFT one, we construct the mel subgraph in build.rs, give it a name, and invoke the generated forward function in the user code. compare this against the original tflites results for the same subgraph and benchmark speed, memory watermark, and blob footprint.

