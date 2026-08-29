# PSPBird App
the existing birdnet example is simply a benchmark that compares against golden.
Now that we have a baseline of under 5 seconds and a memory footprint of <25MB, we can design an actual birding app for the PSP.

## Integration with Birdnet Example
Since the birdnet example contains a lot of code that is evolving, we implement the pspbird app as a frontend wrapper (alternative binary)
for the `birdnet/device` module.

To this end, we refactor the example into a `benchmark` and `app` binaries that call
into a common set of functions in the `psp_bird` library exported within the same crate.
```
    src
        bin
            app/
            benchmark/
        lib/
            psp_bird/
```

The main function that both call into shall be called:
```rust
    psp_bird::classify_birds(&audio, &out);
```

## Multi-Model Support
The pruning mechanism (TODO link the pruning design) enables higher accuracy and lower memory footprint when a geographic region is selected. The PSP is not capable of JIT-compiling a pruned model, so we need to do this offline.

However, experimentally, every generated.rs compiles to ~32MB (weights are `include_bytes!` into rodata, and arenas are wide allocated buffers, plus codesize).

To this end, we snip out the classifier block from the birdnet tflite. Instead, we use the subgraph builder `PspModelBuilder` to construct a custom FC layer of shape [TOPK, 1024] called `ExternalWeightFC`. this subgraph exposes a generated API for filling the weight data at runtime. Then, in the main app loop, we store a map of region to a blob at a particular ms0 path, e.g. `eastern-na -> ms0:/PSPBIRD/eastern_na_fc_.bin`. The user is presented with the regions present in prune_classifier.py.

they can use the up/down arrows to select which one to use, at which point pspbird will call:
```rust
    classifier_subgraph->load_weights(&bin_lookup[selected_bbox])
```
this will internally load the data from ms0 into RAM such that later, when the `psp_bird::classify_birds` api is called, the classifier weights are already in memory:
```rust
fn classify_birds()
    custom_stft_frontend::forward()
    generated::forward()
    classifier::forward() // reads weights from the in-memory blob
```

## Audio Recording
The overall app looks like:
- welcome menu: select from the pruned classifiers
- recording mode: single-step or live recording.

in single-step mode we let the user play back the recording before running inference manually. (press X to start recording, X again to stop).

in live recording mode, we will perform the inference work on a background thread. This thread is an infinite loop which waits 3 seconds, then reads from a common ring buffer of audio data (with an index based on the last start). scores are then published to the main thread as a dense vector of floats; for now, we will always update the scores as the new state comes back. in the future we can implement smoothing.

the infrastructure is shared - there is one point in the code where a dense vector is received and the results are displayed ot the screen. this is to eventually support image mode, where a picture of the top 5 birds is shown in the app.

## Image loading
we will use the inaturalist API to fetch bird images for the ~1131 unique species across bounding boxes in the pruned_classifier. We will write a script to fetch the images - only the script should get committed.