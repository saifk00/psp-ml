
# Overview
docs/vme-reference.html contains a speculative Programmers Reference Manual for the PSP's Virtual Mobile Engine - a 4-PE CGRA accessible from the secondary CPU. Based on this, we create a new directory vme-emu that contains Verilog RTL for the most likely implementation of the PEs, along with the AGUs and top/base buffers; the result should be that initializing the corresponding config space (102 words) causes a simulation of this RTL to behave closely to the real VME

## RTL Simulator
  (1) a C/C++ (your choice) host driver that wraps a behavioral simulation of the VME using this RTL. the program vme-emu <image> accepts a 1MB file ('machine image') that corresponds to the memory map of the VME (0x4400_0000–0x440F_FFFF). It performs the same steps on the RTL as the initialization library (enables the clock signals, does the context handshake, then asserts the trigger - note that it doesnt need to use the LOAD DMA operation because buffers are initialized by whoever created the image file) runs the simulation until the DMA_STAT VD bit is set (the completion signal)

  (2) a Rust crate in psp-rt called vme-assembler
  The crate provides two things:
  (a) a user-friendly API for configuring the VME:
```rust
      val vme : VMEConfig = VMEConfig::new();
      val pe0 : VMEProcessingElement = vme->pe0
      pe0.set_front(vme::TOP0);
      pe0.set_back(pe1.scratch);
      val operation = VMEOperation::new(OpCodes::DotProduct, pe0.front, pe0.back);
      vme.top[0].set_callback(&construct_top0_buf);
```

  (b) an assembler that accepts a VMEConfig and constructs the 1MB machine image that gets passed to vme-emu:
  - invokes callbacks for initializing the buffers
  - constructs the PE config words in a clean way
  - does things like figure out the correct cycle offsets (based on the RTL - the philosophy is that we can tweak these later as we update the RTL)

## Conformance Tests
(1) a rust binary tool in psp-tc (cargo run -p psp-tc --binary vme-dump <image>) that prints out the top / base buffers as hex words in an easy to understand format

(2) A new sample that demonstrates how host code can use the same VME object to run on the real VME and against the simulator. the device code runs in an infinite loop - accept a MachineImage over stdin (or a exit-causing sentinel value), send it to the VME, write all the buffers back over stdout TOP0..3BASE0..3. The host code uses send_shell_command(ld) to start this program on the device, then runs a number of tests:
  - each test is a function that initializes a VME object, then calls a function to send it over stdin and receive a VMEResult object (unpacked buffers). It then also calls a function to _simulate_ the result in RTL (to this end, we need a new rust crate that wraps vme-emu - vme-emu-sys and provide a simple vme_emu(MachineImage) -> VMEResult) and receives another VMEResult object. It then compares the two to report its result

With the above, we can write a host program in rust that enumerates test cases that have a common interface, and launch them on the emulator and the actual hardware to compare results. Using this we can progressively refine our RTL.