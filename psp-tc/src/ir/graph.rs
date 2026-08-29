//! Core graph types shared across IRs.

pub type TensorId = usize;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub id: TensorId,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub kind: TensorKind,
    /// TFLite quantization parameters, if the tensor is quantized.
    /// `scale.len() == 1` for per-tensor quantization; per-channel weights
    /// carry one scale per output channel (`quantized_dimension` axis).
    pub quant: Option<Quant>,
}

#[derive(Debug, Clone)]
pub struct Quant {
    pub scale: Vec<f32>,
    pub zero_point: Vec<i64>,
    pub axis: i32,
}

impl Quant {
    /// Per-tensor scale/zero-point; errors if per-channel.
    pub fn scalar(&self) -> Result<(f32, i32), String> {
        if self.scale.len() != 1 {
            return Err(format!(
                "expected per-tensor quantization, got {} scales",
                self.scale.len()
            ));
        }
        Ok((self.scale[0], *self.zero_point.first().unwrap_or(&0) as i32))
    }
}

impl Tensor {
    pub fn size_bytes(&self) -> usize {
        self.shape.iter().product::<usize>() * self.dtype.size()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    I32,
    I8,
    U8,
}

impl DType {
    pub fn size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::I32 => 4,
            DType::I8 => 1,
            DType::U8 => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TensorKind {
    /// Graph input - provided by caller
    Input,
    /// Graph output - returned to caller
    Output,
    /// Weights/biases - constant, baked into binary
    Constant { offset: usize, len: usize },
    /// Intermediate activation - scratch buffer
    Intermediate,
    /// A weight slot filled at runtime (`PspModelBuilder::external_f32`):
    /// a zeroed `.bss` static the generated module exposes by `name`, so the
    /// caller can load whichever blob it likes into it before `forward()`.
    /// Not part of the weight blob; never in the arena.
    External { name: String },
}

#[derive(Debug)]
pub struct Graph<Op> {
    pub tensors: Vec<Tensor>,
    pub ops: Vec<Op>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

impl<Op> Graph<Op> {
    pub fn tensor(&self, id: TensorId) -> &Tensor {
        &self.tensors[id]
    }

    pub fn tensor_mut(&mut self, id: TensorId) -> &mut Tensor {
        &mut self.tensors[id]
    }

    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            ops: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn add_tensor(&mut self, shape: Vec<usize>, dtype: DType, kind: TensorKind) -> TensorId {
        let id = self.tensors.len();
        self.tensors.push(Tensor {
            id,
            shape,
            dtype,
            kind,
            quant: None,
        });
        id
    }
}
