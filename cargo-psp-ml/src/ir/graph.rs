//! Core graph types shared across IRs.

pub type TensorId = usize;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub id: TensorId,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub kind: TensorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    I32,
    I8,
    U8,
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
        });
        id
    }
}
