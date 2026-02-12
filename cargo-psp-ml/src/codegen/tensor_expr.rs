use crate::ir::graph::{Graph, TensorId, TensorKind};
use crate::ir::psp::PspOp;
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;

pub struct TensorExprWriter<'a> {
    pub graph: &'a Graph<PspOp>,
    input_id: TensorId,
}

/**
 * Helper struct for generating tensor expressions for a given graph.
 */
impl<'a> TensorExprWriter<'a> {
    pub fn new(graph: &'a Graph<PspOp>) -> Self {
        let input_id = graph.inputs[0];
        Self { graph, input_id }
    }

    /// Variable identifier for a tensor: `t_{id}`.
    pub fn ident(&self, id: TensorId) -> Ident {
        Ident::new(&format!("t_{id}"), Span::call_site())
    }

    /// Read-reference expression for a tensor in a kernel call.
    ///
    /// - Input tensor → `input` (the function parameter)
    /// - Constant tensor → `t_{id}` (already a `&[f32]` slice)
    /// - Intermediate → `t_{id}` (already `&mut [f32]` from aligned static; auto-reborrows)
    /// - Output → `&t_{id}` (borrow local array)
    pub fn read(&self, id: TensorId) -> TokenStream {
        if id == self.input_id {
            return quote!(input);
        }
        let ident = self.ident(id);
        match &self.graph.tensor(id).kind {
            TensorKind::Constant { .. } | TensorKind::Intermediate => quote!(#ident),
            _ => quote!(&#ident),
        }
    }

    /// Write-reference expression for a tensor in a kernel call.
    ///
    /// - Intermediate → `t_{id}` (already `&mut [f32]` from aligned static)
    /// - Output → `&mut t_{id}` (borrow local array)
    pub fn write(&self, id: TensorId) -> TokenStream {
        let ident = self.ident(id);
        match &self.graph.tensor(id).kind {
            TensorKind::Intermediate => quote!(#ident),
            _ => quote!(&mut #ident),
        }
    }

    /// Static buffer identifier for an intermediate tensor: `T_{id}_BUF`.
    pub fn buf_static(&self, id: TensorId) -> Ident {
        Ident::new(&format!("T_{id}_BUF"), Span::call_site())
    }

    /// Offset constant identifier for a weight tensor: `T_{id}_OFFSET`.
    pub fn offset_const(&self, id: TensorId) -> Ident {
        Ident::new(&format!("T_{id}_OFFSET"), Span::call_site())
    }

    /// Length constant identifier for a weight tensor: `T_{id}_LEN`.
    pub fn len_const(&self, id: TensorId) -> Ident {
        Ident::new(&format!("T_{id}_LEN"), Span::call_site())
    }

    /// Shape literal for a tensor: `[dim0, dim1, ...]`.
    pub fn shape(&self, id: TensorId) -> TokenStream {
        let dims = &self.graph.tensor(id).shape;
        quote!([#(#dims),*])
    }
}
