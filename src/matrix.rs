use std::slice::{ChunksExact, ChunksExactMut, Iter};

/// A dense matrix.
pub struct DenseMatrix {
    pub(crate) cols: usize,
    data: Vec<f32>,
}

impl DenseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = vec![0.0; rows * cols];
        Self { cols, data }
    }

    pub fn rows(&self) -> ChunksExact<'_, f32> {
        self.data.chunks_exact(self.cols)
    }

    pub fn rows_mut(&mut self) -> ChunksExactMut<'_, f32> {
        self.data.chunks_exact_mut(self.cols)
    }

    pub fn row(&self, i: usize) -> &[f32] {
        let idx = i * self.cols;
        &self.data[idx..(idx + self.cols)]
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        let idx = i * self.cols;
        &mut self.data[idx..(idx + self.cols)]
    }

    pub fn dot(&self, x: &[f32]) -> Vec<f32> {
        self.data
            .chunks_exact(self.cols)
            .map(|row| row.iter().zip(x).map(|(ri, xi)| ri * xi).sum())
            .collect()
    }
}

/// A coordinate list (COO) matrix.
pub struct CooMatrix {
    // separate vectors to avoid padding
    row_inds: Vec<usize>,
    col_inds: Vec<usize>,
    values: Vec<f32>,
}

impl CooMatrix {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            row_inds: Vec::with_capacity(capacity),
            col_inds: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, row_index: usize, col_index: usize, value: f32) {
        self.row_inds.push(row_index);
        self.col_inds.push(col_index);
        self.values.push(value);
    }

    pub fn len(&self) -> usize {
        self.row_inds.len()
    }

    pub fn get(&self, i: usize) -> (usize, usize, f32) {
        (self.row_inds[i], self.col_inds[i], self.values[i])
    }
}

/// A list of lists (LIL) matrix.
pub struct LilMatrix {
    data: Vec<Vec<(usize, f32)>>,
}

impl LilMatrix {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn push(&mut self, row_index: usize, col_index: usize, value: f32) {
        if row_index == self.data.len() {
            self.data.push(Vec::new());
        }
        self.data[row_index].push((col_index, value));
    }
}

impl<'a> IntoIterator for &'a LilMatrix {
    type Item = &'a Vec<(usize, f32)>;
    type IntoIter = Iter<'a, Vec<(usize, f32)>>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}
