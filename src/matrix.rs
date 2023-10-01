#[derive(Clone, Debug)]
pub struct Matrix {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = vec![0.0; rows * cols];
        Self { rows, cols, data }
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        let idx = i * self.cols;
        &mut self.data[idx..(idx + self.cols)]
    }

    pub fn row(&self, i: usize) -> &[f32] {
        let idx = i * self.cols;
        &self.data[idx..(idx + self.cols)]
    }

    pub fn dot(&self, x: &[f32]) -> Vec<f32> {
        self.data
            .chunks_exact(self.cols)
            .map(|row| row.iter().zip(x).map(|(ri, xi)| ri * xi).sum())
            .collect()
    }
}
