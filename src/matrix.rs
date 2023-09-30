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
        let mut res = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut sum = 0.0;
            let row = self.row(i);
            for j in 0..self.cols {
                sum += row[j] * x[j];
            }
            res.push(sum);
        }
        res
    }
}
