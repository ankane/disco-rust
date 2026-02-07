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

    #[cfg(not(feature = "simd"))]
    pub fn dot(&self, x: &[f32]) -> Vec<f32> {
        self.data
            .chunks_exact(self.cols)
            .map(|row| row.iter().zip(x).map(|(ri, xi)| ri * xi).sum())
            .collect()
    }

    #[cfg(feature = "simd")]
    pub fn dot(&self, x: &[f32]) -> Vec<f32> {
        use std::simd::num::SimdFloat;
        use std::simd::{Simd, StdFloat};

        const LANES: usize = 4;

        let (x_chunks, x_remainder) = x.as_chunks::<LANES>();

        self.data
            .chunks_exact(self.cols)
            .map(|row| {
                let (row_chunks, row_remainder) = row.as_chunks::<LANES>();
                row_chunks
                    .iter()
                    .zip(x_chunks)
                    .fold(Simd::<f32, LANES>::splat(0.0), |sum, (ri, xi)| {
                        Simd::from_slice(ri).mul_add(Simd::from_slice(xi), sum)
                    })
                    .reduce_sum()
                    + row_remainder
                        .iter()
                        .zip(x_remainder)
                        .map(|(ri, xi)| ri * xi)
                        .sum::<f32>()
            })
            .collect()
    }
}

#[cfg(feature = "bench")]
mod tests {
    use super::Matrix;

    #[bench]
    fn bench_dot(b: &mut test::Bencher) {
        let d = 128;
        let m = Matrix::new(100000, d);
        let x = vec![1.0; d];
        b.iter(|| m.dot(&x));
    }
}
