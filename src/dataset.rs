use std::slice::Iter;

/// A dataset.
#[derive(Clone, Debug)]
pub struct Dataset<T, U> {
    data: Vec<(T, U, f32)>,
}

impl<T, U> Dataset<T, U> {
    /// Creates a new dataset.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Creates a new dataset with a minimum capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Adds a rating to the dataset.
    pub fn push(&mut self, user_id: T, item_id: U, value: f32) {
        self.data.push((user_id, item_id, value));
    }

    /// Returns the number of ratings in the dataset.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T, U> Default for Dataset<T, U> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T: 'a, U: 'a> IntoIterator for &'a Dataset<T, U> {
    type Item = &'a (T, U, f32);
    type IntoIter = Iter<'a, (T, U, f32)>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::Dataset;

    #[test]
    fn test_new() {
        let mut data = Dataset::new();
        data.push(1, "A", 1.0);
    }

    #[test]
    fn test_with_capacity() {
        let mut data = Dataset::with_capacity(1);
        data.push(1, "A", 1.0);
    }

    #[test]
    fn test_into_iter() {
        let mut data = Dataset::with_capacity(1);
        data.push(1, "A", 1.0);
        assert_eq!(Some(&(1, "A", 1.0)), data.into_iter().next());
    }
}
