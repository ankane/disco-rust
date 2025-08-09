use std::slice::Iter;

#[derive(Clone, Debug)]
pub(crate) struct Rating<T, U> {
    pub user_id: T,
    pub item_id: U,
    pub value: f32,
}

/// A dataset.
#[derive(Clone, Debug)]
pub struct Dataset<T, U> {
    data: Vec<Rating<T, U>>,
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
        self.data.push(Rating {
            user_id,
            item_id,
            value,
        });
    }

    pub(crate) fn iter(&self) -> Iter<'_, Rating<T, U>> {
        self.data.iter()
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
}
