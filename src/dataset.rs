use std::slice::Iter;

pub(crate) struct Rating<T, U> {
    pub user_id: T,
    pub item_id: U,
    pub value: f32,
}

pub struct Dataset<T, U> {
    data: Vec<Rating<T, U>>,
}

impl<T, U> Dataset<T, U> {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, user_id: T, item_id: U, value: f32) {
        self.data.push(Rating {
            user_id,
            item_id,
            value,
        });
    }

    pub(crate) fn iter(&self) -> Iter<Rating<T, U>> {
        self.data.iter()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

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
