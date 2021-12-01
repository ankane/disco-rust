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
        Self {
            data: Vec::new()
        }
    }

    pub fn push(&mut self, user_id: T, item_id: U, value: f32) {
        self.data.push(Rating { user_id, item_id, value });
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
