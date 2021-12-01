use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hash;

pub struct Map<T> {
    map: HashMap<T, usize>,
    vec: Vec<T>,
}

impl<T: Clone + Eq + Hash> Map<T> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            vec: Vec::new(),
        }
    }

    pub fn add(&mut self, id: T) -> usize {
        let i;
        match self.map.entry(id) {
            Entry::Occupied(o) => {
                i = *o.get();
            }
            Entry::Vacant(v) => {
                i = self.vec.len();
                let key = v.key().clone();
                v.insert(i);
                self.vec.push(key);
            }
        };
        i
    }

    pub fn get(&self, id: &T) -> Option<&usize> {
        self.map.get(id)
    }

    pub fn lookup(&self, index: usize) -> &T {
        &self.vec[index]
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn ids(&self) -> &Vec<T> {
        &self.vec
    }
}
