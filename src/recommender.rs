use crate::map::Map;
use crate::matrix::Matrix;
use crate::prng::Prng;
use crate::Dataset;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// Information about a training iteration.
#[derive(Clone, Debug)]
pub struct FitInfo {
    /// The iteration.
    pub iteration: u32,
    /// The training loss.
    pub train_loss: f32,
    /// The validation loss.
    pub valid_loss: f32,
}

/// A recommender builder.
pub struct RecommenderBuilder<'a> {
    factors: u32,
    iterations: u32,
    regularization: Option<f32>,
    learning_rate: f32,
    alpha: f32,
    callback: Option<Box<dyn 'a + Fn(FitInfo)>>,
    seed: Option<u64>,
}

impl<'a> RecommenderBuilder<'a> {
    /// Starts a new recommender.
    pub fn new() -> Self {
        Self {
            factors: 8,
            iterations: 20,
            // there is regularization by default
            // but explicit and implicit have different defaults
            regularization: None,
            learning_rate: 0.1,
            alpha: 40.0,
            callback: None,
            seed: None,
        }
    }

    /// Sets the number of factors.
    pub fn factors(&mut self, value: u32) -> &mut Self {
        self.factors = value;
        self
    }

    /// Sets the number of iterations.
    pub fn iterations(&mut self, value: u32) -> &mut Self {
        self.iterations = value;
        self
    }

    /// Sets the regularization.
    pub fn regularization(&mut self, value: f32) -> &mut Self {
        self.regularization = Some(value);
        self
    }

    /// Sets the learning rate.
    pub fn learning_rate(&mut self, value: f32) -> &mut Self {
        self.learning_rate = value;
        self
    }

    /// Sets alpha.
    pub fn alpha(&mut self, value: f32) -> &mut Self {
        self.alpha = value;
        self
    }

    /// Sets the callback for each iteration.
    pub fn callback<C: 'a + Fn(FitInfo)>(&mut self, value: C) -> &mut Self {
        self.callback = Some(Box::new(value));
        self
    }

    /// Sets the random seed.
    pub fn seed(&mut self, value: u64) -> &mut Self {
        self.seed = Some(value);
        self
    }

    /// Creates a recommender with explicit feedback.
    pub fn fit_explicit<T: Clone + Eq + Hash, U: Clone + Eq + Hash>(
        &self,
        train_set: &Dataset<T, U>,
    ) -> Recommender<T, U> {
        self.fit(train_set, None, false)
    }

    /// Creates a recommender with implicit feedback.
    pub fn fit_implicit<T: Clone + Eq + Hash, U: Clone + Eq + Hash>(
        &self,
        train_set: &Dataset<T, U>,
    ) -> Recommender<T, U> {
        self.fit(train_set, None, true)
    }

    /// Creates a recommender with explicit feedback and performs cross-validation.
    pub fn fit_eval_explicit<T: Clone + Eq + Hash, U: Clone + Eq + Hash>(
        &self,
        train_set: &Dataset<T, U>,
        valid_set: &Dataset<T, U>,
    ) -> Recommender<T, U> {
        self.fit(train_set, Some(valid_set), false)
    }

    fn fit<T: Clone + Eq + Hash, U: Clone + Eq + Hash>(
        &self,
        train_set: &Dataset<T, U>,
        valid_set: Option<&Dataset<T, U>>,
        implicit: bool,
    ) -> Recommender<T, U> {
        let factors = self.factors as usize;

        let mut user_map = Map::new();
        let mut item_map = Map::new();
        let mut rated = HashMap::new();

        let mut row_inds = Vec::with_capacity(train_set.len());
        let mut col_inds = Vec::with_capacity(train_set.len());
        let mut values = Vec::with_capacity(train_set.len());

        let mut cui = Vec::new();
        let mut ciu = Vec::new();

        for rating in train_set.iter() {
            let u = user_map.add(rating.user_id.clone());
            let i = item_map.add(rating.item_id.clone());

            if implicit {
                if u == cui.len() {
                    cui.push(Vec::new())
                }

                if i == ciu.len() {
                    ciu.push(Vec::new())
                }

                let confidence = 1.0 + self.alpha * rating.value;
                cui[u].push((i, confidence));
                ciu[i].push((u, confidence));
            } else {
                row_inds.push(u);
                col_inds.push(i);
                values.push(rating.value);
            }

            rated.entry(u).or_insert_with(HashSet::new).insert(i);
        }

        let users = user_map.len();
        let items = item_map.len();

        let global_mean = if implicit {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        };

        let end_range = if implicit { 0.01 } else { 0.1 };

        let mut prng = match self.seed {
            Some(s) => Prng::from_seed(s),
            None => Prng::new(),
        };

        let user_factors = create_factors(users, factors, &mut prng, end_range);
        let item_factors = create_factors(items, factors, &mut prng, end_range);

        let mut recommender = Recommender {
            user_map,
            item_map,
            rated,
            global_mean,
            user_factors,
            item_factors,
            normalized_user_factors: RefCell::new(None),
            normalized_item_factors: RefCell::new(None),
        };

        if implicit {
            // conjugate gradient method
            // https://www.benfrederickson.com/fast-implicit-matrix-factorization/

            let regularization = self.regularization.unwrap_or(0.01);

            for iteration in 0..self.iterations {
                least_squares_cg(
                    &cui,
                    &mut recommender.user_factors,
                    &recommender.item_factors,
                    regularization,
                );
                least_squares_cg(
                    &ciu,
                    &mut recommender.item_factors,
                    &recommender.user_factors,
                    regularization,
                );

                if let Some(callback) = &self.callback {
                    let info = FitInfo {
                        iteration: iteration + 1,
                        train_loss: f32::NAN,
                        valid_loss: f32::NAN,
                    };
                    (callback)(info);
                }
            }
        } else {
            // stochastic gradient method with twin learners
            // https://www.csie.ntu.edu.tw/~cjlin/papers/libmf/mf_adaptive_pakdd.pdf
            // algorithm 2

            let learning_rate = self.learning_rate;
            let lambda = self.regularization.unwrap_or(0.1);
            let k = factors;
            let ks = ((k as f32 * 0.08).round() as usize).max(1);

            let mut g_slow: Vec<f32> = vec![1.0; users];
            let mut g_fast: Vec<f32> = vec![1.0; users];
            let mut h_slow: Vec<f32> = vec![1.0; items];
            let mut h_fast: Vec<f32> = vec![1.0; items];

            for iteration in 0..self.iterations {
                let mut train_loss = 0.0;

                // shuffle for each iteration
                for j in sample(&mut prng, train_set.len()) {
                    let u = row_inds[j];
                    let v = col_inds[j];

                    let pu = recommender.user_factors.row_mut(u);
                    let qv = recommender.item_factors.row_mut(v);
                    let e = values[j] - dot(pu, qv);

                    // slow learner
                    let mut g_hat = 0.0;
                    let mut h_hat = 0.0;

                    // fastest inverse square root
                    // https://stackoverflow.com/questions/59081890/is-it-possible-to-write-quakes-fast-invsqrt-function-in-rust
                    let nu = learning_rate * g_slow[u].sqrt().recip();
                    let nv = learning_rate * h_slow[v].sqrt().recip();

                    for d in 0..ks {
                        let gud = -e * qv[d] + lambda * pu[d];
                        let hvd = -e * pu[d] + lambda * qv[d];

                        g_hat += gud * gud;
                        h_hat += hvd * hvd;

                        pu[d] -= nu * gud;
                        qv[d] -= nv * hvd;
                    }

                    g_slow[u] += g_hat / ks as f32;
                    h_slow[v] += h_hat / ks as f32;

                    // fast learner
                    // don't update on first outer iteration
                    if iteration > 0 {
                        let mut g_hat = 0.0;
                        let mut h_hat = 0.0;

                        let nu = learning_rate * g_fast[u].sqrt().recip();
                        let nv = learning_rate * h_fast[v].sqrt().recip();

                        for d in ks..k {
                            let gud = -e * qv[d] + lambda * pu[d];
                            let hvd = -e * pu[d] + lambda * qv[d];

                            g_hat += gud * gud;
                            h_hat += hvd * hvd;

                            pu[d] -= nu * gud;
                            qv[d] -= nv * hvd;
                        }

                        g_fast[u] += g_hat / (k - ks) as f32;
                        h_fast[v] += h_hat / (k - ks) as f32;
                    }

                    train_loss += e * e;
                }

                if let Some(callback) = &self.callback {
                    train_loss = (train_loss / train_set.len() as f32).sqrt();

                    let valid_loss = match &valid_set {
                        Some(ds) => recommender.rmse(ds),
                        None => f32::NAN,
                    };

                    let info = FitInfo {
                        iteration: iteration + 1,
                        train_loss,
                        valid_loss,
                    };
                    (callback)(info);
                }
            }
        }

        recommender
    }
}

impl<'a> Default for RecommenderBuilder<'a> {
    fn default() -> Self {
        Self::new()
    }
}

/// A recommender.
pub struct Recommender<T, U> {
    user_map: Map<T>,
    item_map: Map<U>,
    rated: HashMap<usize, HashSet<usize>>,
    global_mean: f32,
    user_factors: Matrix,
    item_factors: Matrix,

    // use lazy initialization to save memory
    // https://doc.rust-lang.org/std/cell/#implementation-details-of-logically-immutable-methods
    normalized_user_factors: RefCell<Option<Matrix>>,
    normalized_item_factors: RefCell<Option<Matrix>>,
}

impl<T: Clone + Eq + Hash, U: Clone + Eq + Hash> Recommender<T, U> {
    /// Creates a recommender with explicit feedback.
    pub fn fit_explicit(train_set: &Dataset<T, U>) -> Recommender<T, U> {
        RecommenderBuilder::new().fit_explicit(train_set)
    }

    /// Creates a recommender with implicit feedback.
    pub fn fit_implicit(train_set: &Dataset<T, U>) -> Recommender<T, U> {
        RecommenderBuilder::new().fit_implicit(train_set)
    }

    // fit_eval_explicit only defined on builder since not useful without callback

    /// Returns the predicted rating for a specific user and item.
    pub fn predict(&self, user_id: &T, item_id: &U) -> f32 {
        let i = match self.user_map.get(user_id) {
            Some(o) => *o,
            None => return self.global_mean,
        };
        let j = match self.item_map.get(item_id) {
            Some(o) => *o,
            None => return self.global_mean,
        };

        dot(self.user_factors.row(i), self.item_factors.row(j))
    }

    /// Returns recommendations for a user.
    pub fn user_recs(&self, user_id: &T, count: usize) -> Vec<(&U, f32)> {
        let i = match self.user_map.get(user_id) {
            Some(o) => *o,
            None => return Vec::new(),
        };

        let rated = self.rated.get(&i).unwrap();
        let predictions = self.item_factors.dot(self.user_factors.row(i));
        let mut predictions: Vec<_> = predictions.iter().enumerate().collect();
        predictions.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        predictions.truncate(count + rated.len());
        predictions.retain(|v| !rated.contains(&v.0));
        predictions.truncate(count);
        predictions
            .iter()
            .map(|v| (self.item_map.lookup(v.0), *v.1))
            .collect()
    }

    /// Returns recommendations for an item.
    pub fn item_recs(&self, item_id: &U, count: usize) -> Vec<(&U, f32)> {
        similar(
            &self.item_map,
            self.normalized_item_factors(),
            item_id,
            count,
        )
    }

    /// Returns similar users.
    pub fn similar_users(&self, user_id: &T, count: usize) -> Vec<(&T, f32)> {
        similar(
            &self.user_map,
            self.normalized_user_factors(),
            user_id,
            count,
        )
    }

    fn normalized_user_factors(&self) -> &RefCell<Option<Matrix>> {
        self.normalized_user_factors
            .borrow_mut()
            .get_or_insert_with(|| normalize(&self.user_factors));
        &self.normalized_user_factors
    }

    fn normalized_item_factors(&self) -> &RefCell<Option<Matrix>> {
        self.normalized_item_factors
            .borrow_mut()
            .get_or_insert_with(|| normalize(&self.item_factors));
        &self.normalized_item_factors
    }

    /// Returns user ids.
    pub fn user_ids(&self) -> &Vec<T> {
        self.user_map.ids()
    }

    /// Returns item ids.
    pub fn item_ids(&self) -> &Vec<U> {
        self.item_map.ids()
    }

    /// Returns factors for a specific user.
    pub fn user_factors(&self, user_id: &T) -> Option<&[f32]> {
        self.user_map
            .get(user_id)
            .map(|o| self.user_factors.row(*o))
    }

    /// Returns factors for a specific item.
    pub fn item_factors(&self, item_id: &U) -> Option<&[f32]> {
        self.item_map
            .get(item_id)
            .map(|o| self.item_factors.row(*o))
    }

    /// Returns the global mean.
    pub fn global_mean(&self) -> f32 {
        self.global_mean
    }

    /// Calculates the root mean square error for a dataset.
    pub fn rmse(&self, data: &Dataset<T, U>) -> f32 {
        (data
            .iter()
            .map(|r| (self.predict(&r.user_id, &r.item_id) - r.value).powf(2.0))
            .sum::<f32>()
            / data.len() as f32)
            .sqrt()
    }
}

fn least_squares_cg(cui: &[Vec<(usize, f32)>], x: &mut Matrix, y: &Matrix, regularization: f32) {
    let cg_steps = 3;

    // calculate YtY
    let factors = x.cols;
    let mut yty = Matrix::new(factors, factors);
    for i in 0..factors {
        for j in 0..factors {
            yty.data[i * factors + j] = (0..y.rows)
                .map(|k| y.data[k * factors + i] * y.data[k * factors + j])
                .sum();
        }
    }
    for i in 0..factors {
        yty.data[i * factors + i] += regularization;
    }

    for (u, row_vec) in cui.iter().enumerate() {
        // start from previous iteration
        let xi = x.row_mut(u);

        // calculate residual r = (YtCuPu - (YtCuY.dot(Xu), without computing YtCuY
        let mut r = yty.dot(xi);
        neg(&mut r);
        for (i, confidence) in row_vec {
            scaled_add(
                &mut r,
                confidence - (confidence - 1.0) * dot(y.row(*i), xi),
                y.row(*i),
            );
        }

        let mut p = r.clone();
        let mut rsold = dot(&r, &r);

        for _ in 0..cg_steps {
            // calculate Ap = YtCuYp - without actually calculating YtCuY
            let mut ap = yty.dot(&p);
            for (i, confidence) in row_vec {
                scaled_add(&mut ap, (confidence - 1.0) * dot(y.row(*i), &p), y.row(*i));
            }

            // standard CG update
            let alpha = rsold / dot(&p, &ap);
            scaled_add(xi, alpha, &p);
            scaled_add(&mut r, -alpha, &ap);
            let rsnew = dot(&r, &r);

            if rsnew < 1e-20 {
                break;
            }

            let rs = rsnew / rsold;
            for (pi, ri) in p.iter_mut().zip(&r) {
                *pi = ri + rs * (*pi);
            }
            rsold = rsnew;
        }
    }
}

fn create_factors(rows: usize, cols: usize, prng: &mut Prng, end_range: f32) -> Matrix {
    let mut m = Matrix::new(rows, cols);
    for v in m.data.iter_mut() {
        *v = (prng.next() as f32) * end_range;
    }
    m
}

fn similar<'a, T: Clone + Eq + Hash>(
    map: &'a Map<T>,
    norm_factors: &RefCell<Option<Matrix>>,
    id: &T,
    count: usize,
) -> Vec<(&'a T, f32)> {
    let i = match map.get(id) {
        Some(o) => *o,
        None => return Vec::new(),
    };

    let borrowed = norm_factors.borrow();
    let norm_factors = borrowed.as_ref().unwrap();

    let predictions = norm_factors.dot(norm_factors.row(i));
    let mut predictions: Vec<_> = predictions.iter().enumerate().collect();
    predictions.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    predictions.truncate(count + 1);
    predictions.retain(|v| v.0 != i);
    predictions.truncate(count);
    predictions
        .iter()
        .map(|v| (map.lookup(v.0), *v.1))
        .collect()
}

fn normalize(factors: &Matrix) -> Matrix {
    let mut res = factors.clone();

    for row in res.data.chunks_exact_mut(res.cols) {
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in row {
                *v /= norm;
            }
        }
    }

    res
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(ai, bi)| ai * bi).sum()
}

fn scaled_add(x: &mut [f32], a: f32, v: &[f32]) {
    for (xi, vi) in x.iter_mut().zip(v) {
        *xi += a * vi;
    }
}

fn neg(x: &mut [f32]) {
    for v in x {
        *v = -(*v);
    }
}

fn sample(prng: &mut Prng, n: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    // Fisher–Yates shuffle
    for i in (1..=n - 1).rev() {
        let j = (prng.next() * (i as f64 + 1.0)) as usize;
        (v[i], v[j]) = (v[j], v[i]);
    }
    v
}
