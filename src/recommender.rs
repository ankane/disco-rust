use crate::map::Map;
use crate::Dataset;
use ndarray::{Array, Array2, ArrayView1, Axis};
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;
use sprs::{CsMat, TriMat};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

#[derive(Debug)]
pub struct FitInfo {
    pub iteration: u32,
    pub train_loss: f32,
    pub valid_loss: f32,
}

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

    pub fn factors(&mut self, value: u32) -> &mut Self {
        self.factors = value;
        self
    }

    pub fn iterations(&mut self, value: u32) -> &mut Self {
        self.iterations = value;
        self
    }

    pub fn regularization(&mut self, value: f32) -> &mut Self {
        self.regularization = Some(value);
        self
    }

    pub fn learning_rate(&mut self, value: f32) -> &mut Self {
        self.learning_rate = value;
        self
    }

    pub fn alpha(&mut self, value: f32) -> &mut Self {
        self.alpha = value;
        self
    }

    pub fn callback<C: 'a + Fn(FitInfo)>(&mut self, value: C) -> &mut Self {
        self.callback = Some(Box::new(value));
        self
    }

    pub fn seed(&mut self, value: u64) -> &mut Self {
        self.seed = Some(value);
        self
    }

    pub fn fit_explicit<T: Clone + Eq + Hash, U: Clone + Eq + Hash>(&self, train_set: &Dataset<T, U>) -> Recommender<T, U> {
        self.fit(train_set, None, false)
    }

    pub fn fit_implicit<T: Clone + Eq + Hash, U: Clone + Eq + Hash>(&self, train_set: &Dataset<T, U>) -> Recommender<T, U> {
        self.fit(train_set, None, true)
    }

    pub fn fit_eval_explicit<T: Clone + Eq + Hash, U: Clone + Eq + Hash>(&self, train_set: &Dataset<T, U>, valid_set: &Dataset<T, U>) -> Recommender<T, U> {
        self.fit(train_set, Some(valid_set), false)
    }

    fn fit<T: Clone + Eq + Hash, U: Clone + Eq + Hash>(&self, train_set: &Dataset<T, U>, valid_set: Option<&Dataset<T, U>>, implicit: bool) -> Recommender<T, U> {
        let factors = self.factors as usize;

        let mut user_map = Map::new();
        let mut item_map = Map::new();
        let mut rated = HashMap::new();

        let mut row_inds = Vec::with_capacity(train_set.len());
        let mut col_inds = Vec::with_capacity(train_set.len());
        let mut values = Vec::with_capacity(train_set.len());

        for rating in train_set.iter() {
            let u = user_map.add(rating.user_id.clone());
            let i = item_map.add(rating.item_id.clone());

            row_inds.push(u);
            col_inds.push(i);
            values.push(if implicit { 1.0 + self.alpha * rating.value } else { rating.value });

            rated.entry(u).or_insert_with(HashSet::new).insert(i);
        }

        let users = user_map.len();
        let items = item_map.len();

        let global_mean = if implicit {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        };

        let end_range = if implicit {
            0.01
        } else {
            0.1
        };

        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let uniform: Uniform<f32> = Uniform::new(0.0, end_range);
        let mut rand_fn = || uniform.sample(&mut rng);

        let user_factors = Array::from_shape_simple_fn((users, factors), &mut rand_fn);
        let item_factors = Array::from_shape_simple_fn((items, factors), &mut rand_fn);

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

            let cui = TriMat::from_triplets(
                (users, items),
                row_inds,
                col_inds,
                values,
            ).to_csr();

            // equivalent to transposing csr
            let ciu = cui.to_csc();

            for iteration in 0..self.iterations {
                least_squares_cg(&cui, &mut recommender.user_factors, &recommender.item_factors, regularization);
                least_squares_cg(&ciu, &mut recommender.item_factors, &recommender.user_factors, regularization);

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
                for j in sample(&mut rng, train_set.len(), train_set.len()) {
                    let u = row_inds[j];
                    let v = col_inds[j];

                    let mut pu = recommender.user_factors.row_mut(u);
                    let mut qv = recommender.item_factors.row_mut(v);
                    let e = values[j] - pu.dot(&qv);

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

pub struct Recommender<T, U> {
    user_map: Map<T>,
    item_map: Map<U>,
    rated: HashMap<usize, HashSet<usize>>,
    global_mean: f32,
    user_factors: Array2<f32>,
    item_factors: Array2<f32>,

    // use lazy initialization to save memory
    // https://doc.rust-lang.org/std/cell/#implementation-details-of-logically-immutable-methods
    normalized_user_factors: RefCell<Option<Array2<f32>>>,
    normalized_item_factors: RefCell<Option<Array2<f32>>>,
}

impl<T: Clone + Eq + Hash, U: Clone + Eq + Hash> Recommender<T, U> {
    pub fn fit_explicit(train_set: &Dataset<T, U>) -> Recommender<T, U> {
        RecommenderBuilder::new().fit_explicit(train_set)
    }

    pub fn fit_implicit(train_set: &Dataset<T, U>) -> Recommender<T, U> {
        RecommenderBuilder::new().fit_implicit(train_set)
    }

    // fit_eval_explicit only defined on builder since not useful without callback

    pub fn predict(&self, user_id: &T, item_id: &U) -> f32 {
        let i = match self.user_map.get(user_id) {
            Some(o) => *o,
            None => return self.global_mean,
        };
        let j = match self.item_map.get(item_id) {
            Some(o) => *o,
            None => return self.global_mean,
        };

        self.user_factors.row(i).dot(&self.item_factors.row(j))
    }

    pub fn user_recs(&self, user_id: &T, count: usize) -> Vec<(&U, f32)> {
        let i = match self.user_map.get(user_id) {
            Some(o) => *o,
            None => return Vec::new(),
        };

        let rated = self.rated.get(&i).unwrap();
        let predictions = self.item_factors.dot(&self.user_factors.row(i));
        let mut predictions: Vec<_> = predictions.iter().enumerate().collect();
        predictions.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        predictions.truncate(count + rated.len());
        predictions.retain(|v| !rated.contains(&v.0));
        predictions.truncate(count);
        predictions.iter().map(|v| (self.item_map.lookup(v.0), *v.1) ).collect()
    }

    pub fn item_recs(&self, item_id: &U, count: usize) -> Vec<(&U, f32)> {
        similar(&self.item_map, self.normalized_item_factors(), item_id, count)
    }

    pub fn similar_users(&self, user_id: &T, count: usize) -> Vec<(&T, f32)> {
        similar(&self.user_map, self.normalized_user_factors(), user_id, count)
    }

    fn normalized_user_factors(&self) -> &RefCell<Option<Array2<f32>>> {
        self.normalized_user_factors
            .borrow_mut()
            .get_or_insert_with(|| normalize(&self.user_factors));
        &self.normalized_user_factors
    }

    fn normalized_item_factors(&self) -> &RefCell<Option<Array2<f32>>> {
        self.normalized_item_factors
            .borrow_mut()
            .get_or_insert_with(|| normalize(&self.item_factors));
        &self.normalized_item_factors
    }

    pub fn user_ids(&self) -> &Vec<T> {
        self.user_map.ids()
    }

    pub fn item_ids(&self) -> &Vec<U> {
        self.item_map.ids()
    }

    pub fn user_factors(&self, user_id: &T) -> Option<ArrayView1<f32>> {
        self.user_map.get(user_id).map(|o| self.user_factors.row(*o) )
    }

    pub fn item_factors(&self, item_id: &U) -> Option<ArrayView1<f32>> {
        self.item_map.get(item_id).map(|o| self.item_factors.row(*o) )
    }

    pub fn global_mean(&self) -> f32 {
        self.global_mean
    }

    pub fn rmse(&self, data: &Dataset<T, U>) -> f32 {
        (data.iter().map(|r| (self.predict(&r.user_id, &r.item_id) - r.value).powf(2.0)).sum::<f32>() / data.len() as f32).sqrt()
    }
}

fn least_squares_cg(cui: &CsMat<f32>, x: &mut Array2<f32>, y: &Array2<f32>, regularization: f32) {
    let cg_steps = 3;

    let factors = x.ncols();
    let yty = y.t().dot(y) + regularization * Array::eye(factors);

    for (u, row_vec) in cui.outer_iterator().enumerate() {
        // start from previous iteration
        let mut xi = x.row_mut(u);

        // calculate residual r = (YtCuPu - (YtCuY.dot(Xu), without computing YtCuY
        let mut r = -yty.dot(&xi);
        for (i, confidence) in row_vec.iter() {
            r.scaled_add(confidence - (confidence - 1.0) * y.row(i).dot(&xi), &y.row(i));
        }

        let mut p = r.clone();
        let mut rsold = r.dot(&r);

        for _ in 0..cg_steps {
            // calculate Ap = YtCuYp - without actually calculating YtCuY
            let mut ap = yty.dot(&p);
            for (i, confidence) in row_vec.iter() {
                ap.scaled_add((confidence - 1.0) * y.row(i).dot(&p), &y.row(i));
            }

            // standard CG update
            let alpha = rsold / p.dot(&ap);
            xi.scaled_add(alpha, &p);
            r.scaled_add(-alpha, &ap);
            let rsnew = r.dot(&r);

            if rsnew < 1e-20 {
                break;
            }

            p = &r + (rsnew / rsold) * p;
            rsold = rsnew;
        }
    }
}

fn similar<'a, T: Clone + Eq + Hash>(map: &'a Map<T>, norm_factors: &RefCell<Option<Array2<f32>>>, id: &T, count: usize) -> Vec<(&'a T, f32)> {
    let i = match map.get(id) {
        Some(o) => *o,
        None => return Vec::new(),
    };

    let borrowed = norm_factors.borrow();
    let norm_factors = borrowed.as_ref().unwrap();

    let predictions = norm_factors.dot(&norm_factors.row(i));
    let mut predictions: Vec<_> = predictions.iter().enumerate().collect();
    predictions.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    predictions.truncate(count + 1);
    predictions.retain(|v| v.0 != i);
    predictions.truncate(count);
    predictions.iter().map(|v| (map.lookup(v.0), *v.1)).collect()
}

fn normalize(factors: &Array2<f32>) -> Array2<f32> {
    let norms = (factors * factors).sum_axis(Axis(1)).mapv(f32::sqrt);
    let norms = norms.mapv(|v| if v == 0.0 { 1e-10 } else { v });
    factors / norms.insert_axis(Axis(1))
}
