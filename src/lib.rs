#![doc = include_str!("../README.md")]
#![allow(clippy::needless_doctest_main)]

mod dataset;
mod map;
mod matrix;
mod prng;
mod recommender;

pub use recommender::{FitInfo, Recommender, RecommenderBuilder};

#[allow(deprecated)]
pub use dataset::Dataset;

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::path::Path;

    use crate::{Recommender, RecommenderBuilder};

    #[test]
    fn test_explicit() {
        let Some(data) = load_movielens() else {
            return;
        };

        let recommender = RecommenderBuilder::new().factors(20).fit_explicit(&data);

        let recs = recommender.item_recs(&"Star Wars (1977)".to_string(), 5);
        assert_eq!(recs.len(), 5);

        let item_ids = recs.iter().map(|v| v.0).collect::<Vec<_>>();
        assert!(item_ids.contains(&&"Empire Strikes Back, The (1980)".to_string()));
        assert!(item_ids.contains(&&"Return of the Jedi (1983)".to_string()));
        assert!(!item_ids.contains(&&"Star Wars (1977)".to_string()));
        assert!((recs[0].1 - 0.9972).abs() < 0.01);
    }

    #[test]
    fn test_implicit() {
        let Some(data) = load_movielens() else {
            return;
        };

        let recommender = RecommenderBuilder::new().factors(20).fit_implicit(&data);

        let recs = recommender.item_recs(&"Star Wars (1977)".to_string(), 5);
        assert_eq!(recs.len(), 5);

        let item_ids = recs.iter().map(|v| v.0).collect::<Vec<_>>();
        assert!(item_ids.contains(&&"Return of the Jedi (1983)".to_string()));
        assert!(!item_ids.contains(&&"Star Wars (1977)".to_string()));
    }

    #[test]
    fn test_rated() {
        let data = [
            (1, "A", 1.0),
            (1, "B", 1.0),
            (1, "C", 1.0),
            (1, "D", 1.0),
            (2, "C", 1.0),
            (2, "D", 1.0),
            (2, "E", 1.0),
            (2, "F", 1.0),
        ];

        let recommender = Recommender::fit_implicit(&data);

        let mut item_ids = recommender
            .user_recs(&1, 5)
            .iter()
            .map(|v| v.0)
            .collect::<Vec<&&str>>();
        item_ids.sort();
        assert_eq!(item_ids, vec![&"E", &"F"]);

        let mut item_ids = recommender
            .user_recs(&2, 5)
            .iter()
            .map(|v| v.0)
            .collect::<Vec<&&str>>();
        item_ids.sort();
        assert_eq!(item_ids, vec![&"A", &"B"]);
    }

    #[test]
    fn test_item_recs_same_score() {
        let data = [(1, "A", 1.0), (1, "B", 1.0), (2, "C", 1.0)];

        let recommender = Recommender::fit_implicit(&data);
        let item_ids = recommender
            .item_recs(&"A", 5)
            .iter()
            .map(|v| v.0)
            .collect::<Vec<&&str>>();
        assert_eq!(item_ids, vec![&"B", &"C"]);
    }

    #[test]
    fn test_similar_users() {
        let data = [(1, "A", 1.0), (1, "B", 1.0), (2, "B", 1.0)];

        let recommender = Recommender::fit_explicit(&data);
        assert_eq!(recommender.similar_users(&1, 5).len(), 1);
        assert_eq!(recommender.similar_users(&100000, 5).len(), 0);
    }

    #[test]
    fn test_ids() {
        let data = [(1, "A", 1.0), (1, "B", 1.0), (2, "B", 1.0)];

        let recommender = Recommender::fit_implicit(&data);
        assert_eq!(recommender.user_ids(), &vec![1, 2]);
        assert_eq!(recommender.item_ids(), &vec!["A", "B"]);
    }

    #[test]
    fn test_factors() {
        let data = [(1, "A", 1.0), (1, "B", 1.0), (2, "B", 1.0)];

        let recommender = RecommenderBuilder::new().factors(20).fit_implicit(&data);

        assert_eq!(recommender.user_factors(&1).unwrap().len(), 20);
        assert_eq!(recommender.item_factors(&"A").unwrap().len(), 20);

        assert_eq!(recommender.user_factors(&3), None);
        assert_eq!(recommender.item_factors(&"C"), None);
    }

    #[test]
    fn test_validation_set_explicit() {
        let Some(data) = load_movielens() else {
            return;
        };

        let (train_set, valid_set) = data.split_at(80000);
        let recommender = RecommenderBuilder::new()
            .factors(20)
            .fit_eval_explicit(train_set, valid_set);
        let rmse = recommender.rmse(valid_set);
        assert!((rmse - 0.91).abs() < 0.02);
    }

    #[test]
    fn test_user_recs_new_user() {
        let data = [(1, "A", 1.0), (1, "B", 1.0), (2, "B", 1.0)];

        let recommender = Recommender::fit_explicit(data);
        assert_eq!(recommender.user_recs(&1000, 5).len(), 0)
    }

    #[test]
    fn test_predict_new_user() {
        let data = [(1, "A", 1.0), (1, "B", 1.0), (2, "B", 1.0)];

        let recommender = Recommender::fit_explicit(data);
        assert_eq!(recommender.predict(&3, &"A"), recommender.global_mean());
    }

    #[test]
    fn test_predict_new_item() {
        let data = [(1, "A", 1.0), (1, "B", 1.0), (2, "B", 1.0)];

        let recommender = Recommender::fit_explicit(data);
        assert_eq!(recommender.predict(&1, &"C"), recommender.global_mean());
    }

    #[test]
    fn test_callback_explicit() {
        RecommenderBuilder::new()
            .callback(|info| {
                assert!((1..=20).contains(&info.iteration));
                assert!(!info.train_loss.is_nan());
                assert!(info.valid_loss.is_nan());
            })
            .fit_explicit(&[(1, "A", 1.0)]);
    }

    #[test]
    fn test_callback_implicit() {
        RecommenderBuilder::new()
            .callback(|info| {
                assert!((1..=20).contains(&info.iteration));
                assert!(info.train_loss.is_nan());
                assert!(info.valid_loss.is_nan());
            })
            .fit_implicit(&[(1, "A", 1.0)]);
    }

    #[test]
    fn test_callback_explicit_eval() {
        let data = [(1, "A", 1.0)];

        RecommenderBuilder::new()
            .callback(|info| {
                assert!((1..=20).contains(&info.iteration));
                assert!(!info.train_loss.is_nan());
                assert!(!info.valid_loss.is_nan());
            })
            .fit_eval_explicit(&data, &data);
    }

    #[test]
    fn test_data() {
        let data = vec![(1, "A", 1.0)];
        Recommender::fit_implicit(&data);
        Recommender::fit_implicit(data.as_slice());
        Recommender::fit_implicit(data.iter().map(|v| v));
        Recommender::fit_implicit(data);
        RecommenderBuilder::new().fit_eval_explicit([(1, "A", 1.0)], &[(1, "A", 1.0)]);
    }

    fn load_movielens() -> Option<Vec<(i32, String, f32)>> {
        // https://grouplens.org/datasets/movielens/100k/
        let Some(path) = std::env::var("MOVIELENS_100K_PATH").ok() else {
            return None;
        };
        let path = Path::new(&path);

        let mut movies = HashMap::with_capacity(2000);
        let movies_file = File::open(path.join("u.item")).unwrap();
        let rdr = BufReader::new(movies_file);
        for line in rdr.split(b'\n') {
            let line = line.unwrap();
            let line = String::from_utf8_lossy(&line);
            let mut row = line.split('|');
            let id = row.next().unwrap().to_string();
            let name = row.next().unwrap().to_string();
            movies.insert(id, name);
        }

        let mut data = Vec::with_capacity(100000);
        let ratings_file = File::open(path.join("u.data")).unwrap();
        let rdr = BufReader::new(ratings_file);
        for line in rdr.lines() {
            let line = line.unwrap();
            let mut row = line.split('\t');
            let user_id = row.next().unwrap().parse().unwrap();
            let item_id = movies.get(row.next().unwrap()).unwrap().to_string();
            let rating = row.next().unwrap().parse().unwrap();
            data.push((user_id, item_id, rating));
        }

        Some(data)
    }
}
