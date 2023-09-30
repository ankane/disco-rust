#![doc = include_str!("../README.md")]

mod dataset;
mod map;
mod matrix;
mod prng;
mod recommender;

pub use dataset::Dataset;
pub use recommender::{FitInfo, Recommender, RecommenderBuilder};

#[cfg(test)]
mod tests {
    use crate::{Dataset, Recommender, RecommenderBuilder};

    #[test]
    fn test_rated() {
        let mut data = Dataset::new();
        data.push(1, "A", 1.0);
        data.push(1, "B", 1.0);
        data.push(1, "C", 1.0);
        data.push(1, "D", 1.0);
        data.push(2, "C", 1.0);
        data.push(2, "D", 1.0);
        data.push(2, "E", 1.0);
        data.push(2, "F", 1.0);

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
        let mut data = Dataset::new();
        data.push(1, "A", 1.0);
        data.push(1, "B", 1.0);
        data.push(2, "C", 1.0);

        let recommender = Recommender::fit_implicit(&data);
        let item_ids = recommender
            .item_recs(&"A", 5)
            .iter()
            .map(|v| v.0)
            .collect::<Vec<&&str>>();
        assert_eq!(item_ids, vec![&"B", &"C"]);
    }

    #[test]
    fn test_ids() {
        let mut data = Dataset::new();
        data.push(1, "A", 1.0);
        data.push(1, "B", 1.0);
        data.push(2, "B", 1.0);

        let recommender = Recommender::fit_implicit(&data);
        assert_eq!(recommender.user_ids(), &vec![1, 2]);
        assert_eq!(recommender.item_ids(), &vec!["A", "B"]);
    }

    #[test]
    fn test_factors() {
        let mut data = Dataset::new();
        data.push(1, "A", 1.0);
        data.push(1, "B", 1.0);
        data.push(2, "B", 1.0);

        let recommender = RecommenderBuilder::new().factors(20).fit_implicit(&data);

        assert_eq!(recommender.user_factors(&1).unwrap().len(), 20);
        assert_eq!(recommender.item_factors(&"A").unwrap().len(), 20);

        assert_eq!(recommender.user_factors(&3), None);
        assert_eq!(recommender.item_factors(&"C"), None);
    }

    #[test]
    fn test_callback_explicit() {
        let mut data = Dataset::new();
        data.push(1, "A", 1.0);

        RecommenderBuilder::new()
            .callback(|info| {
                assert!((1..=20).contains(&info.iteration));
                assert!(!info.train_loss.is_nan());
                assert!(info.valid_loss.is_nan());
            })
            .fit_explicit(&data);
    }

    #[test]
    fn test_callback_implicit() {
        let mut data = Dataset::new();
        data.push(1, "A", 1.0);

        RecommenderBuilder::new()
            .callback(|info| {
                assert!((1..=20).contains(&info.iteration));
                assert!(info.train_loss.is_nan());
                assert!(info.valid_loss.is_nan());
            })
            .fit_implicit(&data);
    }

    #[test]
    fn test_callback_explicit_eval() {
        let mut data = Dataset::new();
        data.push(1, "A", 1.0);

        RecommenderBuilder::new()
            .callback(|info| {
                assert!((1..=20).contains(&info.iteration));
                assert!(!info.train_loss.is_nan());
                assert!(!info.valid_loss.is_nan());
            })
            .fit_eval_explicit(&data, &data);
    }
}
