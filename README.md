# Disco Rust

🔥 Recommendations for Rust using collaborative filtering

- Supports user-based and item-based recommendations
- Works with explicit and implicit feedback
- Uses high-performance matrix factorization

🎉 Zero dependencies

[![Build Status](https://github.com/ankane/disco-rust/actions/workflows/build.yml/badge.svg)](https://github.com/ankane/disco-rust/actions)

## Installation

Add this line to your application’s `Cargo.toml` under `[dependencies]`:

```toml
discorec = "0.2"
```

## Getting Started

Prep your data in the format `user_id, item_id, value`

```rust
use discorec::{Dataset, Recommender};

let mut data = Dataset::new();
data.push("user_a", "item_a", 5.0);
data.push("user_a", "item_b", 3.5);
data.push("user_b", "item_a", 4.0);
```

IDs can be integers, strings, or any other hashable data type

```rust
data.push(1, "item_a".to_string(), 5.0);
```

If users rate items directly, this is known as explicit feedback. Fit the recommender with:

```rust
let recommender = Recommender::fit_explicit(&data);
```

If users don’t rate items directly (for instance, they’re purchasing items or reading posts), this is known as implicit feedback. Use `1.0` or a value like number of purchases or page views for the dataset, and fit the recommender with:

```rust
let recommender = Recommender::fit_implicit(&data);
```

Get user-based recommendations - “users like you also liked”

```rust
recommender.user_recs(&user_id, 5);
```

Get item-based recommendations - “users who liked this item also liked”

```rust
recommender.item_recs(&item_id, 5);
```

Get predicted ratings for a specific user and item

```rust
recommender.predict(&user_id, &item_id);
```

Get similar users

```rust
recommender.similar_users(&user_id, 5);
```

## Examples

### MovieLens

Download the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/).

Add these lines to your application’s `Cargo.toml` under `[dependencies]`:

```toml
csv = "1"
serde = { version = "1", features = ["derive"] }
```

And use:

```rust
use csv::ReaderBuilder;
use discorec::{Dataset, RecommenderBuilder};
use serde::Deserialize;
use std::fs::File;

#[derive(Debug, Deserialize)]
struct Row {
    user_id: i32,
    item_id: i32,
    rating: f32,
}

fn main() {
    let mut train_set = Dataset::new();
    let mut valid_set = Dataset::new();

    let file = File::open("u.data").unwrap();
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b'\t')
        .from_reader(file);
    for (i, record) in rdr.records().enumerate() {
        let row: Row = record.unwrap().deserialize(None).unwrap();
        let dataset = if i < 80000 { &mut train_set } else { &mut valid_set };
        dataset.push(row.user_id, row.item_id, row.rating);
    }

    let recommender = RecommenderBuilder::new()
        .factors(20)
        .fit_explicit(&train_set);
    println!("RMSE: {:?}", recommender.rmse(&valid_set));
}
```

## Storing Recommendations

Save recommendations to your database.

Alternatively, you can store only the factors and use a library like [pgvector-rust](https://github.com/pgvector/pgvector-rust). See an [example](https://github.com/pgvector/pgvector-rust/blob/master/examples/disco/src/main.rs).

## Algorithms

Disco uses high-performance matrix factorization.

- For explicit feedback, it uses the [stochastic gradient method with twin learners](https://www.csie.ntu.edu.tw/~cjlin/papers/libmf/mf_adaptive_pakdd.pdf)
- For implicit feedback, it uses the [conjugate gradient method](https://www.benfrederickson.com/fast-implicit-matrix-factorization/)

Specify the number of factors and iterations

```rust
RecommenderBuilder::new()
    .factors(8)
    .iterations(20)
    .fit_explicit(&train_set);
```

## Progress

Pass a callback to show progress

```rust
RecommenderBuilder::new()
    .callback(|info| println!("{:?}", info))
    .fit_explicit(&train_set);
```

Note: `train_loss` and `valid_loss` are not available for implicit feedback

## Validation

Pass a validation set with explicit feedback

```rust
RecommenderBuilder::new()
    .callback(|info| println!("{:?}", info))
    .fit_eval_explicit(&train_set, &valid_set);
```

The loss function is RMSE

## Cold Start

Collaborative filtering suffers from the [cold start problem](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems)). It’s unable to make good recommendations without data on a user or item, which is problematic for new users and items.

```rust
recommender.user_recs(&new_user_id, 5); // returns empty array
```

There are a number of ways to deal with this, but here are some common ones:

- For user-based recommendations, show new users the most popular items
- For item-based recommendations, make content-based recommendations

## Reference

Get ids

```rust
recommender.user_ids();
recommender.item_ids();
```

Get the global mean

```rust
recommender.global_mean();
```

Get factors

```rust
recommender.user_factors(&user_id);
recommender.item_factors(&item_id);
```

## References

- [A Learning-rate Schedule for Stochastic Gradient Methods to Matrix Factorization](https://www.csie.ntu.edu.tw/~cjlin/papers/libmf/mf_adaptive_pakdd.pdf)
- [Faster Implicit Matrix Factorization](https://www.benfrederickson.com/fast-implicit-matrix-factorization/)

## History

View the [changelog](https://github.com/ankane/disco-rust/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/disco-rust/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/disco-rust/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/disco-rust.git
cd disco-rust
cargo test
```
