use std::time::{Duration, SystemTime};

pub struct Prng {
    s: [u64; 4],
}

impl Prng {
    pub fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .subsec_nanos()
            .into();
        Self::from_seed(seed)
    }

    pub fn from_seed(seed: u64) -> Self {
        let mut x = seed;
        let mut s = [0; 4];
        for v in &mut s {
            *v = splitmix64(&mut x);
        }
        Self { s }
    }

    // use upper 53 bits as recommended by xoshiro256+ authors
    // detailed explanation: https://lemire.me/blog/2017/02/28/how-many-floating-point-numbers-are-in-the-interval-01/
    pub fn next(&mut self) -> f64 {
        let bits = xoshiro256plus(&mut self.s) >> 11;
        bits as f64 / (1_u64 << 53) as f64
    }
}

// Ported from code written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)
//
// To the extent possible under law, the author has dedicated all copyright
// and related and neighboring rights to this software to the public domain
// worldwide. This software is distributed without any warranty.
//
// See <http://creativecommons.org/publicdomain/zero/1.0/>.
fn xoshiro256plus(s: &mut [u64; 4]) -> u64 {
    let result = s[0].wrapping_add(s[3]);

    let t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;

    s[3] = rotl(s[3], 45);

    result
}

fn rotl(x: u64, k: i32) -> u64 {
    (x << k) | (x >> (64 - k))
}

// Ported from code written in 2015 by Sebastiano Vigna (vigna@acm.org)
//
// To the extent possible under law, the author has dedicated all copyright
// and related and neighboring rights to this software to the public domain
// worldwide. This software is distributed without any warranty.
//
// See <http://creativecommons.org/publicdomain/zero/1.0/>.
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
