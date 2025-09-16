pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_x = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    for v in x.iter_mut() {
        *v = (*v - max_x).exp();
    }

    let sum: f32 = x.iter().sum();
    if sum == 0.0 {
        let p = 1.0 / (x.len() as f32);
        for v in x.iter_mut() {
            *v = p;
        }
    } else {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_inplace() {
        let mut x = [1.0, 2.0, 3.0];
        softmax_inplace(&mut x);
        assert_eq!(x, [0.09003057, 0.24472847, 0.66524094]);
    }
}
