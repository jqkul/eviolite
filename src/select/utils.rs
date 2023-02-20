use crate::Solution;

pub fn find_best<T>(pop: &[T]) -> &T
where
    T: Solution,
{
    let mut best_idx: usize = 0;
    for i in 1..pop.len() {
        if pop[i].collapsed() > pop[best_idx].collapsed() {
            best_idx = i;
        }
    }
    &pop[best_idx]
}

// Mutate `vec` in place, keeping only the elements at the positions
// specified by `indices`. Clones elements only for duplicate indices.
pub fn retain_indices<T>(vec: &mut Vec<T>, mut indices: Vec<usize>)
where
    T: Clone,
{
    let n_indices = indices.len();
    indices.sort();

    vec.swap(indices[0], 0);
    let mut swap_to: usize = 1;
    let mut i: usize = 1;
    while i < indices.len() {
        if indices[i] == indices[i - 1] {
            indices.push(vec.len());
            vec.push(vec[swap_to - 1].clone());
        } else {
            vec.swap(indices[i], swap_to);
            swap_to += 1;
        }
        i += 1;
    }
    vec.truncate(n_indices);
}

// Invariants: `indices` must be no longer than `slice`
// and not contain any index twice
pub fn bring_indices_to_front<T>(slice: &mut [T], mut indices: Vec<usize>) {
    debug_assert!(indices.len() <= slice.len());

    indices.sort();
    let mut swap_to: usize = 0;
    for idx in indices {
        slice.swap(swap_to, idx);
        swap_to += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retain_indices() {
        let mut myvec = vec!['a', 'b', 'c', 'd', 'e', 'f', 'g'];
        retain_indices(&mut myvec, vec![5, 4, 5, 1]);
        myvec.sort();
        assert_eq!(myvec, vec!['b', 'e', 'f', 'f']);

        let mut myvec2 = vec!['a', 'b', 'c', 'd', 'e', 'f', 'g'];
        retain_indices(&mut myvec2, vec![2, 4, 1, 2, 1, 2, 1]);
        myvec2.sort();
        assert_eq!(myvec2, vec!['b', 'b', 'b', 'c', 'c', 'c', 'e']);

        let mut myvec3 = vec!['a', 'b', 'c', 'd', 'e', 'f', 'g'];
        retain_indices(&mut myvec3, vec![0; 10]);
        assert_eq!(myvec3, vec!['a'; 10]);
    }
}
