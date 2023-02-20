mod cached;
pub use cached::Cached;

pub trait IterIndices {
    type Item;
    fn iter_indices<I>(&self, indices: I) -> IndicesIter<Self::Item, I>
    where
        I: Iterator<Item = usize>;
}

impl<T> IterIndices for Vec<T> {
    type Item = T;
    fn iter_indices<I>(&self, indices: I) -> IndicesIter<Self::Item, I>
    where
        I: Iterator<Item = usize>,
    {
        IndicesIter {
            inner: self,
            indices,
        }
    }
}

pub struct IndicesIter<'a, T, I>
where
    I: Iterator<Item = usize>,
{
    inner: &'a [T],
    indices: I,
}

impl<'a, T, I> Iterator for IndicesIter<'a, T, I>
where
    I: Iterator<Item = usize>,
{
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.indices.next().map(|idx| &self.inner[idx])
    }
}

pub trait NFromFunction<T> {
    fn n_from_function(n: usize, f: impl Fn() -> T) -> Self;
}

impl<T> NFromFunction<T> for Vec<T> {
    fn n_from_function(n: usize, f: impl Fn() -> T) -> Self {
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(f());
        }
        v
    }
}
