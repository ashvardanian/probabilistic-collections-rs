use crate::bit_vec::BitVec;
use crate::{DoubleHasher, SipHasherBuilder};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;

/// A space-efficient probabilistic data structure to test for membership in a set.
///
/// This particular implementation of a bloom filter uses `K` partitions and `K` hash functions.
/// Each hash function maps to a bit in its respective partition. A partitioned bloom filter is
/// more robust than its traditional counterpart, but the memory usage is varies based on how many
/// hash functions you are using.
///
/// # Examples
///
/// ```
/// use probabilistic_collections::bloom::PartitionedBloomFilter;
///
/// let mut filter = PartitionedBloomFilter::<String>::from_item_count(10, 0.01);
///
/// assert!(!filter.contains("foo"));
/// filter.insert("foo");
/// assert!(filter.contains("foo"));
///
/// filter.clear();
/// assert!(!filter.contains("foo"));
///
/// assert_eq!(filter.len(), 98);
/// assert_eq!(filter.bit_count(), 14);
/// assert_eq!(filter.hasher_count(), 7);
/// ```
#[derive(Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Deserialize, Serialize),
    serde(crate = "serde_crate")
)]
pub struct PartitionedBloomFilter<T, B = SipHasherBuilder> {
    bit_vec: BitVec,
    hasher: DoubleHasher<T, B>,
    bit_count: usize,
    hasher_count: usize,
    _marker: PhantomData<T>,
}

impl<T> PartitionedBloomFilter<T> {
    /// Constructs a new, empty `PartitionedBloomFilter` with an estimated max capacity of
    /// `item_count` items, and a maximum false positive probability of `fpp`.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::PartitionedBloomFilter;
    ///
    /// let filter = PartitionedBloomFilter::<String>::from_item_count(10, 0.01);
    /// ```
    pub fn from_item_count(item_count: usize, fpp: f64) -> Self {
        Self::from_item_count_with_hashers(
            item_count,
            fpp,
            [
                SipHasherBuilder::from_entropy(),
                SipHasherBuilder::from_entropy(),
            ],
        )
    }

    /// Constructs a new, empty `PartitionedBloomFilter` with `bit_count` bits per partition, and a
    /// maximum false positive probability of `fpp`.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::PartitionedBloomFilter;
    ///
    /// let filter = PartitionedBloomFilter::<String>::from_bit_count(10, 0.01);
    /// ```
    pub fn from_bit_count(bit_count: usize, fpp: f64) -> Self {
        Self::from_bit_count_with_hashers(
            bit_count,
            fpp,
            [
                SipHasherBuilder::from_entropy(),
                SipHasherBuilder::from_entropy(),
            ],
        )
    }
}

impl<T, B> PartitionedBloomFilter<T, B>
where
    B: BuildHasher,
{
    fn get_hasher_count(fpp: f64) -> usize {
        (1.0 / fpp).log2().ceil() as usize
    }

    /// Constructs a new, empty `PartitionedBloomFilter` with an estimated max capacity of
    /// `item_count` items, a maximum false positive probability of `fpp`, and two hash builders
    /// for double hashing.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::PartitionedBloomFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let filter = PartitionedBloomFilter::<String>::from_item_count_with_hashers(
    ///     10,
    ///     0.01,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// ```
    pub fn from_item_count_with_hashers(
        item_count: usize,
        fpp: f64,
        hash_builders: [B; 2],
    ) -> Self {
        let hasher_count = Self::get_hasher_count(fpp);
        let bit_count =
            (item_count as f64 * fpp.ln() / -(2f64.ln().powi(2)) / (hasher_count as f64)).ceil()
                as usize;
        PartitionedBloomFilter {
            bit_vec: BitVec::new(bit_count * hasher_count),
            hasher: DoubleHasher::with_hashers(hash_builders),
            bit_count,
            hasher_count,
            _marker: PhantomData,
        }
    }

    /// Constructs a new, empty `PartitionedBloomFilter` with `bit_count` bits per partition, a
    /// maximum false positive probability of `fpp`, and two hash builders for double hashing.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::PartitionedBloomFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let filter = PartitionedBloomFilter::<String>::from_bit_count_with_hashers(
    ///     10,
    ///     0.01,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// ```
    pub fn from_bit_count_with_hashers(bit_count: usize, fpp: f64, hash_builders: [B; 2]) -> Self {
        let hasher_count = Self::get_hasher_count(fpp);
        PartitionedBloomFilter {
            bit_vec: BitVec::new(bit_count * hasher_count),
            hasher: DoubleHasher::with_hashers(hash_builders),
            bit_count,
            hasher_count,
            _marker: PhantomData,
        }
    }

    /// Inserts an element into the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::PartitionedBloomFilter;
    ///
    /// let mut filter = PartitionedBloomFilter::<String>::from_item_count(10, 0.01);
    ///
    /// filter.insert("foo");
    /// ```
    pub fn insert<U>(&mut self, item: &U)
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        self.hasher
            .hash(item)
            .take(self.hasher_count)
            .enumerate()
            .for_each(|(index, hash)| {
                let offset = (hash % self.bit_count as u64) + (index * self.bit_count) as u64;
                self.bit_vec.set(offset as usize, true);
            })
    }

    /// Checks if an element is possibly in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::PartitionedBloomFilter;
    ///
    /// let mut filter = PartitionedBloomFilter::<String>::from_item_count(10, 0.01);
    ///
    /// assert!(!filter.contains("foo"));
    /// filter.insert("foo");
    /// assert!(filter.contains("foo"));
    /// ```
    pub fn contains<U>(&self, item: &U) -> bool
    where
        T: Borrow<U>,
        U: Hash + ?Sized,
    {
        self.hasher
            .hash(item)
            .take(self.hasher_count)
            .enumerate()
            .all(|(index, hash)| {
                let offset = (hash % self.bit_count as u64) + (index * self.bit_count) as u64;
                self.bit_vec[offset as usize]
            })
    }

    /// Returns the number of bits in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::PartitionedBloomFilter;
    ///
    /// let filter = PartitionedBloomFilter::<String>::from_item_count(10, 0.01);
    ///
    /// assert_eq!(filter.len(), 98);
    /// ```
    pub fn len(&self) -> usize {
        self.bit_vec.len()
    }

    /// Returns `true` if the bloom filter is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::PartitionedBloomFilter;
    ///
    /// let filter = PartitionedBloomFilter::<String>::from_item_count(10, 0.01);
    ///
    /// assert!(!filter.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of bits in each partition in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::PartitionedBloomFilter;
    ///
    /// let filter = PartitionedBloomFilter::<String>::from_item_count(10, 0.01);
    ///
    /// assert_eq!(filter.bit_count(), 14);
    /// ```
    pub fn bit_count(&self) -> usize {
        self.bit_count
    }

    /// Returns the number of hash functions used by the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::PartitionedBloomFilter;
    ///
    /// let filter = PartitionedBloomFilter::<String>::from_item_count(10, 0.01);
    ///
    /// assert_eq!(filter.hasher_count(), 7);
    /// ```
    pub fn hasher_count(&self) -> usize {
        self.hasher_count
    }

    /// Clears the bloom filter, removing all elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::PartitionedBloomFilter;
    ///
    /// let mut filter = PartitionedBloomFilter::<String>::from_item_count(10, 0.01);
    ///
    /// filter.insert("foo");
    /// filter.clear();
    ///
    /// assert!(!filter.contains("foo"));
    /// ```
    pub fn clear(&mut self) {
        self.bit_vec.set_all(false)
    }

    /// Returns the number of set bits in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::PartitionedBloomFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let mut filter = PartitionedBloomFilter::<String>::from_item_count_with_hashers(
    ///     10,
    ///     0.01,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// filter.insert("foo");
    ///
    /// assert_eq!(filter.count_ones(), 7);
    /// ```
    pub fn count_ones(&self) -> usize {
        self.bit_vec.count_ones()
    }

    /// Returns the number of unset bits in the bloom filter.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::PartitionedBloomFilter;
    /// use probabilistic_collections::SipHasherBuilder;
    ///
    /// let mut filter = PartitionedBloomFilter::<String>::from_item_count_with_hashers(
    ///     10,
    ///     0.01,
    ///     [SipHasherBuilder::from_seed(0, 0), SipHasherBuilder::from_seed(1, 1)],
    /// );
    /// filter.insert("foo");
    ///
    /// assert_eq!(filter.count_zeros(), 91);
    /// ```
    pub fn count_zeros(&self) -> usize {
        self.bit_vec.count_zeros()
    }

    /// Returns the estimated false positive probability of the bloom filter. This value will
    /// increase as more items are added.
    ///
    /// This is a fairly rough estimate as it takes the overall fill ratio of all
    /// partitions instead of considering each partition individually.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::PartitionedBloomFilter;
    ///
    /// let mut filter = PartitionedBloomFilter::<String>::from_item_count(100, 0.01);
    /// assert!(filter.estimated_fpp() < std::f64::EPSILON);
    ///
    /// filter.insert("foo");
    /// assert!(filter.estimated_fpp() > std::f64::EPSILON);
    /// assert!(filter.estimated_fpp() < 0.01);
    /// ```
    pub fn estimated_fpp(&self) -> f64 {
        let single_fpp = self.bit_vec.count_ones() as f64 / self.bit_vec.len() as f64;
        single_fpp.powi(self.hasher_count as i32)
    }

    /// Returns a reference to the bloom filter's hasher builders.
    ///
    /// # Examples
    ///
    /// ```
    /// use probabilistic_collections::bloom::PartitionedBloomFilter;
    ///
    /// let filter = PartitionedBloomFilter::<String>::from_item_count(100, 0.01);
    /// let hashers = filter.hashers();
    /// ```
    pub fn hashers(&self) -> &[B; 2] {
        self.hasher.hashers()
    }
}

#[cfg(test)]
mod tests {
    use super::PartitionedBloomFilter;
    use crate::util::tests::{hash_builder_1, hash_builder_2};

    #[test]
    fn test_from_item_count() {
        let mut filter = PartitionedBloomFilter::<String>::from_item_count_with_hashers(
            10,
            0.01,
            [hash_builder_1(), hash_builder_2()],
        );

        assert!(!filter.contains("foo"));
        filter.insert("foo");
        assert!(filter.contains("foo"));
        assert_eq!(filter.count_ones(), 7);
        assert_eq!(filter.count_zeros(), 91);

        filter.clear();
        assert!(!filter.contains("foo"));

        assert_eq!(filter.len(), 98);
        assert_eq!(filter.bit_count(), 14);
        assert_eq!(filter.hasher_count(), 7);
    }

    #[test]
    fn test_from_bit_count() {
        let mut filter = PartitionedBloomFilter::<String>::from_bit_count_with_hashers(
            10,
            0.01,
            [hash_builder_1(), hash_builder_2()],
        );

        assert!(!filter.contains("foo"));
        filter.insert("foo");
        assert!(filter.contains("foo"));
        assert_eq!(filter.count_ones(), 7);
        assert_eq!(filter.count_zeros(), 63);

        filter.clear();
        assert!(!filter.contains("foo"));

        assert_eq!(filter.len(), 70);
        assert_eq!(filter.bit_count(), 10);
        assert_eq!(filter.hasher_count(), 7);
    }

    #[test]
    fn test_estimated_fpp() {
        let mut filter = PartitionedBloomFilter::<String>::from_item_count_with_hashers(
            100,
            0.01,
            [hash_builder_1(), hash_builder_2()],
        );
        assert!(filter.estimated_fpp() < std::f64::EPSILON);

        filter.insert("foo");

        let expected_fpp = (7f64 / 959f64).powi(7);
        assert!((filter.estimated_fpp() - expected_fpp).abs() < std::f64::EPSILON);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_ser_de() {
        let mut filter = PartitionedBloomFilter::<String>::from_item_count(100, 0.01);
        filter.insert("foo");

        let serialized_filter = bincode::serialize(&filter).unwrap();
        let de_filter: PartitionedBloomFilter<String> =
            bincode::deserialize(&serialized_filter).unwrap();

        assert!(de_filter.contains("foo"));
        assert_eq!(filter.bit_vec, de_filter.bit_vec);
        assert_eq!(filter.bit_count, de_filter.bit_count);
        assert_eq!(filter.hashers(), filter.hashers());
    }
}
