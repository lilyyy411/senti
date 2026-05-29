use std::{borrow::Cow, ffi::c_char, fmt::Debug, hash::Hash, mem::MaybeUninit, ptr::NonNull};

use bytemuck::NoUninit;

use crate::{are_bitwise_equal, transpose_mu};

/// Represents that a type has a sentinel value, ie., it is used to terminate an array.
/// Bitwise comparisons are used for checking for the terminal value.
/// The sentinel must obviously be a valid instance of the type
pub trait Terminated: NoUninit {
    const SENTINEL: Self;
}

/// An array of up to `N` elements of type `T` with a sentinel value indicating the
/// array length when the entire array is not filled. That is, if `len < N`, the `len`th
/// element in the array is guaranteed to be the sentinel value.
///
/// This type is designed to make dealing with sentinel-terminated C arrays
/// safer and easier. All comparisons are done by *bitwise comparison* and this type is
/// ABI-compatible with `[MaybeUninit<T>; N]`.
///
/// Given that the type is meant to be ABI-compatible with `[MaybeUninit<T>; N]`, most safe operations
/// need to recompute the length. Therefore it is recommended to keep the returned
/// slice around as long as possible to prevent recomputation.
///
/// The current implementation is designed to be const-compatible and therefore may have
/// sub-optimal codegen since the intrinsics for doing the needed comparisons cannot be used
/// in `const` contexts. An optimal implementation for the byte case would require calling
/// `memchr`/`memccpy` for larger arrays, the latter of which is not that good anyway.
/// I believe there are wide variants of the former functions,
/// but they're probably bad. It's even worse for other types.
/// An optimal implementation for other types would require hand-written assembly to escape
/// the Rust abstract machine and read uninitialized / out of bounds memory and the compiler
/// would not be able understand what's going on. No, I will not do `repne scas(b/w/d/q)` on x86-64
/// because that is terribly slow.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Senti<T: Copy, const N: usize> {
    buf: [MaybeUninit<T>; N],
}

impl<T: Terminated, const N: usize> Senti<T, N> {
    /// Creates a terminated array that is empty and has a buffer initialized with `N` terminal elements.
    pub const fn empty() -> Self {
        Self::from_array([T::SENTINEL; N])
    }

    /// Creates a new [`Senti`] by copying from a slice. If the slice cannot
    /// fit into the buffer after accounting for any premature truncation, returns [`None`],
    pub const fn from_slice(slice: &[T]) -> Option<Senti<T, N>> {
        // Quick check before checking the actual data
        if slice.len() <= N {
            Some(unsafe { Self::from_slice_unchecked(slice) })
        } else {
            // If the slice is terminated before N elements, then we can safely copy those
            if let Some(terminated_len) = unsafe {
                Self::terminated_len_from_ptr(NonNull::new_unchecked(slice.as_ptr().cast_mut()))
            } {
                Some(unsafe {
                    Self::from_slice_unchecked(std::slice::from_raw_parts(
                        slice.as_ptr(),
                        terminated_len,
                    ))
                })
            } else {
                None
            }
        }
    }

    /// Creates a new [`Senti`] by copying from a slice. If the `slice.len() > N`,
    /// the result is truncated to the first `N` elements.
    ///
    /// Note that if the slice contains the terminator within the first `N` elements,
    /// the terminated array is effectively truncated, but data is still continued to
    /// be copied from the slice until either the end of the slice or the end of the buffer
    ///
    /// If `slice.len() < N`, the rest of the buffer is guaranteed to be initialized and
    /// filled with the terminator
    pub const fn from_slice_truncate(slice: &[T]) -> Senti<T, N> {
        let mut buf = [T::SENTINEL; N];
        // SAFETY: We write at most `N` elements and the buffer can hold `N` elements and read
        // at most `N` elements from the slice.
        // We also know that `buf` and `slice` can't alias because `buf` is a local variable.
        unsafe {
            buf.as_mut_ptr()
                .copy_from_nonoverlapping(slice.as_ptr(), min(slice.len(), N))
        };
        Self::from_array(buf)
    }

    /// Creates a new [`Senti`] by copying from a slice.
    ///
    /// Note that if the slice contains the terminator within the first `N` elements,
    /// the terminated array is effectively truncated, but data is still continued to
    /// be copied from the slice until either the end of the slice or the end of the buffer
    ///
    /// The remainder of the buffer is guaranteed to be filled with the terminator.
    ///
    /// # Safety
    /// The caller must ensure that `slice.len() <= N`
    pub const unsafe fn from_slice_unchecked(slice: &[T]) -> Senti<T, N> {
        let mut buf = [T::SENTINEL; N];
        // SAFETY: caller must ensure validity
        unsafe {
            buf.as_mut_ptr()
                .copy_from_nonoverlapping(slice.as_ptr(), slice.len())
        };
        Self::from_array(buf)
    }

    /// Creates a [`Senti`] with a new capacity from `self`, preserving the length, by creating a new buffer
    /// and copying elements over. If `M < self.compute_len()` the resulting array is guaranteed to not have a terminator.
    /// All data past the first terminator, or N elements, whatever is first, is considered uninitialized.
    pub const fn extend_or_truncate<const M: usize>(self) -> Senti<T, M> {
        if M == N {
            // SAFETY: We are literally the same type and can be bitwise copied
            return unsafe { std::mem::transmute_copy(&self) };
        }
        if M <= N {
            // SAFETY: `self` is guaranteed to either be terminated or have the first `N` elements initialized
            // and M <= N
            return unsafe { Senti::from_ptr(self.as_nonnull()) };
        }
        // we could do it the less efficient way and do `Senti::from_slice(self.to_slice())`,
        // but that is less efficient than just a straight copy if there isn't a lot of data, especially
        // if size_of::<T> != 1
        let mut buf = [MaybeUninit::uninit(); M];

        // SAFETY: M > N and initialization status is preserved
        unsafe {
            buf.as_mut_ptr()
                .copy_from_nonoverlapping(self.buf.as_ptr(), N);
        };
        // Now guarantee that the new buffer has a terminator no matter the true length
        //
        // SAFETY: M > N, therefore M >= N + 1, meaning that writing at offset N is always
        // in bounds
        unsafe {
            buf.as_mut_ptr().add(N).cast::<T>().write(T::SENTINEL);
        };
        Senti { buf }
    }
    /// Reads a [`Senti`] from a raw pointer, reading elements until it finds a terminator, up to `N` elements.
    /// This is not guaranteed to initialize the remainder of the internal buffer with the terminator.
    ///
    /// # Safety
    /// - `ptr` must be valid for reads of up to `min(terminator_pos, N)` elements
    pub const unsafe fn from_ptr(mut ptr: NonNull<T>) -> Senti<T, N> {
        let mut buf = [MaybeUninit::<T>::uninit(); N];
        let mut buf_ptr = buf.as_mut_ptr();
        let mut i = 0;
        // SAFETY: the caller must guarantee safety
        unsafe {
            // ideally we'd have something like a nice and fast memccpy, but
            // nobody likes that function
            while i < N {
                let elem = *ptr.as_ptr();
                *buf_ptr.cast() = elem;

                if are_bitwise_equal(elem, T::SENTINEL) {
                    break;
                }

                ptr = ptr.add(1);
                buf_ptr = buf_ptr.add(1);

                i += 1
            }
        }
        Senti { buf }
    }

    /// Computes the length of the array including the first terminator if any. This is an O(len) operation.
    pub const fn compute_terminated_len(&self) -> Option<usize> {
        // SAFETY: our invariants say this is valid
        unsafe { Self::terminated_len_from_ptr(self.as_nonnull()) }
    }

    /// Computes the length of the array. This is O(len) operation.
    pub const fn compute_len(&self) -> usize {
        if let Some(len) = self.compute_terminated_len() {
            len + 1
        } else {
            N
        }
    }

    const unsafe fn slice_up_to(&self, len: usize) -> &[T] {
        let ptr = self.as_ptr();
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }

    const unsafe fn slice_up_to_mut(&mut self, len: usize) -> &mut [T] {
        let ptr = self.as_mut_ptr();
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }

    /// Converts the [`Senti`] into a slice including the first terminator if any. This is an `O(len)`
    /// operation since it has to compute the length. The returned slice is guaranteed to have at
    /// least 1 element.
    pub const fn to_slice_with_terminator(&self) -> Option<&[T]> {
        if let Some(len) = self.compute_terminated_len() {
            // SAFETY: we know that the buffer is at least initialized to the terminator
            Some(unsafe { self.slice_up_to(len) })
        } else {
            None
        }
    }

    /// Converts the [`Senti`] into a slice including the first terminator if any.
    /// If there is no terminator, the full slice is still returned. This is an `O(len)`
    /// operation since it has to compute the length. The returned slice is guaranteed to have at
    /// least 1 element.
    pub const fn to_slice_with_terminator_or_full(&self) -> &[T] {
        let len = if let Some(len) = self.compute_terminated_len() {
            len
        } else {
            N
        };
        // SAFETY: If the terminated length is None, that means that all elements in the array are initialized
        unsafe { self.slice_up_to(len) }
    }

    /// Converts the [`Senti`] into a mutable slice including the first terminator if any. This is an `O(len)`
    /// operation since it has to compute the length
    ///
    /// # Safety
    /// The terminator should not be overwritten unless the array would still be valid, ie.
    /// there must either be another terminator or all of the data in the buffer must be valid
    pub const unsafe fn to_slice_with_terminator_mut(&mut self) -> Option<&mut [T]> {
        if let Some(len) = self.compute_terminated_len() {
            // SAFETY: we know that the buffer is at least initialized to the terminator
            Some(unsafe { self.slice_up_to_mut(len) })
        } else {
            None
        }
    }

    /// Converts the [`Senti`] into a slice including the first terminator if any.
    /// If there is no terminator, the full slice is still returned. This is an `O(len)`
    /// operation since it has to compute the length. The returned slice is guaranteed to have at
    /// least 1 element.
    ///
    /// # Safety
    /// The terminator should not be overwritten unless the array would still be valid, ie.
    /// there must either be another terminator or all of the data in the buffer must be valid.
    pub const fn to_slice_with_terminator_or_full_mut(&mut self) -> &mut [T] {
        let len = if let Some(len) = self.compute_terminated_len() {
            len
        } else {
            N
        };
        // SAFETY: If the terminated length is None, that means that all elements in the array are initialized
        unsafe { self.slice_up_to_mut(len) }
    }

    /// Converts the [`Senti`] into a slice up to, but not including, the first terminator.
    /// The returned slice is guaranteed to contain no terminator.
    /// This is an `O(len)` operation since it has to compute the length.
    pub const fn to_slice(&self) -> &[T] {
        let len = if let Some(len) = self.compute_terminated_len() {
            len - 1
        } else {
            N
        };
        // SAFETY: we are valid up until the terminator if there is one and len < terminator_pos
        unsafe { self.slice_up_to(len) }
    }

    /// Converts the [`Senti`] into a slice up to, but not including, the first terminator.
    /// The returned slice is guaranteed to contain no terminator. This is an `O(len)` operation since it
    ///  has to compute the length.
    ///
    /// This is always safe since there is no terminator in the slice; the array cannot grow from mutating
    /// this portion
    pub const fn to_slice_mut(&mut self) -> &mut [T] {
        let len = if let Some(len) = self.compute_terminated_len() {
            len - 1
        } else {
            N
        };
        // SAFETY: we are valid up until the terminator if there is one and len < terminator_pos
        unsafe { self.slice_up_to_mut(len) }
    }

    const unsafe fn terminated_len_from_ptr(ptr: NonNull<T>) -> Option<usize> {
        let mut p = ptr.as_ptr();
        let mut i = 0;
        unsafe {
            while i < N {
                i += 1;
                if are_bitwise_equal(*p, T::SENTINEL) {
                    return Some(i);
                }
                p = p.add(1);
            }
            None
        }
    }

    /// Converts the [`Senti`] into a slice that is guaranteed to be terminated
    /// with the terminator by moving data to the heap if there is no terminator.
    pub fn guarantee_termination(&self) -> Cow<'_, [T]> {
        if let Some(len) = self.compute_terminated_len() {
            Cow::Borrowed(unsafe { self.slice_up_to(len) })
        } else {
            let mut v = unsafe { self.slice_up_to(N) }.to_vec();
            v.reserve_exact(1);
            v.push(T::SENTINEL);
            Cow::Owned(v)
        }
    }
}

impl<T: Copy, const N: usize> Senti<T, N> {
    /// Creates a [`Senti`] from a regular array, effectively truncating the array to the first terminator if there is one.
    /// This is always safe since there is no termination guarantee for `N` elements.
    pub const fn from_array(array: [T; N]) -> Senti<T, N> {
        Self {
            buf: transpose_mu(MaybeUninit::new(array)),
        }
    }

    /// Transmute a [`Senti`] to a new new element type with the same layout and validity
    /// as `T`. This is a very unsafe operation, but useful for casting between repr(transparent)
    /// wrappers.
    ///
    /// A compile-time error is emitted if `size_of::<T>() != size_of::<U>()`
    ///
    /// # Safety
    /// - Both `U` and `T` must have no uninitialized bytes within them, including padding.
    /// - All elements in the buffer must be valid when transmuted to type `U`
    /// - If `U: Terminated`, and there are any uninitialized elements in `self`'s buffer, there must be a
    ///   an element with bitwise equivalence to `U::TERMINAL` before that uninitialized element
    /// - All other invariants of [`Senti`] must hold for `U`
    /// - All other invariants for transmuting `[MaybeUninit<T>; N]` to `[MaybeUninit<U>; N]` must hold
    /// - Basically, just don't do this unless either `U` or `T` is a `repr(transparent)` wrapper around the other
    pub const unsafe fn transmute<U: Copy>(self) -> Senti<U, N> {
        const {
            assert!(
                size_of::<T>() == size_of::<U>(),
                "Cannot transmute between `Senti`s with elements of different sizes"
            );
            assert!(
                size_of::<Senti<T, N>>() == size_of::<Senti<U, N>>(),
                "Cannot transmute between `Senti`s with elements that have padding"
            );
        };
        // SAFETY: the caller must ensure that the transmute is valid. We at least made sure that
        // the sizes match
        unsafe { std::mem::transmute_copy(&self) }
    }

    /// Transmute a [`Senti`] to a new element type `U` that might have a different size
    /// from `T`. This is a very *very* unsafe operation if you're not careful, but it does have its uses.
    ///
    /// A compile-time error if `size_of::<Senti<U, M>>() != size_of::<Self>()`
    /// # Safety
    /// - See [`Self::transmute`] and [`std::mem::transmute`] for details
    pub const unsafe fn transmute_with_new_elem_size<U: Copy, const M: usize>(self) -> Senti<U, M> {
        const {
            assert!(
                size_of::<Senti<T, N>>() == size_of::<Senti<U, N>>(),
                "Cannot transmute between `Senti`s of different sizes"
            );
        };
        // SAFETY: the caller must ensure that the transmute is valid. We at least made sure that
        // the sizes match
        unsafe { std::mem::transmute_copy(&self) }
    }

    /// Gets the entire potentially uninitialized buffer of the [`Senti`].
    pub const fn buffer(&self) -> &[MaybeUninit<T>; N] {
        &self.buf
    }
    /// Gets the entire potentially uninitialized buffer of the [`Senti`].
    /// # Safety
    /// You must never uninitialize any elements before the first terminator
    pub const unsafe fn buffer_mut(&mut self) -> &mut [MaybeUninit<T>; N] {
        &mut self.buf
    }

    /// Gets a pointer to the underlying buffer
    pub const fn as_ptr(&self) -> *const T {
        self.buf.as_ptr().cast()
    }

    /// Gets a mut pointer to the underlying buffer
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.buf.as_mut_ptr().cast()
    }

    /// Gets a non-null pointer to the underlying buffer
    pub const fn as_nonnull(&self) -> NonNull<T> {
        unsafe { NonNull::new_unchecked(self.buf.as_ptr().cast_mut().cast()) }
    }
}

impl<T: Eq + Terminated, const N: usize> Eq for Senti<T, N> {}

impl<T, U, const M: usize, const N: usize> PartialEq<Senti<U, M>> for Senti<T, N>
where
    T: PartialEq + Terminated + PartialEq<U>,
    U: Terminated,
{
    fn eq(&self, other: &Senti<U, M>) -> bool {
        self.to_slice() == other.to_slice()
    }
}

impl<T, U, const N: usize> PartialEq<[U]> for Senti<T, N>
where
    T: PartialEq<U> + Terminated,
{
    fn eq(&self, other: &[U]) -> bool {
        self.to_slice() == other
    }
}

impl<T, U, const M: usize, const N: usize> PartialOrd<Senti<U, M>> for Senti<T, N>
where
    T: Terminated + PartialOrd<U> + PartialEq,
    U: Terminated,
{
    fn partial_cmp(&self, other: &Senti<U, M>) -> Option<std::cmp::Ordering> {
        self.to_slice().iter().partial_cmp(other.to_slice())
    }
}

impl<T, U, const N: usize> PartialOrd<[U]> for Senti<T, N>
where
    T: PartialOrd<U> + Terminated,
{
    fn partial_cmp(&self, other: &[U]) -> Option<std::cmp::Ordering> {
        self.to_slice().iter().partial_cmp(other)
    }
}
impl<T, const N: usize> Ord for Senti<T, N>
where
    T: Ord + Terminated,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_slice().cmp(other.to_slice())
    }
}
impl<T: Hash, const N: usize> Hash for Senti<T, N>
where
    T: Terminated,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.to_slice().hash(state);
    }
}
impl<T, const N: usize> Debug for Senti<T, N>
where
    T: Debug + Terminated,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_slice().fmt(f)
    }
}

// FIXME: implement custom iterators that don't need need to precompute the slice
impl<'a, T, const N: usize> IntoIterator for &'a Senti<T, N>
where
    T: Terminated,
{
    type IntoIter = std::slice::Iter<'a, T>;
    type Item = &'a T;
    fn into_iter(self) -> Self::IntoIter {
        self.to_slice().iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut Senti<T, N>
where
    T: Terminated,
{
    type IntoIter = std::slice::IterMut<'a, T>;
    type Item = &'a mut T;
    fn into_iter(self) -> Self::IntoIter {
        self.to_slice_mut().iter_mut()
    }
}

impl Terminated for c_char {
    const SENTINEL: Self = 0;
}

const fn min(x: usize, y: usize) -> usize {
    if x < y { x } else { y }
}
