//! Various raw pointer variants and utilities

use std::{
    cell::UnsafeCell, fmt::Debug, hash::Hash, marker::PhantomData, mem::MaybeUninit, ops::Deref,
    ptr::NonNull,
};

/// An exclusive raw reference to a slice that has "forgotten" its length
/// and needs to "remember" its length for accesses.
/// Used for passing a slice to C code without accidentally extending lifetimes.
///
/// This type has the exact same invariants as `&mut [T]` except that it has forgotten
/// its length. It is ABI-compatible with `NonNull<T>` / `&mut T`
#[derive(Debug)]
#[repr(transparent)]
pub struct Buffer<'buf, T>(MutPtr<'buf, T>, PhantomData<&'buf mut [T]>);

impl<'buf, T> Buffer<'buf, T> {
    /// Consume an exclusive slice, converting it into a [`Buffer`] and length pair. This length is guaranteed
    /// to be valid to call [`Buffer::remember`].
    pub const fn new(buf: &'buf mut [T]) -> (Buffer<'buf, T>, usize) {
        let len = buf.len();
        // SAFETY: this is always not null
        let ptr = unsafe { NonNull::new_unchecked(buf.as_mut_ptr()) };
        (unsafe { Self::from_raw(ptr) }, len)
    }

    /// "Remembers" the length of the [`Buffer`] and reconstructs the original slice.
    ///
    /// # Safety
    /// `length` must be <= the length of the original slice
    pub const unsafe fn remember(self, length: usize) -> &'buf mut [T] {
        // SAFETY: the caller must ensure this is valid
        unsafe { std::slice::from_raw_parts_mut(self.0.as_raw_ptr().as_ptr(), length) }
    }

    /// Performs an operation on an a slice by consuming it as a `(buffer, length)` pair
    /// and then reborrows the original slice, intended for FFI stuff where you tend
    /// to do operations on a buffer by slicing i
    #[inline(always)]
    pub fn with<U>(
        slice: &'buf mut [T],
        func: impl for<'a> FnOnce(Buffer<'a, T>, usize) -> U,
    ) -> (&'buf mut [T], U) {
        let (buf, len) = Buffer::new(slice);
        let ptr = buf.0;
        let res = func(buf, len);
        // SAFETY:
        // - We have the right length since we got it from the originating slice
        // - We can construct this `Buffer` safely since:
        //    - The pointer comes from a valid `Buffer`, so it is valid
        //    - We know that result outlives `'buf` since it was derived from an input
        //      that outlives `'buf`
        //    - We have ensured no aliasing can happen for since it is impossible for
        //      `func` to return data that references the buffer because the input to the
        //      function is `'a`, which does not outlive `'buf` and the compiler prevents
        //      the aliasing.
        (
            unsafe { Self::from_raw(ptr.as_raw_ptr()).remember(len) },
            res,
        )
    }

    /// Constructs a buffer from a raw pointer
    /// # Safety
    /// See `std::slice::from_raw_parts`, but the length is provided later
    pub const unsafe fn from_raw(ptr: NonNull<T>) -> Self {
        unsafe { Buffer(MutPtr::from_raw(ptr), PhantomData) }
    }

    /// Gets the underlying raw pointer.
    pub const fn as_mut_ptr(&mut self) -> MutPtr<'_, T> {
        self.0
    }
}

/// A non-nullable raw mutable pointer with a lifetime attached indicating that
/// the allocation it originates from must live for at least `'a`.
/// It is useful for keeping track of lifetimes when the data may not necessarily be
/// meet the requirments of `&mut T`, eg., it may alias, point to partially uninitialized
/// data or not be aligned.
#[repr(transparent)]
pub struct MutPtr<'a, T> {
    ptr: NonNull<T>,
    _lt: PhantomData<&'a MaybeUninit<UnsafeCell<T>>>,
}

impl<'a, T> Clone for MutPtr<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T> Copy for MutPtr<'a, T> {}

impl<'a, T> Debug for MutPtr<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.ptr, f)
    }
}
impl<'a, T> Hash for MutPtr<'a, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ptr.hash(state);
    }
}

impl<'a, T> MutPtr<'a, T> {
    /// Create a raw [`MutPtr`] from an exclusive reference
    ///
    /// # Safety
    /// You must ensure that you do not cause aliasing issues.
    pub const unsafe fn from_mut(r: &'a mut T) -> Self {
        Self {
            ptr: NonNull::from_ref(r),
            _lt: PhantomData,
        }
    }
    /// Create a [`MutPtr`] from a raw pointer.
    ///
    /// # Safety
    /// You must ensure that you don't cause aliasing issues
    /// and that if `ptr` is valid, it is valid for at least `'a`
    /// and that nobody aliases with this `ptr` for `'a`
    pub const unsafe fn from_raw(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            _lt: PhantomData,
        }
    }
    /// Gets the regular raw pointer
    pub const fn as_raw_ptr(self) -> NonNull<T> {
        self.ptr
    }
}

impl<T> Deref for MutPtr<'_, T> {
    type Target = NonNull<T>;
    fn deref(&self) -> &Self::Target {
        unsafe { std::mem::transmute(&self.ptr) }
    }
}

/// A non-nullable raw const pointer with a lifetime attached indicating that
/// the allocation it originates from must live for at least `'a`.
/// It is useful for keeping track of lifetimes when the data may not necessarily be
/// meet the requirments of `&T`, eg., it may point to partially uninitialized
/// data or not be aligned.
#[repr(transparent)]
pub struct Ptr<'a, T> {
    ptr: NonNull<T>,
    _lt: PhantomData<&'a MaybeUninit<T>>,
}

impl<'a, T> Clone for Ptr<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T> Copy for Ptr<'a, T> {}

impl<'a, T> Debug for Ptr<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.ptr, f)
    }
}
impl<'a, T> Hash for Ptr<'a, T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ptr.hash(state);
    }
}

impl<'a, T> Ptr<'a, T> {
    /// Create a raw [`Ptr`] from an shared reference
    pub const fn from_ref(r: &'a T) -> Self {
        Self {
            ptr: NonNull::from_ref(r),
            _lt: PhantomData,
        }
    }
    /// Create a [`Ptr`] from a raw pointer.
    ///
    /// # Safety
    /// You must ensure that if `ptr` is valid, it is valid for at least `'a`
    /// and that nobody aliases with this `ptr` for `'a`
    pub const unsafe fn from_raw(ptr: NonNull<T>) -> Self {
        Self {
            ptr,
            _lt: PhantomData,
        }
    }

    /// Gets the underlying raw pointer
    pub const fn as_raw_ptr(self) -> NonNull<T> {
        self.ptr
    }
}

impl<T> Deref for Ptr<'_, T> {
    type Target = NonNull<T>;
    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}
