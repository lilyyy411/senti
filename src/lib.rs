//! Const-compatible, compile-time bounded, (mostly) sentinel-terminated arrays, bounded C-strings, and various assorted FFI
//! utilities. Yatta!
//!
//! This crate contains random nonsense that I use for FFI-related code. Many of these types are meant to be used especially
//! in hand-written FFI bindings to make ensuring safety easier.
use std::{
    any::{Any, type_name},
    ffi::{c_char, c_int},
    fmt::{Debug, Display},
    hash::Hash,
    marker::PhantomData,
    mem::MaybeUninit,
};
pub mod cstring;
pub mod ptr;
pub mod senti;

/// A boolean value represented as a typical C-style enum, using way more storage than necessary
#[repr(C)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Bool {
    #[default]
    False = 0,
    True = 1,
}

impl Bool {
    #[inline(always)]
    pub const fn new(b: bool) -> Self {
        if b { Bool::True } else { Bool::False }
    }
    #[inline(always)]
    pub const fn as_bool(self) -> bool {
        matches!(self, Self::True)
    }
}
impl From<bool> for Bool {
    fn from(value: bool) -> Self {
        Self::new(value)
    }
}
impl From<Bool> for bool {
    fn from(value: Bool) -> Self {
        value.as_bool()
    }
}

/// Reserved bytes for a struct. These essentially just act as padding
/// for things that may be added in the future.
///
/// This struct has a [`PartialEq`] implementation that always returns equal
/// since all instances of the struct act as the same (and we don't want to ruin
/// `PartialEq` derives)
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Reserved<const N: usize> {
    _reserved: [MaybeUninit<u8>; N],
}

impl<const N: usize> PartialEq for Reserved<N> {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}
impl<const N: usize> Eq for Reserved<N> {}

impl<const N: usize> Debug for Reserved<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Reserved")
    }
}

/// A wrapper around a C-style enum type that may or may not have valid data.
///
/// Used for when the enum value may have more values added in future versions
/// of a library or you just cannot be sure that the library won't give you random garbage
/// like a [`Bool`] with a value of `2`
#[repr(transparent)]
#[derive(Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct MaybeInvalid<T: CEnum>(pub(crate) c_int, pub(crate) PhantomData<T>);

impl<T: CEnum> MaybeInvalid<T> {
    /// Attempts to get the inner value, returning [`None`]
    /// if the value is not valid for the type.
    pub fn get(self) -> Result<T, ValidationError<T>> {
        T::try_from(self.0)
    }

    /// Gets the inner value without checking for validity.
    ///
    /// Currently, in debug mode, this does perform a check
    /// and panics if the value is invalid.
    ///
    /// # Safety
    /// The caller must ensure that the value is valid.
    #[track_caller] // track_caller because we panic in debug builds
    pub unsafe fn get_unchecked(self) -> T {
        // SAFETY: caller must ensure this is valid.
        unsafe { T::convert_unchecked(self.0) }
    }

    pub fn new(data: T) -> Self {
        Self(data.into_value(), PhantomData)
    }
}

impl<T: Debug + CEnum> Debug for MaybeInvalid<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Ok(v) = self.get() {
            v.fmt(f)
        } else {
            f.debug_tuple("Invalid").field(&self.0).finish()
        }
    }
}
/// A trait marking that a type is a C-style enum that can be validated
///
/// # Safety
/// The imp
pub unsafe trait CEnum: NoUninit + TryFrom<c_int, Error = ValidationError<Self>> {
    #[doc(hidden)]
    unsafe fn convert_unchecked(x: c_int) -> Self;
    #[doc(hidden)]
    fn into_value(self) -> c_int;
}

/// The error for a failed conversion from a c_int to an enum
#[derive(Debug, Clone, Copy)]
pub struct ValidationError<T>(c_int, PhantomData<T>);

impl<T> ValidationError<T> {
    #[doc(hidden)]
    pub fn new(x: c_int) -> Self {
        Self(x, PhantomData)
    }
}
impl<T: Any> Display for ValidationError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} is an invalid value for {}", self.0, type_name::<T>())
    }
}

impl<T: Debug + 'static> std::error::Error for ValidationError<T> {}

#[macro_export]
macro_rules! c_enum {
    {
        $(
            #[$meta:meta]
        )*
        $vis:vis enum $name:ident {
            $(
                $(#[$var_meta:meta])*
                $variant:ident $(= $value:expr)?
            ),* $(,)?
        }
    } => {
        $(#[$meta])*
        #[repr(C)]
        $vis enum $name {
            $(
                $(#[$var_meta])*
                $variant $(= $value)?
            ),*
        }
        impl TryFrom<std::ffi::c_int> for $name {
            type Error = $crate::ffi_util::ValidationError<Self>;
            fn try_from(x: std::ffi::c_int) -> ::std::result::Result<Self, Self::Error> {
                #![allow(non_upper_case_globals)]
            $(
                const $variant: ::std::ffi::c_int = $name::$variant as _;
            )*
                ::std::result::Result::Ok(match x {
                    $(
                        $variant => Self::$variant,
                    )*
                    __failed => return ::std::result::Result::Err($crate::ValidationError::new(__failed))
                })
            }
        }
        unsafe impl $crate::ffi_util::CEnum for $name {

            #[inline(always)]
            #[track_caller]
            unsafe fn convert_unchecked(x: std::ffi::c_int) -> Self {
                if cfg!(debug_assertions) {
                    Self::try_from(x).expect("Invalid data passed to `get_unchecked`")
                } else {
                    unsafe { std::mem::transmute(x) }
                }
            }
            fn into_value(self) -> c_int {
                self as _
            }
        }
        unsafe impl $crate::bytemuck::NoUninit for $name {}
    };
}

pub use bytemuck;
use bytemuck::NoUninit;
// pub(crate) use c_enum;

/// Compares the bitwise representation representation of 2 values for equality
/// in a const context
pub const fn are_bitwise_equal<T: NoUninit>(x: T, y: T) -> bool {
    let mut x = (&raw const x).cast::<u8>();
    let mut y = (&raw const y).cast::<u8>();
    let mut i = 0;
    let mut res = true;
    unsafe {
        // Don't short-circuit. `Senti` is meant for small primitives
        // and if we short-circuit, LLVM will likely try to do something
        // stupid... Although for medium T llvm does recognize it as a
        while i < size_of::<T>() {
            res &= *x == *y;
            x = x.add(1);
            y = y.add(1);
            i += 1;
        }
        res
    }
}

pub const fn transpose_mu<T, const N: usize>(x: MaybeUninit<[T; N]>) -> [MaybeUninit<T>; N] {
    // SAFETY: this is always safe since both have the same layout and can both be uninitialized
    unsafe { std::mem::transmute_copy(&x) }
}
/// Converts a slice of [`c_char`]s into a byte slice
pub const fn c_char_to_bytes(x: &[c_char]) -> &[u8] {
    const {
        assert!(
            size_of::<c_char>() == size_of::<u8>(),
            "What kind of degenerate platform are you on"
        )
    };
    // SAFETY: this is always safe since we can be sure that the layout is the same
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), x.len()) }
}

/// Converts an exclusive slice of [`c_char`]s into an exclusive byte slice
pub const fn c_char_to_bytes_mut(x: &mut [c_char]) -> &mut [u8] {
    const {
        assert!(
            size_of::<c_char>() == size_of::<u8>(),
            "What kind of degenerate platform are you on"
        )
    };
    // SAFETY: this is always safe since we can be sure that the layout is the same
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), x.len()) }
}

/// Converts a byte slice int a slice of [`c_char`]s
pub const fn bytes_to_c_char(x: &[u8]) -> &[c_char] {
    const {
        assert!(
            size_of::<c_char>() == size_of::<u8>(),
            "What kind of degenerate platform are you on"
        )
    };
    // SAFETY: this is always safe since we can be sure that the layout is the same
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), x.len()) }
}

/// Converts a [`Vec<c_char>`] into a [`Vec<u8>`] by transferring ownership
pub fn c_char_to_byte_vec(mut x: Vec<c_char>) -> Vec<u8> {
    const {
        assert!(
            size_of::<c_char>() == size_of::<u8>(),
            "What kind of degenerate platform are you on"
        )
    };
    // SAFETY: this is always valid since `c_char` and `u8` have the same layout
    unsafe {
        let res = Vec::from_raw_parts(x.as_mut_ptr().cast(), x.len(), x.capacity());
        std::mem::forget(x);
        res
    }
}
#[cfg(test)]
mod test {
    use std::os::raw::c_char;

    use crate::senti::Senti;

    #[test]
    fn doesnt_explode() {
        let mut x: Senti<c_char, 8> = Senti::from_slice_truncate(&[1, 2, 3, 4, 5, 6, 7, 8, 0]);
        assert_eq!(x.to_slice(), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(x.to_slice_with_terminator(), None);
        assert_eq!(
            x.to_slice_with_terminator_or_full(),
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(x.to_slice_mut(), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            x.to_slice_with_terminator_or_full_mut(),
            [1, 2, 3, 4, 5, 6, 7, 8]
        );
        // assert!(Senti::from_slice::<_, 4>(&[1, 2, 3, 4, 5, 6, 7, 8, 9]));
    }
}
