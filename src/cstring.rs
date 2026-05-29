use std::{
    borrow::Cow,
    ffi::{CStr, CString, c_char},
    fmt::{Debug, Display},
    mem::MaybeUninit,
    ptr::NonNull,
    str::Utf8Error,
};

use crate::{
    bytes_to_c_char, c_char_to_byte_vec, c_char_to_bytes, c_char_to_bytes_mut, senti::Senti,
};

/// A C string that is bounded to a maximum `N` bytes, guaranteed to be null-terminated if `len < N`.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BoundedCString<const N: usize> {
    data: Senti<c_char, N>,
}

impl<const N: usize> PartialEq<CStr> for BoundedCString<N> {
    fn eq(&self, other: &CStr) -> bool {
        self.to_bytes() == other.to_bytes()
    }
}

impl<const N: usize> PartialEq<str> for BoundedCString<N> {
    fn eq(&self, other: &str) -> bool {
        self.to_bytes() == other.as_bytes()
    }
}

impl<const N: usize> Debug for BoundedCString<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let slice = self.to_bytes();
        write!(f, "\"")?;
        for chunk in slice.utf8_chunks() {
            for c in chunk.valid().chars() {
                match c {
                    '\x01'..='\x7f' => write!(f, "{}", (c as u8).escape_ascii())?,
                    _ => write!(f, "{}", c.escape_debug())?,
                }
            }
            write!(f, "{}", chunk.invalid().escape_ascii())?;
        }
        write!(f, "\"")?;
        Ok(())
    }
}

impl<const N: usize> Display for BoundedCString<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // TODO: do this efficiently without allocating
        f.write_str(&self.to_str_lossy())
    }
}

impl<const N: usize> BoundedCString<N> {
    /// Creates a [`BoundedCString`] from a string, truncating the returned string
    /// if it contains a nul byte, returning `None` if the string is too long.
    pub const fn from_str(data: &str) -> Option<Self> {
        Self::from_bytes(data.as_bytes())
    }

    /// Creates a [`BoundedCString`] from a byte slice, truncating the returned string
    /// if it contains a nul byte, returning `None` if the string is too long.
    pub const fn from_bytes(data: &[u8]) -> Option<Self> {
        Self::from_c_chars(bytes_to_c_char(data))
    }

    /// Creates a [`BoundedCString`] from a [`c_char`] slice, truncating the returned string
    /// if it contains a nul byte, returning `None` if the string is too long.
    pub const fn from_c_chars(data: &[c_char]) -> Option<Self> {
        if let Some(data) = Senti::from_slice(data) {
            Some(Self { data })
        } else {
            None
        }
    }

    /// Creates a [`BoundedCString`] from a string, truncating the returned string
    /// if it contains a null byte as well as truncating the string to `N` bytes if it is too long.
    ///
    /// This truncation is guaranteed to preserve the UTF-8 validity of the string. In the case
    /// that a codepoint would be cut in the middle, that final codepoint is cut off.
    pub const fn from_str_truncate(data: &str) -> Self {
        let last_cp = data.floor_char_boundary(N);
        let slice = unsafe { std::slice::from_raw_parts(data.as_ptr(), last_cp) };

        Self {
            // SAFETY: slice is at most length N
            data: unsafe { Senti::from_slice_unchecked(bytes_to_c_char(slice)) },
        }
    }

    /// Creates a [`BoundedCString`] from a byte slice, truncating the returned string
    /// if it contains a null byte as well as truncating the string to `N` bytes if it is too long
    pub const fn from_bytes_truncate(data: &[u8]) -> Self {
        Self::from_c_chars_truncate(bytes_to_c_char(data))
    }

    /// Creates a [`BoundedCString`] from a [`c_char`] slice, truncating the returned string
    /// if it contains a null byte as well as truncating the string to `N` bytes if it is too long
    pub const fn from_c_chars_truncate(data: &[c_char]) -> Self {
        Self {
            data: Senti::from_slice_truncate(data),
        }
    }

    /// Creates a [`BoundedCString`] from a [`CStr`], truncating the returned string
    ///truncating the string to `N` bytes if it is too long.
    pub const fn from_cstr_truncate(data: &CStr) -> Self {
        Self {
            // SAFETY: `data` is guaranteed to be null-terminated or have a len >= N
            data: unsafe { Senti::from_ptr(NonNull::new_unchecked(data.as_ptr().cast_mut())) },
        }
    }

    /// Converts the [`BoundedCString`] into a [`c_char`] slice without the trailing nul byte.
    ///
    /// This is an `O(len)` operation since the length has to be computed.
    pub const fn to_c_chars(&self) -> &[c_char] {
        self.data.to_slice()
    }

    /// Converts the [`BoundedCString`] into a [`c_char`] slice without the trailing nul byte.
    ///
    /// This is an `O(len)` operation since the length has to be computed.
    pub const fn to_c_chars_mut(&self) -> &[c_char] {
        self.data.to_slice()
    }
    /// Converts the [`BoundedCString`] into a byte slice without the trailing nul byte.
    ///
    /// This is an `O(len)` operation since the length has to be computed.
    pub const fn to_bytes(&self) -> &[u8] {
        c_char_to_bytes(self.to_c_chars())
    }

    /// Converts the [`BoundedCString`] into a byte slice without the trailing nul byte.
    ///
    /// This is an `O(len)` operation since the length has to be computed.
    pub const fn to_bytes_mut(&mut self) -> &mut [u8] {
        c_char_to_bytes_mut(self.data.to_slice_mut())
    }

    /// Attempts to convert the [`BoundedCString`] into a byte slice with the trailing nul byte if there is one
    pub const fn to_bytes_with_nul(&self) -> Option<&[u8]> {
        if let Some(slice) = self.to_c_chars_with_nul() {
            Some(c_char_to_bytes(slice))
        } else {
            None
        }
    }

    /// Attempts to convert the [`BoundedCString`] into a byte slice with the trailing nul byte if there is one
    ///
    /// # Safety
    /// The nul terminator should not be overwritten unless the array would still be valid, ie.
    /// there must either be another terminator or all of the data in the buffer must be valid
    pub const unsafe fn to_bytes_with_nul_mut(&mut self) -> Option<&mut [u8]> {
        // SAFETY: the caller must ensure this is valid
        if let Some(slice) = unsafe { self.to_c_chars_with_nul_mut() } {
            Some(c_char_to_bytes_mut(slice))
        } else {
            None
        }
    }
    /// Attempts to convert the [`BoundedCString`] into a [`c_char`] slice with the trailing nul byte if there is one
    pub const fn to_c_chars_with_nul(&self) -> Option<&[c_char]> {
        self.data.to_slice_with_terminator()
    }

    /// Attempts to convert the [`BoundedCString`] into a [`c_char`] slice with the trailing nul byte if there is one
    ///
    /// # Safety
    /// The nul terminator should not be overwritten unless the array would still be valid, ie.
    /// there must either be another terminator or all of the data in the buffer must be valid
    pub const unsafe fn to_c_chars_with_nul_mut(&mut self) -> Option<&mut [c_char]> {
        unsafe { self.data.to_slice_with_terminator_mut() }
    }

    /// Attempts to convert the [`BoundedCString`] into an `&`[`CStr`], returning [`None`]
    /// there is no nul byte. This is an `O(len)` operation since the length has to be computed every time
    pub const fn to_cstr(&self) -> Option<&CStr> {
        if let Some(buf) = self.to_bytes_with_nul() {
            // SAFETY: buf is guaranteed to have exactly 1 terminator
            Some(unsafe { CStr::from_bytes_with_nul_unchecked(buf) })
        } else {
            None
        }
    }
    /// Converts the [`BoundedCString`] into a nul-terminated C-string, moving
    /// the data to the heap if there is no terminator.
    pub fn to_cstring(&self) -> Cow<'_, CStr> {
        // SAFETY: we are guaranteed to have nul termination
        unsafe {
            match self.data.guarantee_termination() {
                Cow::Borrowed(x) => {
                    Cow::Borrowed(CStr::from_bytes_with_nul_unchecked(c_char_to_bytes(x)))
                }
                Cow::Owned(x) => {
                    Cow::Owned(CString::from_vec_with_nul_unchecked(c_char_to_byte_vec(x)))
                }
            }
        }
    }
    /// Returns an empty [`BoundedCString`] with a guaranteed zeroed buffer.
    pub const fn zeroed() -> Self {
        // SAFETY: We are guaranteed to be intialized with all zero bytes
        unsafe { std::mem::zeroed() }
    }

    /// Returns a reference to the underlying buffer. This buffer may be unitialized, but is guaranteed
    /// to be initialized at least until the first null byte
    pub const fn buffer(&self) -> &[MaybeUninit<c_char>; N] {
        self.data.buffer()
    }

    /// Returns an exclusive reference to the underlying buffer.
    ///
    /// # Safety
    /// You must ensure that the buffer is initialized up until the first null byte before accessing the
    /// [`BoundedCString`] again
    pub const unsafe fn buffer_mut(&mut self) -> &mut [MaybeUninit<c_char>; N] {
        unsafe { self.data.buffer_mut() }
    }

    /// Attempts to convert the string to an [`&str`]
    pub const fn try_to_str(&self) -> Result<&str, Utf8Error> {
        std::str::from_utf8(self.to_bytes())
    }

    /// Converts the string to an [`&str`] without checking UTF-8 validity
    ///
    /// # Safety
    /// The internal string must be valid UTF-8
    pub const unsafe fn to_str_unchecked(&self) -> &str {
        // SAFETY: Caller must ensure this is safe
        unsafe { std::str::from_utf8_unchecked(self.to_bytes()) }
    }

    /// Converts the string to an `&mut str` without checking UTF-8 validity
    ///
    /// # Safety
    /// The internal string must be valid UTF-8
    pub const unsafe fn to_str_unchecked_mut(&mut self) -> &mut str {
        // SAFETY: Caller must ensure this is safe
        unsafe { std::str::from_utf8_unchecked_mut(self.to_bytes_mut()) }
    }

    /// Lossily converts the [`BoundedCString`] to a string
    pub fn to_str_lossy(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(self.to_bytes())
    }
}
