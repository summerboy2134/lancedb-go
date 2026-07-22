// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The LanceDB Authors

//! Core FFI infrastructure and result types

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;

pub const NATIVE_SEGMENT_WIRE_VERSION: u32 = 1;
pub const NATIVE_RUNTIME_VERSION: &str =
    "lancedb=0.31.0;lance=8.0.0;arrow=58.0.0;rust=1.91.0;native-segment-wire=1";

pub const FFI_ERROR_NONE: c_int = 0;
pub const FFI_ERROR_OPERATION: c_int = 1;
pub const FFI_ERROR_PANIC: c_int = 2;

/// Result type for C interface
#[repr(C)]
pub struct SimpleResult {
    pub success: bool,
    pub error_message: *mut c_char,
    pub error_code: c_int,
    pub runtime_version: *mut c_char,
}

impl SimpleResult {
    pub fn ok() -> Self {
        Self {
            success: true,
            error_message: ptr::null_mut(),
            error_code: FFI_ERROR_NONE,
            runtime_version: CString::new(NATIVE_RUNTIME_VERSION).unwrap().into_raw(),
        }
    }

    pub fn error(msg: String) -> Self {
        let c_msg =
            CString::new(msg).unwrap_or_else(|_| CString::new("Invalid error message").unwrap());
        Self {
            success: false,
            error_message: c_msg.into_raw(),
            error_code: FFI_ERROR_OPERATION,
            runtime_version: CString::new(NATIVE_RUNTIME_VERSION).unwrap().into_raw(),
        }
    }

    pub fn panic(msg: String) -> Self {
        let mut result = Self::error(msg);
        result.error_code = FFI_ERROR_PANIC;
        result
    }
}

/// Convert C string to Rust string
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn from_c_str(s: *const c_char) -> Result<String, Box<dyn std::error::Error>> {
    if s.is_null() {
        return Err("Null pointer".into());
    }
    let c_str = unsafe { CStr::from_ptr(s) };
    Ok(c_str.to_str()?.to_string())
}

/// Initialize the simple LanceDB library
#[no_mangle]
pub extern "C" fn simple_lancedb_init() -> c_int {
    env_logger::try_init().ok();
    log::info!("Simple LanceDB Go bindings initialized");
    0
}

/// Free a SimpleResult
#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn simple_lancedb_result_free(result: *mut SimpleResult) {
    if result.is_null() {
        return;
    }
    unsafe {
        let result = Box::from_raw(result);
        if !result.error_message.is_null() {
            let _ = CString::from_raw(result.error_message);
        }
        if !result.runtime_version.is_null() {
            let _ = CString::from_raw(result.runtime_version);
        }
    }
}

/// Free a C string allocated by the library
#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn simple_lancedb_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(s);
    }
}

/// Free a byte buffer allocated by the library.
#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn simple_lancedb_free_bytes(data: *mut u8, len: usize) {
    if data.is_null() {
        return;
    }
    unsafe {
        let slice = ptr::slice_from_raw_parts_mut(data, len);
        let _ = Box::<[u8]>::from_raw(slice);
    }
}
