use cxx::Exception;
use log::error;

pub fn log_ffi_error<T>(result: std::result::Result<T, Exception>, context: &str) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(err) => {
            error!("{}: {}", context, err);
            None
        }
    }
}
