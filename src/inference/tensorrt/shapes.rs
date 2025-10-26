use anyhow::{Context, Result};
use cxx::let_cxx_string;

use crate::tensorrt::bridge::ffi;

use super::constants::{
    BOARD_INPUT_NAME, POLICY_OUTPUT_NAME, TENSORRT_FLOAT_DATATYPE, VALUE_OUTPUT_NAME,
};

pub struct TensorShapes {
    pub input_elements_per_item: usize,
    pub value_elements_per_item: usize,
    pub policy_elements_per_item: usize,
    pub policy_shape_without_batch: Vec<usize>,
}

impl TensorShapes {
    pub fn from_engine(engine: &cxx::UniquePtr<ffi::TrtEngine>) -> Result<Self> {
        let_cxx_string!(board_name = BOARD_INPUT_NAME);
        let_cxx_string!(value_name = VALUE_OUTPUT_NAME);
        let_cxx_string!(policy_name = POLICY_OUTPUT_NAME);

        let board_shape = ffi::get_tensor_shape(engine, &board_name)
            .context("Failed to get board tensor shape")?;
        let value_shape = ffi::get_tensor_shape(engine, &value_name)
            .context("Failed to get value tensor shape")?;
        let policy_shape = ffi::get_tensor_shape(engine, &policy_name)
            .context("Failed to get policy tensor shape")?;

        ensure_float_tensor(engine, &board_name)?;
        ensure_float_tensor(engine, &value_name)?;
        ensure_float_tensor(engine, &policy_name)?;

        let policy_shape_without_batch = shape_without_batch(&policy_shape);
        if policy_shape_without_batch.len() != 3 {
            panic!(
                "Expected policy tensor to have 3 dimensions (excluding batch), got {}",
                policy_shape_without_batch.len()
            );
        }

        Ok(Self {
            input_elements_per_item: product_without_batch(&board_shape),
            value_elements_per_item: product_without_batch(&value_shape),
            policy_elements_per_item: product_without_batch(&policy_shape),
            policy_shape_without_batch,
        })
    }
}

fn ensure_float_tensor(
    engine: &cxx::UniquePtr<ffi::TrtEngine>,
    tensor_name: &cxx::CxxString,
) -> Result<()> {
    let dtype = ffi::get_tensor_dtype(engine, tensor_name).with_context(|| {
        format!(
            "Failed to get dtype for tensor {}",
            tensor_name.to_string_lossy()
        )
    })?;
    if dtype != TENSORRT_FLOAT_DATATYPE {
        panic!(
            "Expected tensor {} to have float dtype, got {}",
            tensor_name.to_string_lossy(),
            dtype
        );
    }
    Ok(())
}

fn product_without_batch(shape: &[i32]) -> usize {
    if shape.is_empty() {
        panic!("Tensor shape must include batch dimension");
    }
    if shape[0] != -1 && shape[0] != 1 {
        panic!("Tensor batch dimension must be dynamic (-1) or 1");
    }
    shape[1..]
        .iter()
        .map(|&d| {
            if d <= 0 {
                panic!("Tensor dimensions must be positive");
            }
            d as usize
        })
        .product()
}

fn shape_without_batch(shape: &[i32]) -> Vec<usize> {
    if shape.is_empty() {
        panic!("Tensor shape must include batch dimension");
    }
    shape
        .iter()
        .skip(1)
        .map(|&d| {
            if d <= 0 {
                panic!("Tensor dimensions must be positive");
            }
            d as usize
        })
        .collect()
}
