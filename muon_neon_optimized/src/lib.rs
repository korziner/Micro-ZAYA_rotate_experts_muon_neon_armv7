//! Optimized Newton-Schulz using ndarray with OpenBLAS backend
//! Uses BLAS for matrix multiplication (NEON-optimized on ARMv7)

use pyo3::prelude::*;
use numpy::{PyReadonlyArray2, PyArray2, PyArrayMethods};
use ndarray::Array2;

const A_COEFF: f32 = 3.4445;
const B_COEFF: f32 = -4.7750;
const C_COEFF: f32 = 2.0315;
const ITERATIONS: usize = 5;

// ─────────────────────────────────────────────────────────────────
// Newton-Schulz implementation using ndarray + BLAS
// ─────────────────────────────────────────────────────────────────

fn newton_schulz_optimized(x: &[f32], m: usize, n: usize) -> Vec<f32> {
    // Convert to ndarray
    let mut current = Array2::from_shape_vec((m, n), x.to_vec())
        .expect("Failed to create array");

    // Normalization
    let norm = current.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-7 {
        let inv_norm = 1.0 / norm;
        current.mapv_inplace(|v| v * inv_norm);
    }

    // 5 iterations of Newton-Schulz
    for _ in 0..ITERATIONS {
        // A = X @ X^T (m x m)
        // This uses BLAS (OpenBLAS) for optimized matrix multiplication
        let a = current.dot(&current.t());
        
        // A² = A @ A (m x m)
        let a2 = a.dot(&a);

        // B = B_COEFF * A + C_COEFF * A² (element-wise)
        let mut b = Array2::zeros((m, m));
        // Use zip_mut_with instead of zip_with for mutable operations
        b.zip_mut_with(&a, |b_val, &a_val| {
            *b_val = B_COEFF * a_val;
        });
        let a2_scaled = a2.mapv(|v| C_COEFF * v);
        b = b + a2_scaled;

        // B @ X (m x n) - uses BLAS
        let bx = b.dot(&current);

        // X_new = A_COEFF * X + B @ X
        // Use zip for element-wise operations
        let x_scaled = current.mapv(|v| A_COEFF * v);
        let current_new = x_scaled + bx;

        current = current_new;
    }

    // Convert back to Vec<f32>
    current.into_raw_vec()
}

// ─────────────────────────────────────────────────────────────────
// Python API
// ─────────────────────────────────────────────────────────────────

#[pyfunction]
fn newton_schulz_5steps_optimized<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
) -> Bound<'py, PyArray2<f32>> {
    let x = x.as_array();
    let m = x.shape()[0];
    let n = x.shape()[1];

    let x_flat: Vec<f32> = x.iter().cloned().collect();
    let result_data = newton_schulz_optimized(&x_flat, m, n);

    let result = PyArray2::zeros(py, (m, n), false);
    unsafe {
        result.as_slice_mut().unwrap().copy_from_slice(&result_data);
    }
    result
}

#[pyfunction]
fn check_arch_support() -> String {
    let blas_info = if cfg!(feature = "blas") {
        "with OpenBLAS (NEON-optimized)"
    } else {
        "without BLAS (pure Rust)"
    };
    
    format!(
        "ARMv7 {} + rayon ({} threads)",
        blas_info,
        rayon::current_num_threads()
    )
}

#[pyfunction]
fn benchmark_matmul<'py>(
    _py: Python<'py>,
    size: usize,
) -> PyResult<String> {
    use std::time::Instant;
    use ndarray::Array2;
    
    // Create random matrices
    let a = Array2::<f32>::zeros((size, size));
    let b = Array2::<f32>::zeros((size, size));
    
    let start = Instant::now();
    let _c = a.dot(&b);
    let duration = start.elapsed();
    
    Ok(format!(
        "Matrix multiplication {}x{} took: {:?}",
        size, size, duration
    ))
}

#[pymodule]
fn muon_neon_optimized(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(newton_schulz_5steps_optimized, m)?)?;
    m.add_function(wrap_pyfunction!(check_arch_support, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_matmul, m)?)?;
    Ok(())
}
