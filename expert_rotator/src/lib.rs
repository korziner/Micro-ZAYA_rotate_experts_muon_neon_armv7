use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyArray2, PyArrayMethods};
use rayon::prelude::*;

/// GELU аппроксимация
#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())
}

/// Batch matmul: C = A × B с параллелизмом по строкам
fn matmul_batch(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
        let a_row = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a_row[kk] * b[kk * n + j];
            }
            c_row[j] = sum;
        }
    });
}

/// Batch forward всех экспертов в одном matmul
#[pyfunction]
fn forward_experts_batch<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f32>,
    up_all: PyReadonlyArray3<f32>,
    down_all: PyReadonlyArray3<f32>,
    expert_indices: PyReadonlyArray1<i64>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let x_arr = x.as_array();
    let num_tokens = x_arr.shape()[0];
    let dim = x_arr.shape()[1];
    
    let up_arr = up_all.as_array();
    let down_arr = down_all.as_array();
    
    let hidden_dim = up_arr.shape()[2];
    let num_experts_total = up_arr.shape()[0];

    // Преобразуем индексы в Vec
    let indices_vec: Vec<usize> = expert_indices
        .as_array()
        .iter()
        .map(|&x| x as usize)
        .filter(|&idx| idx < num_experts_total)
        .collect();

    let num_active = indices_vec.len();
    if num_active == 0 {
        return Ok(PyArray2::zeros(py, (num_tokens, dim), false));
    }

    let x_flat = x_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("x is not contiguous")
    })?;

    let up_flat = up_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("up_all is not contiguous")
    })?;

    let down_flat = down_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("down_all is not contiguous")
    })?;

    // === BATCH PROCESSING ===
    
    // 1. Конкатенируем up-матрицы всех активных экспертов
    // up_concat: [dim, num_active * hidden_dim]
    let up_concat_size = dim * num_active * hidden_dim;
    let mut up_concat = vec![0.0f32; up_concat_size];
    
    for (idx_pos, &expert_idx) in indices_vec.iter().enumerate() {
        let src_offset = expert_idx * dim * hidden_dim;
        let dst_offset = idx_pos * dim * hidden_dim;
        
        for i in 0..dim {
            for j in 0..hidden_dim {
                up_concat[dst_offset + i * (num_active * hidden_dim) + j] = 
                    up_flat[src_offset + i * hidden_dim + j];
            }
        }
    }
    
    // 2. Batch matmul: x @ up_concat → h_concat
    // h_concat: [num_tokens, num_active * hidden_dim]
    let h_concat_size = num_tokens * num_active * hidden_dim;
    let mut h_concat = vec![0.0f32; h_concat_size];
    
    matmul_batch(x_flat, &up_concat, &mut h_concat, num_tokens, dim, num_active * hidden_dim);
    
    // 3. Применяем GELU ко всему h_concat
    h_concat.par_iter_mut().for_each(|v| *v = gelu(*v));
    
    // 4. Конкатенируем down-матрицы всех активных экспертов
    // down_concat: [num_active * hidden_dim, dim]
    let down_concat_size = num_active * hidden_dim * dim;
    let mut down_concat = vec![0.0f32; down_concat_size];
    
    for (idx_pos, &expert_idx) in indices_vec.iter().enumerate() {
        let src_offset = expert_idx * hidden_dim * dim;
        let dst_offset = idx_pos * hidden_dim * dim;
        
        for i in 0..hidden_dim {
            for j in 0..dim {
                down_concat[dst_offset + i * dim + j] = 
                    down_flat[src_offset + i * dim + j];
            }
        }
    }
    
    // 5. Batch matmul: h_concat @ down_concat → output_concat
    // output_concat: [num_tokens, num_active * dim]
    let output_concat_size = num_tokens * num_active * dim;
    let mut output_concat = vec![0.0f32; output_concat_size];
    
    matmul_batch(&h_concat, &down_concat, &mut output_concat, num_tokens, num_active * hidden_dim, num_active * dim);
    
    // 6. Суммируем выходы всех экспертов и усредняем
    let mut output = vec![0.0f32; num_tokens * dim];
    
    for i in 0..num_tokens {
        for j in 0..dim {
            let mut sum = 0.0f32;
            for e in 0..num_active {
                sum += output_concat[i * (num_active * dim) + e * dim + j];
            }
            output[i * dim + j] = sum / num_active as f32;
        }
    }
    
    // Создаём numpy массив
    let result = PyArray2::zeros(py, (num_tokens, dim), false);
    unsafe {
        result
            .as_slice_mut()
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("Failed to get result slice"))?
            .copy_from_slice(&output);
    }
    
    Ok(result)
}

#[pyfunction]
fn get_num_threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn expert_rotator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(forward_experts_batch, m)?)?;
    m.add_function(wrap_pyfunction!(get_num_threads, m)?)?;
    Ok(())
}
