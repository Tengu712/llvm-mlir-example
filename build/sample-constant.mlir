func.func @ret_sample_contant() -> i32 {
    %ret = "sample.constant"() { value = 13 : i32 } : () -> i32
    return %ret : i32
}
