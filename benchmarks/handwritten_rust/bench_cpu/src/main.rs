fn fib_recursive(n: i64) -> i64 {
    if n <= 1 {
        n
    } else {
        fib_recursive(n - 1) + fib_recursive(n - 2)
    }
}

fn heavy_loop(n: i64) -> i64 {
    let mut total: i64 = 0;
    for i in 0..n {
        total += (i * 37 + 11) % 1009;
        total %= 1_000_000_007;
    }
    total
}

fn main() {
    let r = fib_recursive(32);
    println!("{r}");
    let h = heavy_loop(10_000_000);
    println!("{h}");
}
