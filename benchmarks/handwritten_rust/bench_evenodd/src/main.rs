fn count_even_odd(n: i64) -> (i64, i64, i64, i64) {
    let mut evens = 0;
    let mut odds = 0;
    let mut even_sum = 0;
    let mut odd_sum = 0;
    for i in 0..n {
        if i & 1 == 0 {
            evens += 1;
            even_sum += i;
        } else {
            odds += 1;
            odd_sum += i;
        }
    }
    (evens, odds, even_sum, odd_sum)
}

fn main() {
    let (e1, o1, es1, os1) = count_even_odd(7_000_000);
    println!("n=7000000 evens={e1} odds={o1} even_sum={es1} odd_sum={os1}");
    let (e2, o2, es2, os2) = count_even_odd(34_500_000);
    println!("n=34500000 evens={e2} odds={o2} even_sum={es2} odd_sum={os2}");
}
