import constructor_dispatch_test.{type Colour, Blue, Green, Red, describe}

fn sum_prefix(values: List(Int)) -> Int {
  case values {
    [a, b, c] -> a + b + c
    [first, second, ..rest] -> first + second + sum_prefix(rest)
    [single] -> single
    [] -> 0
  }
}

fn match_nested(values: List(Colour)) -> String {
  case values {
    [Red, Green, third, ..] -> describe(third)
    [Blue, ..] -> "blue-first"
    [Red, ..] -> "red-leading"
    [] -> "empty"
    _ -> "other"
  }
}

pub fn run() {
  assert sum_prefix([1, 2, 3]) == 6
  assert sum_prefix([1, 2, 3, 4]) == 10
  assert match_nested([Red, Green, Blue]) == "#0000ff"
  assert match_nested([Blue, Red]) == "blue-first"
  assert match_nested([Red]) == "red-leading"
  assert match_nested([]) == "empty"
  assert match_nested([Green]) == "other"
}
