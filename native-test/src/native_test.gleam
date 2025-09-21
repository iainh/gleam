import gleam/result.{Result, Ok, Error}

pub fn try_fold(list: List(Int), initial: Int, fun: fn(Int, Int) -> Result(Int, Int)) -> Result(Int, Int) {
  case list {
    [] -> Ok(initial)
    [first, ..rest] ->
      case fun(initial, first) {
        Ok(result) -> try_fold(rest, result, fun)
        Error(_) as error -> error
      }
  }
}
