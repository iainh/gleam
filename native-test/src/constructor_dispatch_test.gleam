pub type Colour {
  Red
  Green
  Blue
  Wobble
}

pub fn describe(colour: Colour) -> String {
  case colour {
    Red -> "#ff0000"
    Green -> "#00ff00"
    Blue -> "#0000ff"
    Wobble -> "???"
  }
}

fn guarded(colour: Colour) -> String {
  case colour {
    Red if False -> "nope"
    Red if True -> "matched"
    colour -> describe(colour)
  }
}

fn sparse(colour: Colour) -> String {
  case colour {
    Red -> "red"
    Wobble -> "wobble"
    Green -> "green"
    Blue -> "blue"
  }
}

pub fn run() {
  assert describe(Red) == "#ff0000"
  assert describe(Wobble) == "???"
  assert guarded(Red) == "matched"
  assert guarded(Blue) == "#0000ff"
  assert sparse(Green) == "green"
}
